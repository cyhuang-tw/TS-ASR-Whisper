import json
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Callable

import lhotse
import pandas as pd
import wandb
import  meeteval
from accelerate.utils import broadcast_object_list
from jiwer import cer, compute_measures
from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.cut.data import DataCut
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging

from data.local_datasets import LhotseLongFormDataset
from data.postprocess import truncate_at_repeating_ngram
from utils.general import supervisions_to_seglst, df_to_seglst, get_cut_recording_id, remove_custom_attributes
from utils.logging_def import get_logger
from utils.wer import calc_wer
from utils.wer_utils import aggregate_wer_metrics, normalize_segment

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

_LOG = get_logger('wer')


def get_metrics(labels: List[str], preds: List[str]):
    metrics = compute_measures(labels, preds)
    return {"cer": cer(labels, preds), **metrics}


def write_wandb_pred(pred_str: List[str], label_str: List[str], rows_to_log: int = 10):
    current_step = wandb.run.step
    columns = ["id", "label_str", "hyp_str"]
    wandb.log(
        {
            f"eval_predictions/step_{int(current_step)}": wandb.Table(
                columns=columns,
                data=[
                    [i, ref, hyp] for i, hyp, ref in
                    zip(range(min(len(pred_str), rows_to_log)), pred_str, label_str)
                ],
            )
        },
        current_step,
    )


def compute_metrics(output_dir: os.path, text_norm: Callable, tokenizer: PreTrainedTokenizer, pred: PredictionOutput,
                    wandb_pred_to_save: int = 500, decode_with_timestamps=False) -> Dict[
    str, float]:
    preds = pred.predictions
    labels = pred.label_ids

    preds[preds == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    pred_str = [text_norm(re.sub(r"\<\|\d+\.\d+\|\>", " ", pred)) for pred in
                tokenizer.batch_decode(preds, skip_special_tokens=True, normalize=False,
                                       decode_with_timestamps=decode_with_timestamps)]
    label_str = [text_norm(re.sub(r"\<\|\d+\.\d+\|\>", " ", label).strip()) for label in
                 tokenizer.batch_decode(labels, skip_special_tokens=True, normalize=False,
                                        decode_with_timestamps=decode_with_timestamps)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    path = f"{output_dir}/predictions.csv"
    df = pd.DataFrame({"label": label_str, "prediction": pred_str})
    df.to_csv(path, index=False)

    # ensure that for jiwer all labels are non empty by replacing empty labels with hyphen
    label_str = [label if label else "-" for label in label_str]

    return get_metrics(label_str, pred_str)


def write_hypothesis_jsons(out_dir, session_id: str,
                           attributed_segments_df: pd.DataFrame,
                           text_normalizer):
    """
    Write hypothesis transcripts for session, to be used for tcp_wer and tc_orc_wer metrics.
    """

    def write_json(df, filename):
        filepath = Path(out_dir) / 'wer' / session_id / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        seglst = df_to_seglst(df)
        seglst = seglst.map(partial(normalize_segment, tn=text_normalizer))
        seglst.dump(filepath)
        return filepath

    # I. hyp file for tcpWER
    tcp_wer_hyp_json = write_json(attributed_segments_df, 'tcp_wer_hyp.json')

    # II. hyp file for tcORC-WER, a supplementary metric for analysis.
    # meeteval.wer.tcorcwer requires a stream ID, which depends on the system.
    # Overlapped words should go into different streams, or appear in one stream while respecting the order
    # in reference. See https://github.com/fgnt/meeteval.
    # In NOTSOFAR we define the streams as the outputs of CSS (continuous speech separation).
    # If your system does not have CSS you need to define the streams differently.
    # For example: for end-to-end multi-talker ASR you might use a single stream.
    # Alternatively, you could use the predicted speaker ID as the stream ID.

    # The wav_file_name column of attributed_segments_df indicates the source CSS stream.
    # Note that the diarization module ensures the words within each segment have a consistent channel.
    df_tcorc = attributed_segments_df.copy()
    # Use factorize to map each unique wav_file_name to an index.
    # meeteval.wer.tcorcwer treats speaker_id field as stream id.
    # df_tcorc = assign_streams(df_tcorc)
    tcorc_wer_hyp_json = write_json(df_tcorc, 'tc_orc_wer_hyp.json')

    return {
        'session_id': session_id,
        'tcp_wer_hyp_json': tcp_wer_hyp_json,
        'tcorc_wer_hyp_json': tcorc_wer_hyp_json,
    }


def parse_string_to_objects(s, duration=None):
    # Regular expression to match the time tokens
    time_pattern = re.compile(r'<\|([\d.]+)\|>')

    # Find all time tokens
    times = time_pattern.findall(s)

    # Split the text using the time tokens to get the text segments
    text_segments = time_pattern.split(s)[1:]  # Ignore the first empty element

    # Create the list of objects with start, end, and text
    objects = []
    for i in range(0, len(times) - 1):
        start_time = float(times[i])
        end_time = float(times[i + 1])
        text = text_segments[2 * i + 1].strip()
        if text:  # Only add if there is some text
            objects.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })

    # Handle trailing text after the last timestamp (no closing timestamp)
    if times and duration is not None:
        last_idx = len(times) - 1
        trailing_text_idx = 2 * last_idx + 1
        if trailing_text_idx < len(text_segments):
            trailing_text = text_segments[trailing_text_idx].strip()
            if trailing_text:
                objects.append({
                    'start': float(times[last_idx]),
                    'end': duration,
                    'text': trailing_text
                })

    return objects


def sot_to_annotation(sot_string, session_id, duration=None):
    """Convert an SOT hypothesis string into a pyannote Annotation.

    Splits on '????' to get per-speaker blocks, assigns synthetic speaker labels
    (spk_0, spk_1, ...), and extracts timed segments via parse_string_to_objects.
    """
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=session_id)
    blocks = sot_string.split("????")
    for spk_idx, block in enumerate(blocks):
        speaker = f"spk_{spk_idx}"
        segments = parse_string_to_objects(block, duration=duration)
        for seg in segments:
            annotation[Segment(seg["start"], seg["end"]), speaker] = speaker
    return annotation


def ref_units_to_annotation(ref_ts_units, session_id):
    """Convert reference transcript units (with timestamps) into a pyannote Annotation.

    Each unit has 'speaker' and 'text' fields; timestamps are extracted from the text
    via parse_string_to_objects.
    """
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=session_id)
    for unit in ref_ts_units:
        speaker = unit["speaker"]
        segments = parse_string_to_objects(unit["text"])
        for seg in segments:
            annotation[Segment(seg["start"], seg["end"]), speaker] = speaker
    return annotation


def ref_cut_to_annotation(cut, session_id):
    """Build reference pyannote Annotation directly from lhotse supervision objects.

    This bypasses the text serialization roundtrip, avoiding silent drops of
    segments whose last timestamp has no closing pair.
    """
    from pyannote.core import Annotation, Segment
    annotation = Annotation(uri=session_id)
    for sup in cut.supervisions:
        if sup.duration > 0:
            annotation[Segment(sup.start, sup.start + sup.duration), sup.speaker] = sup.speaker
    return annotation


def process_session(session_preds, tokenizer, spk_id, cut: DataCut, break_to_characters=False, overflow_margin=5.0):
    session_preds[session_preds == -100] = tokenizer.pad_token_id
    transcript = tokenizer.decode(session_preds, decode_with_timestamps=True,
                                  skip_special_tokens=True)
    segments = parse_string_to_objects(transcript)
    cut_duration = cut.end - cut.start
    for segment in segments:
        if break_to_characters:
            segment['text'] = LhotseLongFormDataset.add_space_between_chars(segment['text'])
        if segment['end'] <= cut_duration + overflow_margin:
            yield {
                'session_id': get_cut_recording_id(cut),
                'start_time': segment['start'] + cut.start,
                'end_time': segment['end'] + cut.start,
                'text': truncate_at_repeating_ngram(segment['text']),
                'speaker_id': spk_id,
                'wav_file_name': "in_mem" if isinstance(cut, MixedCut) else cut.recording.sources[0].source,
            }
        else:
            logger.warning(f"""Detected segment out of bounds of cut. {str({
                'session_id': get_cut_recording_id(cut),
                'start_time': segment['start'] + cut.start,
                'end_time': segment['end'] + cut.start,
                'text': truncate_at_repeating_ngram(segment['text']),
                'speaker_id': spk_id,
                'wav_file_name': "in_mem" if isinstance(cut, MixedCut) else cut.recording.sources[0].source,
            })}""")



def shift_timestamps(cut: lhotse.MonoCut):
    def shift_timestamps_supervision(supervision):
        supervision.start += offset
        supervision.end += offset
        return supervision

    offset = cut.start
    if offset > 0:
        return map(shift_timestamps_supervision, cut.supervisions)
    return cut.supervisions

def save_session_outputs(processed_sessions: dict, current_dir, text_norm, references_cs: CutSet):
    for session_id, outputs in processed_sessions.items():
        attributed_segments_df = pd.DataFrame(outputs)
        write_hypothesis_jsons(
            current_dir, session_id, attributed_segments_df, text_norm)


        if session_id in references_cs:
            gt_cut = references_cs[session_id]
        else:
            gt_cutset = references_cs.filter(lambda c: get_cut_recording_id(c) == session_id)
            if len(gt_cutset) == 0:
                raise ValueError(f"Session {session_id} not found in GT dataset.")
            if len(gt_cutset) > 1:
                raise ValueError(f"Detected more sessions with session id: {session_id}")
            gt_cut = gt_cutset[0]

        remove_custom_attributes(gt_cut)
        filepath = Path(current_dir) / 'wer' / session_id
        # Potentially correct shifted cutsets
        supervisions = shift_timestamps(gt_cut)
        ref_seglst = supervisions_to_seglst(supervisions, session_id)
        ref_seglst = ref_seglst.map(partial(normalize_segment, tn=text_norm))
        ref_seglst.dump(filepath / 'ref.json')


def calculate_tcp_wer(processed_sessions, current_dir, metrics_list,
                      save_visualizations=True,
                      collar=5):
    wer_dfs = []
    for session_id in processed_sessions:
        calc_wer_out = Path(current_dir) / 'wer' / session_id
        out_tcp_file = Path(current_dir) / 'wer' / session_id / 'tcp_wer_hyp.json'
        out_tc_file = Path(current_dir) / 'wer' / session_id / 'tc_orc_wer_hyp.json'
        ref_file = Path(current_dir) / 'wer' / session_id / 'ref.json'

        session_wer: pd.DataFrame = calc_wer(
            calc_wer_out,
            out_tcp_file,
            out_tc_file,
            ref_file,
            collar=collar,
            save_visualizations=save_visualizations,
            metrics_list=metrics_list)
        wer_dfs.append(session_wer)
    return wer_dfs


def compute_longform_metrics(pred, trainer, output_dir, text_norm, metrics_list=None, dataset=None,
                             save_visualizations=True):
    # if not main process, return
    metrics = {}
    if trainer.accelerator.is_main_process:
        if dataset is not None:
            orig_cs = dataset.cset.to_eager()
            references_cs = dataset.references.to_eager()
        else:
            # This doesn't work for test (predict) evaluation.
            # In that case, we pass the dataset argument.
            orig_cs = trainer.eval_dataset.cset.to_eager()
            references_cs = trainer.eval_dataset.references.to_eager()

        processed_sessions = {}
        # Iterate over the predictions and process them
        processed_sessions_ids = set()
        for index, session_preds in enumerate(pred.predictions):
            label_ids = pred.label_ids[index]
            if (label_ids == -100).all():
                continue
            label_ids[label_ids == -100] = trainer.processing_class.pad_token_id
            cut_id, spk_id = trainer.processing_class.decode(label_ids, skip_special_tokens=True).split(",")
            if (cut_id, spk_id) in processed_sessions_ids:
                # In DDP setup sampler can return the same session multiple times
                continue
            try:
                cut = orig_cs[cut_id]  # this will raise StopIteration, if not found
            except Exception as e:
                raise KeyError(f"Key '{cut_id}' not found in dataset, {e}")

            if get_cut_recording_id(cut) not in processed_sessions:
                processed_sessions[get_cut_recording_id(cut)] = []
            processed_sessions[get_cut_recording_id(cut)].extend(
                process_session(session_preds, trainer.processing_class, spk_id, cut,
                                break_to_characters=dataset.break_to_characters if dataset is not None else False)
            )
            processed_sessions_ids.add((cut_id, spk_id))

        # Save the session outputs
        save_session_outputs(processed_sessions, output_dir, text_norm, references_cs)

        # Calculate WER
        wer_dfs = calculate_tcp_wer(processed_sessions, output_dir, collar=5,
                                    save_visualizations=save_visualizations, metrics_list=metrics_list)

        # Save the WER results and calculate the average
        all_session_wer_df = pd.concat(wer_dfs, ignore_index=True)
        all_session_wer_df.to_csv(output_dir + '/all_session_wer.csv')
        metrics = aggregate_wer_metrics(all_session_wer_df, metrics_list)

    metrics = broadcast_object_list([metrics], from_process=0)
    return metrics[0]


# def process_session_sot(session_preds, tokenizer, text_norm):
#     session_preds[session_preds == -100] = tokenizer.pad_token_id
#     transcript = tokenizer.decode(
#         session_preds,
#         decode_with_timestamps=True,
#         skip_special_tokens=True,
#     )
#
#     # Split on speaker markers: one or more '!' followed by whitespace
#     per_spk_transcripts = re.split(r'!+\s+', transcript)
#
#     output = []
#     for t in per_spk_transcripts:
#         if not t.strip():
#             continue
#         output.append(
#             text_norm(truncate_at_repeating_ngram(t.strip()))
#         )
#
#     return output

def process_session_sot(session_preds, tokenizer, text_norm,
                        sot_split_token="????", sot_split_token_id=25629):
    session_preds[session_preds == -100] = tokenizer.pad_token_id

    # Split token IDs by separator BEFORE decoding each speaker block.
    # This avoids the Whisper tokenizer's decode_with_timestamps adding
    # spurious 30s offsets when timestamps restart (go backwards) after
    # a speaker change.
    blocks = []
    current = []
    for tok_id in session_preds.tolist():
        tok_id = int(tok_id)
        if tok_id == sot_split_token_id:
            blocks.append(current)
            current = []
        elif tok_id != tokenizer.pad_token_id:
            current.append(tok_id)
    if current:
        blocks.append(current)

    output = []
    raw_parts = []
    for block_ids in blocks:
        decoded = tokenizer.decode(block_ids, decode_with_timestamps=True,
                                   skip_special_tokens=True)
        raw_parts.append(decoded)
        output.append(text_norm(truncate_at_repeating_ngram(decoded)))

    raw_transcript = sot_split_token.join(raw_parts)
    return output, raw_transcript

def compute_sot_longform_metrics(pred, trainer, output_dir, text_norm, metrics_list=None, dataset=None,
                             save_visualizations=True):
    first_eval_set = next(iter(trainer.eval_dataset.values()))
    # if not main process, return
    metrics = {}
    if trainer.accelerator.is_main_process:
        if dataset is not None:
            references_cs = dataset.references.to_eager()
        else:
            references_cs = trainer.eval_dataset.references.to_eager()

        refs = {}
        hyps = {}
        sot_raw = {}
        ref_sot_raw = {}
        ref_ts_units_by_cut = {}
        # Iterate over the predictions and process them
        processed_sessions_ids = set()
        for index, session_preds in enumerate(pred.predictions):
            label_ids = pred.label_ids[index]
            if (label_ids == -100).all():
                continue
            label_ids[label_ids == -100] = trainer.processing_class.pad_token_id
            cut_id = trainer.processing_class.decode(label_ids, skip_special_tokens=True)
            if cut_id in processed_sessions_ids:
                # In DDP setup sampler can return the same session multiple times
                continue
            session_out, raw_transcript = process_session_sot(session_preds, trainer.processing_class, text_norm)
            ref_units = first_eval_set.get_transcript_units(references_cs[cut_id], use_timestamps=False)
            if "speaker" in first_eval_set.sot_strategy:
                ref_units = first_eval_set.merge_speaker_units(ref_units)
            ref = [item['text'] for item in ref_units]

            # Reference with timestamps for SOT output
            ref_ts_units = first_eval_set.get_transcript_units(references_cs[cut_id], use_timestamps=True)
            if "speaker" in first_eval_set.sot_strategy:
                ref_ts_units = first_eval_set.merge_speaker_units(ref_ts_units)
            ref_sot_raw[cut_id] = "????".join(item['text'] for item in ref_ts_units)
            ref_ts_units_by_cut[cut_id] = ref_ts_units

            if len(session_out) > 20 or len(session_out) > 2 * len(ref):
                print(f"Produced too many speakers in {cut_id}: {session_out}\nClearing session output.")
                session_out = []
            refs[cut_id] = ref
            hyps[cut_id] = session_out
            sot_raw[cut_id] = raw_transcript
            processed_sessions_ids.add(cut_id)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "hyp.json"), "w") as file:
            json.dump(hyps, file)
        with open(os.path.join(output_dir, "ref.json"), "w") as file:
            json.dump(refs, file)
        with open(os.path.join(output_dir, "hyp_sot.json"), "w") as file:
            json.dump(sot_raw, file)
        with open(os.path.join(output_dir, "ref_sot.json"), "w") as file:
            json.dump(ref_sot_raw, file)
        cp_wer = meeteval.wer.wer.cp_word_error_rate_multifile(reference=refs, hypothesis=hyps)
        with open(os.path.join(output_dir, "cp.wer"), "w") as file:
            json.dump(str(cp_wer), file)
        combined = meeteval.wer.combine_error_rates(cp_wer)
        metrics = vars(combined)
        with open(os.path.join(output_dir, "cpwer.json"), "w") as file:
            json.dump({"cpwer": combined.error_rate, "errors": combined.errors,
                       "length": combined.length, "insertions": combined.insertions,
                       "deletions": combined.deletions, "substitutions": combined.substitutions}, file)

        # Build speaker-count mapping
        num_spks_by_cut = {}
        for cut_id in refs:
            cut = references_cs[cut_id]
            num_spks_by_cut[cut_id] = len(set(s.speaker for s in cut.supervisions))

        spk_groups = defaultdict(list)
        for cut_id, n in num_spks_by_cut.items():
            spk_groups[n].append(cut_id)

        # Compute per-speaker-count cpWER
        cpwer_by_nspk = {}
        for n, cut_ids in sorted(spk_groups.items()):
            group_refs = {cid: refs[cid] for cid in cut_ids}
            group_hyps = {cid: hyps[cid] for cid in cut_ids}
            group_cp = meeteval.wer.wer.cp_word_error_rate_multifile(
                reference=group_refs, hypothesis=group_hyps
            )
            group_combined = meeteval.wer.combine_error_rates(group_cp)
            cpwer_by_nspk[n] = {
                "cpwer": group_combined.error_rate,
                "errors": group_combined.errors,
                "length": group_combined.length,
                "insertions": group_combined.insertions,
                "deletions": group_combined.deletions,
                "substitutions": group_combined.substitutions,
                "num_sessions": len(cut_ids),
            }

        metrics["cpwer_by_num_speakers"] = cpwer_by_nspk
        with open(os.path.join(output_dir, "cpwer_by_num_speakers.json"), "w") as f:
            json.dump(cpwer_by_nspk, f, indent=2)

        _LOG.info("=== cpWER by number of speakers ===")
        for n in sorted(cpwer_by_nspk):
            _LOG.info(f"  {n} spk(s): cpWER={cpwer_by_nspk[n]['cpwer']:.4f}  (n={cpwer_by_nspk[n]['num_sessions']})")

        # Speaker confusion matrix: predicted # speakers (rows) vs ground-truth # speakers (cols)
        num_pred_spks_by_cut = {cut_id: len(hyps[cut_id]) for cut_id in hyps}
        all_counts = sorted(set(num_spks_by_cut.values()) | set(num_pred_spks_by_cut.values()))

        # Build raw count matrix
        confusion_counts = {pred_n: {gt_n: 0 for gt_n in all_counts} for pred_n in all_counts}
        for cut_id in hyps:
            gt_n = num_spks_by_cut[cut_id]
            pred_n = num_pred_spks_by_cut[cut_id]
            confusion_counts[pred_n][gt_n] += 1

        # Normalize each column (gt speaker count) to percentages
        col_totals = {gt_n: sum(confusion_counts[pred_n][gt_n] for pred_n in all_counts) for gt_n in all_counts}
        confusion_pct = {}
        for pred_n in all_counts:
            confusion_pct[pred_n] = {}
            for gt_n in all_counts:
                total = col_totals[gt_n]
                confusion_pct[pred_n][gt_n] = round(100.0 * confusion_counts[pred_n][gt_n] / total, 1) if total > 0 else 0.0

        metrics["speaker_confusion_matrix"] = {"counts": confusion_counts, "percent": confusion_pct}
        with open(os.path.join(output_dir, "speaker_confusion_matrix.json"), "w") as f:
            json.dump({"counts": confusion_counts, "percent": confusion_pct, "row": "predicted", "col": "ground_truth"}, f, indent=2)

        # Log confusion matrix as a table
        header = "pred\\gt " + "".join(f"{gt_n:>8}" for gt_n in all_counts)
        _LOG.info("=== Speaker confusion matrix (%, col-normalized) ===")
        _LOG.info(header)
        for pred_n in all_counts:
            row = f"  {pred_n:>5} " + "".join(f"{confusion_pct[pred_n][gt_n]:>7.1f}%" for gt_n in all_counts)
            _LOG.info(row)

        # Compute DER when timestamps are available
        if first_eval_set.use_timestamps:
            from pyannote.metrics.diarization import DiarizationErrorRate
            der_metric = DiarizationErrorRate(collar=0.25)
            der_metrics_by_nspk = {n: DiarizationErrorRate(collar=0.25) for n in spk_groups}
            for cut_id in sot_raw:
                if cut_id not in ref_ts_units_by_cut:
                    continue
                ref_annotation = ref_units_to_annotation(ref_ts_units_by_cut[cut_id], cut_id)
                hyp_annotation = sot_to_annotation(sot_raw[cut_id], cut_id, duration=references_cs[cut_id].duration)
                der_metric(ref_annotation, hyp_annotation)
                n = num_spks_by_cut[cut_id]
                der_metrics_by_nspk[n](ref_annotation, hyp_annotation)
            der_value = abs(der_metric)
            metrics["der"] = der_value
            with open(os.path.join(output_dir, "der.json"), "w") as file:
                json.dump({"der": der_value}, file)

            der_by_nspk = {}
            for n in sorted(der_metrics_by_nspk):
                der_by_nspk[n] = {
                    "der": abs(der_metrics_by_nspk[n]),
                    "num_sessions": len(spk_groups[n]),
                }
            metrics["der_by_num_speakers"] = der_by_nspk
            with open(os.path.join(output_dir, "der_by_num_speakers.json"), "w") as f:
                json.dump(der_by_nspk, f, indent=2)

            _LOG.info("=== DER by number of speakers ===")
            for n in sorted(der_by_nspk):
                _LOG.info(f"  {n} spk(s): DER={der_by_nspk[n]['der']:.4f}  (n={der_by_nspk[n]['num_sessions']})")

    metrics = broadcast_object_list([metrics], from_process=0)
    return metrics[0]
