import torch
from transformers import WhisperTimeStampLogitsProcessor


class WhisperTimeStampLogitsProcessorCustom(WhisperTimeStampLogitsProcessor):

    def __init__(self, generate_config, begin_index, _detect_timestamp_from_logprob=None):
        super().__init__(generate_config, begin_index, _detect_timestamp_from_logprob)
        self.sot_separator_token_id = getattr(generate_config, "sot_separator_token_id", None)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Step 1: Clone scores, suppress <|notimestamps|>
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        # Step 2: Per-sample loop with speaker-block-scoped state
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index:]
            seq = list(sampled_tokens.tolist())

            # Find last separator position and extract current speaker block
            if self.sot_separator_token_id is not None:
                last_sep_pos = None
                for i in range(len(seq) - 1, -1, -1):
                    if seq[i] == self.sot_separator_token_id:
                        last_sep_pos = i
                        break

                if last_sep_pos is not None:
                    current_block = seq[last_sep_pos + 1:]
                    just_after_separator = len(current_block) == 0
                else:
                    current_block = seq
                    just_after_separator = False
            else:
                # Non-SOT: full sequence, no separator handling
                current_block = seq
                just_after_separator = False

            # Handle just-after-separator: force timestamp or EOS
            if just_after_separator:
                scores_processed[k, :self.timestamp_begin] = -float("inf")
                # Restore EOS so the model can stop
                scores_processed[k, self.eos_token_id] = scores[k, self.eos_token_id]
                # Suppress separator (no double-separator)
                scores_processed[k, self.sot_separator_token_id] = -float("inf")
                continue

            # Pairing rules scoped to current_block
            last_was_timestamp = len(current_block) >= 1 and current_block[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(current_block) < 2 or current_block[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:
                    # Two consecutive timestamps (or single timestamp at block start):
                    # text must follow, suppress all timestamps
                    scores_processed[k, self.timestamp_begin:] = -float("inf")
                    # In SOT mode, also suppress separator (text must follow within block)
                    if self.sot_separator_token_id is not None:
                        scores_processed[k, self.sot_separator_token_id] = -float("inf")
                else:
                    # Single end timestamp after text: suppress text tokens below eos
                    scores_processed[k, :self.eos_token_id] = -float("inf")
                    # In SOT mode, restore separator score (speaker switch is valid here)
                    if self.sot_separator_token_id is not None:
                        scores_processed[k, self.sot_separator_token_id] = scores[k, self.sot_separator_token_id]

            # Non-decreasing timestamps scoped to current_block only
            block_tokens = torch.tensor(current_block, device=input_ids.device)
            timestamps = block_tokens[block_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid re-emitting the same timestamp
                    timestamp_last = timestamps[-1] + 1
                scores_processed[k, self.timestamp_begin:timestamp_last] = -float("inf")

        # Step 3: Initial step — force timestamp, restore EOS, apply max_initial_timestamp_index
        if input_ids.shape[1] == self.begin_index:
            scores_processed[:, :self.timestamp_begin] = -float("inf")
            # Restore EOS to allow early exit from silence
            scores_processed[:, self.eos_token_id] = scores[:, self.eos_token_id]

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores_processed[:, last_allowed + 1:] = -float("inf")

        # Step 4: Probability-based timestamp forcing
        logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, :self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores_processed[k, :self.timestamp_begin] = -float("inf")

        return scores_processed
