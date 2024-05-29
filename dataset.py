import traceback
import numpy as np
from torch.utils.data import IterableDataset
import fim
import functools
import torch
import random

DOCUMENT_SPLIT_RATE = 1


# Adapted from https://github.com/pacman100/LLM-Workshop/blob/0ba41561ce6ea16d3993069c03ec1dca3ab6769d/personal_copilot/training/train.py#L144
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
        shuffle=False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.eot_token_id = tokenizer.encode(
            tokenizer.eot_token, add_special_tokens=False
        )[0]
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.chunked_samples = 0
        self.whole_samples = 0
        self.not_permuted_length = 0
        self.total_length = 0
        self.chars_per_token = chars_per_token
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed
        self.shuffle = shuffle
        self.bos_token_id = self.tokenizer.bos_token_id
        self.suffix_tok_id = self.tokenizer.suffix_id
        self.prefix_tok_id = self.tokenizer.prefix_id
        self.middle_tok_id = self.tokenizer.middle_id
        self.pad_tok_id = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)

        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    document = next(iterator)[self.content_field]

                    if np_rng.binomial(1, 1 - DOCUMENT_SPLIT_RATE):
                        buffer.append(document)
                        buffer_len += len(document)
                        print(f"Added document to buffer. Total length: {buffer_len}")
                        continue

                    old_num_chunks = len(buffer)
                    chunk_num = 0

                    while len(document) > 0:
                        chunk_len = np_rng.randint(
                            round(self.seq_length * 0.8) * self.chars_per_token,
                            round(self.seq_length * 1.2) * self.chars_per_token,
                        )
                        buffer.append(document[:chunk_len])
                        document = document[chunk_len:]
                        chunk_num += 1
                        buffer_len += len(buffer[-1])
                        if buffer_len >= self.max_buffer_size:
                            break

                    print(
                        f"Chunked document into {len(buffer) - old_num_chunks} chunks. Total chunks: {len(buffer)}."
                    )
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(
                buffer, truncation=False, add_special_tokens=False
            )["input_ids"]
            tokenized_inputs = functools.reduce(
                lambda x, y: np.concatenate([x, [self.concat_token_id], y]),
                tokenized_inputs,
            )

            samples = []

            try:
                for i in range(0, len(tokenized_inputs), self.seq_length):
                    sample = tokenized_inputs[i : i + self.seq_length]
                    if len(sample) < self.seq_length:
                        print("Skipping last short sample")
                        break

                    if self.fim_rate > 0:
                        assert (
                            self.fim_rate <= 1 and self.fim_rate >= 0
                        ), "FIM rate must be a probability 0 <= rate <= 1"

                        segment_breaks = np.argwhere(
                            sample == self.concat_token_id
                        )  # split sample by document

                        if segment_breaks.shape[0] > 0:
                            self.chunked_samples += 1
                            curr_start_position = 0
                            new_samples = []
                            for loc in np.nditer(segment_breaks):
                                # Only permute non-empty segments.
                                if loc - curr_start_position > 0:
                                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                                    permuted, np_rng = fim.permute_char_level(
                                        sample[curr_start_position:loc],
                                        np_rng,
                                        self.fim_rate,
                                        self.fim_spm_rate,
                                        self.suffix_tok_id,
                                        self.prefix_tok_id,
                                        self.middle_tok_id,
                                        self.pad_tok_id,
                                        self.tokenizer,
                                    )
                                    new_samples += [
                                        [self.bos_token_id],
                                        permuted,
                                        [self.eot_token_id, self.concat_token_id],
                                    ]

                                curr_start_position = loc + 1  # jump over the EOD token
                            # Permute the segment after the last EOD
                            last_chunk = sample[curr_start_position:]
                            # The last chunk will be truncated after so we'll get a bad example
                            self.not_permuted_length += last_chunk.shape[0]
                            new_samples += [
                                [self.bos_token_id],
                                last_chunk,
                            ]

                            sample = np.concatenate(new_samples)
                        else:
                            self.whole_samples += 1
                            old_sample_length = sample.shape[0]
                            permuted, np_rng = fim.permute_char_level(
                                sample,
                                np_rng,
                                self.fim_rate,
                                self.fim_spm_rate,
                                self.suffix_tok_id,
                                self.prefix_tok_id,
                                self.middle_tok_id,
                                self.pad_tok_id,
                                self.tokenizer,
                                truncate_or_pad=3,
                            )
                            sample = np.concatenate(
                                [
                                    [self.bos_token_id],
                                    permuted,
                                    [self.eot_token_id, self.concat_token_id],
                                ]
                            )
                            if sample.shape[0] != old_sample_length:
                                print(
                                    f"Whole sample permutation error. Length doesn't match. Old: {old_sample_length}, new: {sample.shape[0]}"
                                )

                    # Truncate or pad sequence to max-length
                    diff = sample.shape[0] - self.seq_length
                    if diff > 0:  # too long
                        sample = sample[: self.seq_length]
                    elif diff < 0:  # too short
                        sample = np.concatenate(
                            [sample, np.full((-1 * diff), self.pad_tok_id)]
                        )

                    samples.append(sample)
                    self.total_length += sample.shape[0]
            except Exception as e:
                print(f"Error in sample generation: {str(e)}")
                traceback.print_exc()

            if self.shuffle:
                random.shuffle(samples)
            for sample in samples:
                self.current_size += 1
                if (self.current_size % 1000) == 0:
                    print(f"Current size: {self.current_size}")
                    print(f"Chunked samples: {self.chunked_samples}")
                    print(f"Whole samples: {self.whole_samples}")
                    print(f"Not permuted length: {self.not_permuted_length}")
                    print(f"Total length: {self.total_length}")
                yield {
                    "input_ids": torch.LongTensor(sample),
                    "labels": torch.LongTensor(sample),
                }
