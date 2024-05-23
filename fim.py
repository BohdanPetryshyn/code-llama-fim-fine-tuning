# coding=utf-8
# Copyright 2024 Sourab Mangrulkar. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import numpy as np


# this is expensive so we cache it
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    if "codellama" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.suffix_id,
            tokenizer.prefix_id,
            tokenizer.middle_id,
            0,
        )
    elif "deepseek-coder" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.encode("<｜fim▁hole｜>", add_special_tokens=False)[0],
            tokenizer.encode("<｜fim▁begin｜>", add_special_tokens=False)[0],
            tokenizer.encode("<｜fim▁end｜>", add_special_tokens=False)[0],
            tokenizer.encode("<pad>", add_special_tokens=False)[0],
        )
    elif "stable-code" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.encode("<fim_suffix>")[0],
            tokenizer.encode("<fim_prefix>")[0],
            tokenizer.encode("<fim_middle>")[0],
            tokenizer.encode("<fim_pad>")[0],
        )
    else:
        bos_token_id = None
        try:
            FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map[
                "additional_special_tokens"
            ][1:5]
            suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
                tokenizer.vocab[tok]
                for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
            )
        except KeyError:
            suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
                None,
                None,
                None,
                None,
            )
    return bos_token_id, suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


def _bos_token_processing(prefix_token_list, bos_token):
    if bos_token is not None:
        # add the BOS token to the beginning of the list
        prefix_token_list.insert(0, bos_token)

    return prefix_token_list


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
    bos_token_id=None,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            prefix_special_tokens = _bos_token_processing(
                [prefix_tok_id, suffix_tok_id], bos_token_id
            )
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    prefix_special_tokens,
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            prefix_special_tokens = _bos_token_processing([prefix_tok_id], bos_token_id)
            # PSM
            new_sample = np.concatenate(
                [
                    prefix_special_tokens,
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't do FIM preproc
        new_sample = sample
    return list(new_sample), np_rng


successful_fim = 0
failed_fim = 0


def permute_char_level(
    sample,
    np_rng,
    fim_rate,
    fim_spm_rate,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    tokenizer,
    truncate_or_pad=-1,
):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """
    global successful_fim, failed_fim

    if np_rng.binomial(1, fim_rate):  # sample bernoulli dist

        contents = tokenizer.decode(sample, skip_special_tokens=True)

        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = contents[: boundaries[0]]
        middle = contents[boundaries[0] : boundaries[1]]
        suffix = contents[boundaries[1] :]

        # By adding and removing the <MID> token, we ensure that the tokenizer doesn't add extra leading whitespace
        # The prefix whitespace doesn't matter as also mentioned in the Code Llama paper
        special_token = "▔"
        special_token_id = tokenizer.encode(
            special_token, add_special_tokens=False, return_tensors="np"
        )[0]
        special_token_id_len = special_token_id.shape[0]

        prefix = tokenizer.encode(
            prefix, add_special_tokens=False, return_tensors="np"
        )[0]
        middle = tokenizer.encode(
            special_token + middle, add_special_tokens=False, return_tensors="np"
        )[0][special_token_id_len:]
        suffix = tokenizer.encode(
            special_token + suffix, add_special_tokens=False, return_tensors="np"
        )[0][special_token_id_len:]

        # Remove after debugging
        # if tokenizer.decode(np.concatenate([prefix, middle, suffix])) != contents:
        #     # print(
        #     #     "Error in FIM transformation: ",
        #     #     tokenizer.decode(np.concatenate([prefix, middle, suffix])),
        #     #     " != ",
        #     #     contents,
        #     # )
        #     failed_fim += 1
        #     print("Failed FIM transformations: ", failed_fim)

        # else:
        #     successful_fim += 1
        #     print("Successful FIM transformations: ", successful_fim)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad >= 0:
            # need to make same length as the input. Take the 3 sentinel tokens into account. truncate_or_pad is the number of tokens added after the transformation
            new_length = (
                suffix.shape[0]
                + prefix.shape[0]
                + middle.shape[0]
                + 3
                + truncate_or_pad
            )
            diff = new_length - sample.shape[0]
            if diff > 0:  # too long
                if (
                    suffix.shape[0] <= diff
                ):  # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    print("suffix too short", diff, suffix.shape[0])
                    return sample[: sample.shape[0] - truncate_or_pad], np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )

    else:
        # don't do FIM preproc
        new_sample = sample[: sample.shape[0] - truncate_or_pad]

    return new_sample, np_rng
