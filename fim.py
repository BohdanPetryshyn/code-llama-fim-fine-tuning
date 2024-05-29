import numpy as np


# Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py#L491
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
                ):  # if there's no space to truncate the suffix: stop and report it.
                    print("suffix too short", diff, suffix.shape[0])
                    return sample[: sample.shape[0] - truncate_or_pad], np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM
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
