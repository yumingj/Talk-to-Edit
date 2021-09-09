import numpy as np
import torch

# global variables
PUNCTUATION_TO_KEEP = ['?', ';']
PUNCTUATION_TO_REMOVE = ['.', '!', ',']
SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def build_vocab(text_list,
                min_token_count=1,
                delimiter=' ',
                punct_to_keep=None,
                punct_to_remove=None,
                print_every=10000):
    """
    Build token to index mapping from a list of text strings
    -- Input: a list of text string
    -- Output: a dict which is a mapping from token to index,
    """

    token_to_count = {}
    # tokenize text and add tokens to token_to_count dict
    for text_idx, text in enumerate(text_list):
        if text_idx % print_every == 0:
            print('tokenized', text_idx, '/', len(text_list))
        text_tokens = tokenize(text=text, delimiter=delimiter)
        for token in text_tokens:
            if token in token_to_count:
                token_to_count[token] += 1
            else:
                token_to_count[token] = 1

    token_to_idx = {}
    print('Mapping tokens to indices')

    # reserve indices for special tokens (must-have tokens)
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx

    # assign indices to tokens
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def tokenize(text,
             delimiter=' ',
             add_start_token=False,
             add_end_token=False,
             punctuation_to_keep=PUNCTUATION_TO_KEEP,
             punctuation_to_remove=PUNCTUATION_TO_REMOVE):
    """
    Tokenize a text string
    -- Input: a text string
    -- Output: a list of tokens,
       each token is still a string (usually an english word)
    """

    # (1) Optionally keep or remove certain punctuation
    if punctuation_to_keep is not None:
        for punctuation in punctuation_to_keep:
            text = text.replace(punctuation, '%s%s' % (delimiter, punctuation))
    if punctuation_to_remove is not None:
        for punctuation in punctuation_to_remove:
            text = text.replace(punctuation, '')

    # (2) Split the text string into a list of tokens
    text = text.lower()
    tokens = text.split(delimiter)

    # (3) Optionally add start and end tokens
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')

    return tokens


def encode(text_tokens, token_to_idx, allow_unk=False):
    text_encoded = []
    for token in text_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        text_encoded.append(token_to_idx[token])
    return text_encoded


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def reverse_dict(input_dict):

    reversed_dict = {}
    for key in input_dict.keys():
        val = input_dict[key]
        reversed_dict[val] = key
    return reversed_dict


def to_long_tensor(dset):
    arr = np.asarray(dset, dtype=np.int64)
    tensor = torch.LongTensor(arr)
    return tensor


def proper_capitalize(text):
    if len(text) > 0:
        text = text.lower()
        text = text[0].capitalize() + text[1:]
        for idx, char in enumerate(text):
            if char in ['.', '!', '?'] and (idx + 2) < len(text):
                text = text[:idx + 2] + text[idx + 2].capitalize() + text[idx +
                                                                          3:]
        text = text.replace(' i ', ' I ')
        text = text.replace(',i ', ',I ')
        text = text.replace('.i ', '.I ')
        text = text.replace('!i ', '!I ')
    return text
