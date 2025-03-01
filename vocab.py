import collections

class Vocabulary:
    def __init__(self, sentences, min_freq, unk_token="<unk>"):
        tokens = [token for sentence in sentences for token in sentence]
        counter = collections.Counter(tokens)
        token_freqs = counter.items()
        token_list = sorted(set([unk_token] + [token for token, freq in token_freqs if freq >= min_freq]))
        self.token_to_idx = {token: i for i, token in enumerate(token_list)}
        self.idx_to_token = {i: token for i, token in enumerate(token_list)}

    def __len__(self):
        return len(self.token_to_idx)

    def __getitem__(self, tokens):
        if isinstance(tokens, str):  # Single token case
            return self.token_to_idx.get(tokens, self.token_to_idx["<unk>"])
        return [self.__getitem__(token) for token in tokens]  # List of tokens case

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

