import torch

def mask_sequence(X: torch.Tensor, valid_len: torch.Tensor, value=-1e6):
    num_steps = X.size(1)

    mask = torch.arange(num_steps, device=X.device).unsqueeze(0) < valid_len.unsqueeze(-1)

    X[~mask] = value
    return X

def masked_softmax(X: torch.Tensor, valid_len: torch.Tensor):
    shape = X.shape
    batch_size, num_steps = shape[0], shape[1]

    if valid_len.dim() == 1:
        valid_lens = torch.zeros(size=(batch_size, num_steps), dtype=torch.long)
        for i in range(batch_size):
            valid_lens[i, :valid_len[i]] = torch.tensor([valid_len[i]] * valid_len[i])
        valid_len = valid_lens

    X = mask_sequence(X, valid_len)

    X = torch.softmax(X, dim=-1)

    row_mask = valid_len > 0
    X = X * row_mask.unsqueeze(-1)

    return X