import torch

def mask_sequence(X: torch.Tensor, valid_len: torch.Tensor, value=-1e6):
    num_steps = X.size(1)
    mask = torch.arange(num_steps, dtype=torch.float32)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X: torch.Tensor, valid_len: torch.Tensor):
    shape = X.shape
    print(shape)
    if valid_len.dim() == 1:
        valid_len = torch.repeat_interleave(valid_len, shape[1])
    else:
        valid_len = valid_len.reshape(-1)

    X = mask_sequence(X.reshape(-1, shape[-1]), valid_len)

    return torch.softmax(X.reshape(shape), dim=-1)