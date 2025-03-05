import torch
from torch.utils.data import Dataset

class EngFraTranslationDataset(Dataset):
    def __init__(self, eng_tensor: torch.Tensor, fra_tensor: torch.Tensor, eng_valid_len: torch.Tensor, fra_valid_len: torch.Tensor):
        self.eng_tensor = eng_tensor
        self.fra_tensor = fra_tensor
        self.eng_valid_len = eng_valid_len
        self.fra_valid_len = fra_valid_len

    def __len__(self):
        return len(self.eng_tensor)

    def __getitem__(self, idx):
        return self.eng_tensor[idx], self.eng_valid_len[idx], self.fra_tensor[idx][:-1] , self.fra_valid_len[idx], self.fra_tensor[idx][1:]