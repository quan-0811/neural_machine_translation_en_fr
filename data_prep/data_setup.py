import os
from torch.utils.data import DataLoader, random_split
from data_prep.data_handler import EngFraData
from dataset import EngFraTranslationDataset

def create_dataloaders(data_dir: str, batch_size: int, num_steps: int, train_size=0.8):
    data = EngFraData(num_steps=num_steps, download_folder_path=data_dir)
    dataset = EngFraTranslationDataset(eng_tensor=data.eng_tensor, fra_tensor=data.fra_tensor, eng_valid_len=data.valid_eng_len, fra_valid_len=data.valid_fra_len)

    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    return train_dataloader, test_dataloader, data
