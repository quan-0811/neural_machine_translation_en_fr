import torch
import os
import shutil
import kagglehub
import re
from vocab import Vocabulary

class EngFraData:
    def __init__(self, num_steps: int, download_folder_path = "data"):
        self.download_folder_path = download_folder_path
        self.num_steps = num_steps
        eng_tensor, fra_tensor, valid_eng_len, eng_vocab, fra_vocab = self.build_data()
        self.eng_tensor = eng_tensor
        self.fra_tensor = fra_tensor
        self.valid_eng_len = valid_eng_len
        self.eng_vocab = eng_vocab
        self.fra_vocab = fra_vocab

    # Download the dataset
    def download(self):
        default_path = kagglehub.dataset_download("digvijayyadav/frenchenglish")

        os.makedirs(self.download_folder_path, exist_ok=True)

        for filename in os.listdir(default_path):
            src_file = os.path.join(default_path, filename)
            dest_file = os.path.join(self.download_folder_path, filename)
            shutil.move(src_file, dest_file)

        print("Eng-Fra dataset downloaded at:", self.download_folder_path)

    # Preprocess the dataset
    def preprocess(self):
        cleaned_lines = []

        # Add space before punctuations if there's none
        def fix_spacing(text):
            new_text = ""
            for i, char in enumerate(text):
                if i > 0 and char in ",.!?" and text[i - 1] != " ":
                    new_text += " "
                new_text += char
            return new_text

        # Remove unnecessary texts
        with open(self.download_folder_path + "/fra.txt", "r", encoding="utf-8") as file:
            for line in file:
                line = line.replace('\u202f', ' ').replace('\xa0', ' ')
                match = re.match(r"^(.*?)\t(.*?)\tCC-BY", line)
                if match:
                    english, french = match.groups()
                    english = fix_spacing(english.lower())
                    french = fix_spacing(french.lower())
                    cleaned_lines.append(f"{english}\t{french}\n")

        # Rewrite the file with preprocessed data
        with open(self.download_folder_path + "/fra.txt", "w", encoding="utf-8") as file:
            file.writelines(cleaned_lines)

        print(f"Cleaned data saved")

    # Word-level tokenization
    def tokenize(self, end_token = "<eos>"):
        src, tgt = [], []
        with open(self.download_folder_path + "/fra.txt", "r",  encoding="utf-8") as file:
            lines = file.read().split("\n")
            for line in lines:
                parts = line.split("\t")
                if len(parts) == 2:
                    src.append([word for word in f"{parts[0]} {end_token}".split(" ") if word])
                    tgt.append([word for word in f"{parts[1]} {end_token}".split(" ") if word])
        return src, tgt

    # Pad/Truncate data to fixed length, turn into tensors
    def build_data(self, start_token = "<bos>", pad_token = "<pad>"):
        pad_or_trim = lambda seq, t: seq[:t] if len(seq) > t else seq + [pad_token] * (t - len(seq))

        src, tgt = self.tokenize()

        eng_sentences = [pad_or_trim(s, self.num_steps) for s in src]
        fra_sentences = [[start_token] + pad_or_trim(s, self.num_steps) for s in tgt]

        eng_vocab = Vocabulary(eng_sentences, min_freq=2)
        fra_vocab = Vocabulary(fra_sentences, min_freq=2)

        eng_tensor = torch.tensor([eng_vocab[s] for s in eng_sentences])
        valid_eng_len = (eng_tensor != eng_vocab["<pad>"]).type(torch.int32).sum(1)
        fra_tensor = torch.tensor([fra_vocab[s] for s in fra_sentences])

        return eng_tensor, fra_tensor, valid_eng_len, eng_vocab, fra_vocab