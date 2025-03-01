import os
import shutil
import kagglehub
import re

class EngFraData:
    def __init__(self, download_folder_path = "data"):
        self.download_folder_path = download_folder_path

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
            for line in file:
                parts = line.split("\t")
                if len(parts) == 2:
                    src.append([word for word in f"{parts[0]} {end_token}".split(" ") if word])
                    tgt.append([word for word in f"{parts[1]} {end_token}".split(" ") if word])
        return src, tgt




