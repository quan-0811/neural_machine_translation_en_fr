import os
import torch
import torch.nn as nn
from model.transformers import Transformers
from utils import save_model
from data_prep.data_setup import create_dataloaders
from engine import train

NUM_EPOCHS = 1
BATCH_SIZE = 64
MAX_NUM_STEPS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

data_dir = "data"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, eng_fra_data = create_dataloaders(data_dir=data_dir, batch_size=BATCH_SIZE, num_steps=MAX_NUM_STEPS)

transformers_model = Transformers(num_blks=6,
                                  num_hiddens=512,
                                  num_heads=8,
                                  dropout_rate=0.15,
                                  eng_vocab_size=len(eng_fra_data.eng_vocab),
                                  fra_vocab_size=len(eng_fra_data.fra_vocab)).to(device)

model_path = "models/transformers_model.pth"

if os.path.exists(model_path):
    try:
        transformers_model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print("No pre-trained model found. Initializing a new model.")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(transformers_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train(num_epochs=NUM_EPOCHS,
      model=transformers_model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      tar_vocab=eng_fra_data.fra_vocab,
      device=device)

save_model(model=transformers_model,
                 targ_dir="models",
                 model_name="transformers_model.pth")