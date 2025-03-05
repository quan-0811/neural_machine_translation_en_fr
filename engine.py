import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import clean_sentence
from vocab import Vocabulary
from tqdm import tqdm

def train_step(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, tar_vocab: Vocabulary, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device):
    model.train()
    total_train_loss, total_train_bleu_score = 0, 0
    tar_vocab_size = len(tar_vocab)
    pad_tar_id = tar_vocab.token_to_idx["<pad>"]
    smoother = SmoothingFunction().method1

    for batch, (X_enc_train, X_train_enc_lens, X_dec_train, X_train_dec_lens, Y_train_dec_truth) in enumerate(train_dataloader):
        X_enc_train, X_train_enc_lens, X_dec_train, X_train_dec_lens, Y_train_dec_truth = X_enc_train.to(device), X_train_enc_lens.to(device), X_dec_train.to(device), X_train_dec_lens.to(device), Y_train_dec_truth.to(device)

        # Forward pass
        Y_train_pred_logits = model(X_enc_train, X_train_enc_lens, X_dec_train, X_train_dec_lens)

        # Loss calculation
        train_loss = loss_fn(Y_train_pred_logits.view(-1, tar_vocab_size), Y_train_dec_truth.view(-1))
        total_train_loss += train_loss.item()

        # BLEU score calculation
        Y_train_pred_list = Y_train_pred_logits.argmax(dim=-1).tolist()
        Y_train_truth_list = Y_train_dec_truth.tolist()

        batch_bleu_score = 0
        valid_sentences = 0

        for pred, truth in zip(Y_train_pred_list, Y_train_truth_list):
            pred_cleaned = clean_sentence(pred, pad_tar_id)
            truth_cleaned = clean_sentence(truth, pad_tar_id)

            if pred_cleaned and truth_cleaned:
                batch_bleu_score += sentence_bleu([truth_cleaned], pred_cleaned, smoothing_function=smoother)
                valid_sentences += 1

        if valid_sentences > 0:
            total_train_bleu_score += batch_bleu_score / valid_sentences

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        train_loss.backward()

        # Optimizer step
        optimizer.step()

    avg_train_bleu = total_train_bleu_score / len(train_dataloader)
    avg_train_loss = total_train_loss / len(train_dataloader)

    return avg_train_loss, avg_train_bleu

def test_step(model: nn.Module, test_dataloader: torch.utils.data.DataLoader, tar_vocab: Vocabulary, loss_fn: nn.Module, device):
    model.eval()
    with torch.inference_mode():
        total_test_loss, total_test_bleu_score = 0, 0
        tar_vocab_size = len(tar_vocab)
        pad_tar_id = tar_vocab.token_to_idx["<pad>"]
        smoother = SmoothingFunction().method1
        for X_enc_test, X_test_enc_lens, X_dec_test, X_test_dec_lens, Y_test_dec_truth in test_dataloader:
            X_enc_test, X_test_enc_lens, X_dec_test, X_test_dec_lens, Y_test_dec_truth = X_enc_test.to(device), X_test_enc_lens.to(device), X_dec_test.to(device), X_test_dec_lens.to(device), Y_test_dec_truth.to(device)

            # Forward pass
            Y_test_pred_logits = model(X_enc_test, X_test_enc_lens, X_dec_test, X_test_dec_lens)

            # Loss calculation
            train_loss = loss_fn(Y_test_pred_logits.view(-1, tar_vocab_size), Y_test_dec_truth.view(-1))
            total_test_loss += train_loss.item()

            # BLEU score calculation
            Y_test_pred_list = Y_test_pred_logits.argmax(dim=-1).tolist()
            Y_test_truth_list = Y_test_dec_truth.tolist()

            batch_bleu_score = 0
            valid_sentences = 0

            for pred, truth in zip(Y_test_pred_list, Y_test_truth_list):
                pred_cleaned = clean_sentence(pred, pad_tar_id)
                truth_cleaned = clean_sentence(truth, pad_tar_id)

                if pred_cleaned and truth_cleaned:
                    batch_bleu_score += sentence_bleu([truth_cleaned], pred_cleaned, smoothing_function=smoother)
                    valid_sentences += 1

            if valid_sentences > 0:
                total_test_bleu_score += batch_bleu_score / valid_sentences

        avg_test_bleu = total_test_bleu_score / len(test_dataloader)
        avg_test_loss = total_test_loss / len(test_dataloader)

    return avg_test_loss, avg_test_bleu

def train(model: nn.Module, num_epochs: int, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, tar_vocab: Vocabulary, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device):
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        avg_train_loss, avg_train_bleu = train_step(model, train_dataloader, tar_vocab, loss_fn, optimizer, device)
        avg_test_loss, avg_test_bleu = test_step(model, test_dataloader, tar_vocab, loss_fn, device)
        print(f"Epoch: {epoch} \n--------")
        print(f"Avg Train Loss: {avg_train_loss:.5f} | Avg Train BLEU Score: {avg_train_bleu:.4f}")
        print(f"Avg Test Loss: {avg_test_loss:.5f} | Avg Test BLEU Score: {avg_test_bleu:.4f}")







