import torch
from pathlib import Path
from vocab import Vocabulary

def mask_sequence(X: torch.Tensor, valid_len: torch.Tensor, value=-1e6):
    num_steps = X.size(1)

    valid_len = valid_len.to(X.device)
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

    valid_len = valid_len.to(X.device)
    X = mask_sequence(X, valid_len)

    X = torch.softmax(X, dim=-1)

    row_mask = valid_len > 0
    X = X * row_mask.unsqueeze(-1)

    return X

def clean_sentence(sentence, pad_token_id):
    return [word for word in sentence if word != pad_token_id]

def save_model(model: torch.nn.Module, targ_dir: str, model_name: str):
    target_dir_path = Path(targ_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def fix_spacing(sentence: str):
    sentence = sentence.strip()
    new_text = ""
    for i, char in enumerate(sentence):
        if i > 0 and char in ",.!?" and sentence[i - 1] != " ":
            new_text += " "
        new_text += char
    return new_text

def translate_sentence(model: torch.nn.Module, sentence: str, max_num_steps: int, src_vocab: Vocabulary, tgt_vocab: Vocabulary, device):
    model.to(device)
    model.eval()
    with torch.inference_mode():

        enc_tokens = [src_vocab.token_to_idx[word] for word in sentence.split() if word in src_vocab.token_to_idx]
        enc_tokens.append(src_vocab.token_to_idx["<eos>"])
        enc_input = torch.tensor(enc_tokens, dtype=torch.long, device=device).unsqueeze(0)
        enc_valid_lens = torch.tensor([len(enc_tokens)], dtype=torch.long, device=device)

        dec_tokens = [tgt_vocab.token_to_idx["<bos>"]]
        dec_input = torch.tensor(dec_tokens, dtype=torch.long, device=device).unsqueeze(0)
        dec_valid_lens = torch.tensor([len(dec_tokens)], dtype=torch.long, device=device)

        enc_output = model.encoder(enc_input, enc_valid_lens)

        for _ in range(max_num_steps):
            dec_valid_lens = torch.tensor([len(dec_tokens)], dtype=torch.long, device=device)

            dec_output = model.decoder(dec_input, enc_output, enc_valid_lens, dec_valid_lens)

            predicted_token = dec_output[:, -1, :].argmax(dim=-1).item()

            if predicted_token == tgt_vocab.token_to_idx["<eos>"]:
                break

            dec_tokens.append(predicted_token)
            dec_input = torch.tensor(dec_tokens, dtype=torch.long, device=device).unsqueeze(0)

        return " ".join(tgt_vocab.idx_to_token[token] for token in dec_tokens[1:])