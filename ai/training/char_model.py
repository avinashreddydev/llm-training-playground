"""Tiny character-level language model + (de)tokenization + sampling.

Pure torch, trained from scratch on a small text corpus — no pretrained weights,
no downloads. Adapted from the classroom Gradio trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCharModel(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 48, hidden_size: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, hidden=None):
        x = self.embedding(idx)
        out, hidden = self.rnn(x, hidden)
        logits = self.head(out)
        return logits, hidden


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)


def decode(indices, itos: dict) -> str:
    return "".join(itos[int(i)] for i in indices)


def sample_batch(data: torch.Tensor, block_size: int = 64, batch_size: int = 16):
    if len(data) <= block_size + 1:
        block_size = max(4, len(data) - 2)
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def generate_text(
    model: TinyCharModel,
    start_text: str,
    stoi: dict,
    itos: dict,
    length: int = 300,
    temperature: float = 0.8,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    clean_start = "".join(c for c in start_text if c in stoi)
    if clean_start == "":
        # nothing usable in the prompt: start from any known character
        clean_start = next(iter(stoi))

    idx = torch.tensor([[stoi[c] for c in clean_start]], dtype=torch.long, device=device)
    for _ in range(length):
        idx_cond = idx[:, -64:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 0.1)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return decode(idx[0].tolist(), itos)
