"""Load a run's saved checkpoint and generate text from it.

Each run owns its own trained model instance (see char_train's checkpoint save),
so generation is per-user: the API only loads checkpoints for runs the caller owns.
"""

from pathlib import Path

import torch

from training.char_model import TinyCharModel, generate_text


def generate_from_checkpoint(
    checkpoint_path: str,
    start_text: str,
    length: int = 250,
    temperature: float = 0.8,
) -> str:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyCharModel(ckpt["vocab_size"], ckpt["emb_size"], ckpt["hidden_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return generate_text(
        model,
        start_text=start_text,
        stoi=ckpt["stoi"],
        itos=ckpt["itos"],
        length=length,
        temperature=temperature,
    )
