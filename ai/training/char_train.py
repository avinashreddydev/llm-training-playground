"""Real (tiny) in-process trainer. Runs in a thread (see app/runner.py).

Loads the run + config from the DB, trains a char-level GRU from scratch, writes
a loss row every ``log_every`` steps to ``run_metrics``, checks the cancel event
cooperatively, saves a per-run checkpoint, and records the result on the run.
Uses the SYNC SQLAlchemy session because it runs off the event loop.
"""

import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from app.config import get_settings
from app.datasets import get_dataset_text
from app.db import SyncSessionLocal
from app.models import Run, RunMetric, RunStatus
from app.presets import DEFAULT_BASE_MODEL, HYPERPARAM_DEFAULTS, MODEL_SIZES
from training.char_model import (
    TinyCharModel,
    build_vocab,
    encode,
    generate_text,
    sample_batch,
)

MIN_CHARS = 100


def _set_status(run_id: int, status: RunStatus) -> None:
    with SyncSessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.status = status
            session.commit()


def _checkpoint_dir(run_id: int) -> Path:
    base = Path(get_settings().work_dir) / f"run_{run_id}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def train_char_model(run_id: int, cancel: threading.Event) -> None:
    cfg = dict(HYPERPARAM_DEFAULTS)
    with SyncSessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None:
            return
        cfg.update(run.config_json or {})
        dataset_key = run.dataset
        base_model = run.base_model
        custom_text = (run.config_json or {}).get("dataset_text")

    text = get_dataset_text(dataset_key, custom_text)
    if len(text.strip()) < MIN_CHARS:
        _finish_failed(run_id, f"Dataset too short (need >= {MIN_CHARS} characters).")
        return

    _set_status(run_id, RunStatus.running)

    try:
        result = _run_training(run_id, text, base_model, cfg, cancel)
    except Exception as exc:  # noqa: BLE001
        _finish_failed(run_id, f"{type(exc).__name__}: {exc}")
        raise

    if cancel.is_set():
        _set_status(run_id, RunStatus.cancelled)
        return

    _finish_succeeded(run_id, result)


def _run_training(
    run_id: int,
    text: str,
    base_model: str,
    cfg: dict,
    cancel: threading.Event,
) -> dict:
    torch.manual_seed(int(cfg["seed"]))

    size = MODEL_SIZES.get(base_model, MODEL_SIZES[DEFAULT_BASE_MODEL])
    _, stoi, itos = build_vocab(text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCharModel(len(stoi), size["emb_size"], size["hidden_size"]).to(device)
    data = encode(text, stoi).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))

    steps = int(cfg["training_steps"])
    log_every = max(1, int(cfg["log_every"]))
    block_size = int(cfg["block_size"])
    batch_size = int(cfg["batch_size"])

    model.train()
    first_loss = last_loss = 0.0
    for step in range(1, steps + 1):
        if cancel.is_set():
            break
        xb, yb = sample_batch(data, block_size=block_size, batch_size=batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, _ = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if step == 1:
            first_loss = loss_val
        last_loss = loss_val

        if step % log_every == 0 or step == steps:
            _write_metric(run_id, step, loss_val)

    # Save this run's trained model so its owner can generate from it later.
    ckpt_path = _checkpoint_dir(run_id) / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "emb_size": size["emb_size"],
            "hidden_size": size["hidden_size"],
            "vocab_size": len(stoi),
        },
        ckpt_path,
    )

    sample = generate_text(
        model,
        start_text=str(cfg["start_text"]),
        stoi=stoi,
        itos=itos,
        length=int(cfg["output_length"]),
        temperature=float(cfg["temperature"]),
    )

    return {
        "sample_text": sample,
        "vocab_size": len(stoi),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "steps": steps,
        "device": device,
        "checkpoint_path": str(ckpt_path),
    }


def _write_metric(run_id: int, step: int, loss: float) -> None:
    with SyncSessionLocal() as session:
        session.add(RunMetric(run_id=run_id, step=step, loss=loss, extra_json=None))
        session.commit()


def _finish_succeeded(run_id: int, result: dict) -> None:
    with SyncSessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.status = RunStatus.succeeded
            run.result_json = result
            run.checkpoint_path = result.get("checkpoint_path")
            session.commit()


def _finish_failed(run_id: int, message: str) -> None:
    with SyncSessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.status = RunStatus.failed
            run.result_json = {"error": message}
            session.commit()
