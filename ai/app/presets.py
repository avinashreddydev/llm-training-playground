"""Catalog the UI form reads: preset datasets + model sizes + default hyperparams."""

from app.datasets import DATASETS

# Dataset ids are the keys of the corpus catalog.
PRESET_DATASETS = [{"id": key, "name": key} for key in DATASETS]

# "base_model" here selects the tiny char-GRU size (trained from scratch — no
# pretrained download). Kept in the schema's base_model field.
PRESET_BASE_MODELS = [
    {"id": "tiny-gru", "name": "Tiny GRU (emb 48 / hidden 96)", "emb_size": 48, "hidden_size": 96},
    {
        "id": "small-gru",
        "name": "Small GRU (emb 64 / hidden 128)",
        "emb_size": 64,
        "hidden_size": 128,
    },
    {
        "id": "micro-gru",
        "name": "Micro GRU (emb 32 / hidden 64)",
        "emb_size": 32,
        "hidden_size": 64,
    },
]
MODEL_SIZES = {m["id"]: m for m in PRESET_BASE_MODELS}
DEFAULT_BASE_MODEL = "tiny-gru"

# Hyperparameter defaults; a run's config_json overrides any of these.
HYPERPARAM_DEFAULTS = {
    "training_steps": 500,
    "lr": 2e-3,
    "block_size": 64,
    "batch_size": 16,
    "temperature": 0.8,
    "start_text": "Once upon",
    "output_length": 250,
    "log_every": 5,
    "seed": 7,
}
