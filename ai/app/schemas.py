from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import RunStatus


class HealthOut(BaseModel):
    status: str
    db: str


class LoginIn(BaseModel):
    # The classroom logins are usernames (group8..group16), stored in the email column.
    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class CatalogItem(BaseModel):
    id: str
    name: str


class CatalogOut(BaseModel):
    datasets: list[CatalogItem]
    base_models: list[CatalogItem]
    hyperparam_defaults: dict[str, Any]


class RunCreate(BaseModel):
    dataset: str
    base_model: str
    # Hyperparameters (training_steps, lr, temperature, start_text, ...) +
    # optional dataset_text for custom corpora. Stored verbatim in config_json.
    config: dict[str, Any] = Field(default_factory=dict)


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: RunStatus
    job_id: str | None
    dataset: str
    base_model: str
    config_json: dict[str, Any]
    result_json: dict[str, Any] | None
    checkpoint_path: str | None
    created_at: datetime
    updated_at: datetime


class MetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step: int
    loss: float
    ts: datetime
    extra_json: dict[str, Any] | None = None


class RunDetailOut(RunOut):
    metrics: list[MetricOut] = Field(default_factory=list)


class GenerateIn(BaseModel):
    start_text: str = "Once upon"
    length: int = 250
    temperature: float = 0.8


class GenerateOut(BaseModel):
    text: str
