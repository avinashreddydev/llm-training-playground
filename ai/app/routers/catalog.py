from fastapi import APIRouter

from app.deps import CurrentUser
from app.presets import HYPERPARAM_DEFAULTS, PRESET_BASE_MODELS, PRESET_DATASETS
from app.schemas import CatalogOut

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("", response_model=CatalogOut)
async def catalog(_user: CurrentUser) -> CatalogOut:
    return CatalogOut(
        datasets=PRESET_DATASETS,
        base_models=[{"id": m["id"], "name": m["name"]} for m in PRESET_BASE_MODELS],
        hyperparam_defaults=HYPERPARAM_DEFAULTS,
    )
