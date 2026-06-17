from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

from app.db import get_session
from app.deps import CurrentUser
from app.models import Run, RunMetric, RunStatus, User
from app.runner import cancel_training, launch_training
from app.schemas import (
    GenerateIn,
    GenerateOut,
    MetricOut,
    RunCreate,
    RunDetailOut,
    RunOut,
)
from training.generate import generate_from_checkpoint

router = APIRouter(prefix="/runs", tags=["runs"])

# Statuses from which a (re)submit is allowed.
_SUBMITTABLE = {RunStatus.queued, RunStatus.failed, RunStatus.cancelled}
_CANCELLABLE = {RunStatus.queued, RunStatus.submitted, RunStatus.running}


async def _owned_run(run_id: int, user: User, session: AsyncSession) -> Run:
    run = await session.get(Run, run_id)
    if run is None or run.user_id != user.id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("", response_model=RunOut, status_code=status.HTTP_201_CREATED)
async def create_run(
    body: RunCreate,
    user: CurrentUser,
    session: AsyncSession = Depends(get_session),
) -> Run:
    run = Run(
        user_id=user.id,
        dataset=body.dataset,
        base_model=body.base_model,
        config_json=body.config,
        status=RunStatus.queued,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run


@router.get("", response_model=list[RunOut])
async def list_runs(user: CurrentUser, session: AsyncSession = Depends(get_session)) -> list[Run]:
    result = await session.execute(
        select(Run).where(Run.user_id == user.id).order_by(Run.id.desc())
    )
    return list(result.scalars().all())


@router.get("/{run_id}", response_model=RunDetailOut)
async def get_run(
    run_id: int, user: CurrentUser, session: AsyncSession = Depends(get_session)
) -> Run:
    result = await session.execute(
        select(Run)
        .where(Run.id == run_id, Run.user_id == user.id)
        .options(selectinload(Run.metrics))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/metrics", response_model=list[MetricOut])
async def run_metrics(
    run_id: int, user: CurrentUser, session: AsyncSession = Depends(get_session)
) -> list[RunMetric]:
    await _owned_run(run_id, user, session)
    result = await session.execute(
        select(RunMetric).where(RunMetric.run_id == run_id).order_by(RunMetric.step)
    )
    return list(result.scalars().all())


@router.post("/{run_id}/submit", response_model=RunOut)
async def submit_run(
    run_id: int, user: CurrentUser, session: AsyncSession = Depends(get_session)
) -> Run:
    run = await _owned_run(run_id, user, session)
    if run.status not in _SUBMITTABLE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is {run.status.value}; cannot submit",
        )
    job_id = launch_training(run.id)
    run.job_id = job_id
    run.status = RunStatus.submitted
    await session.commit()
    await session.refresh(run)
    return run


@router.post("/{run_id}/cancel", response_model=RunOut)
async def cancel_run(
    run_id: int, user: CurrentUser, session: AsyncSession = Depends(get_session)
) -> Run:
    run = await _owned_run(run_id, user, session)
    if run.status not in _CANCELLABLE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is {run.status.value}; cannot cancel",
        )
    cancel_training(run.id)
    # If the trainer was already past its loop (or never started), mark it here;
    # otherwise the training thread will set 'cancelled' as it unwinds.
    run.status = RunStatus.cancelled
    await session.commit()
    await session.refresh(run)
    return run


@router.post("/{run_id}/generate", response_model=GenerateOut)
async def generate(
    run_id: int,
    body: GenerateIn,
    user: CurrentUser,
    session: AsyncSession = Depends(get_session),
) -> GenerateOut:
    run = await _owned_run(run_id, user, session)
    if run.status != RunStatus.succeeded or not run.checkpoint_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run has no trained model yet",
        )
    try:
        text = await run_in_threadpool(
            generate_from_checkpoint,
            run.checkpoint_path,
            body.start_text,
            body.length,
            body.temperature,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="Checkpoint missing") from exc
    return GenerateOut(text=text)
