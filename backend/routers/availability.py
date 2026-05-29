"""
routers/availability.py — employee self-service availability endpoints.
Saves availability + syncs GitHub stats when username is provided.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional
import uuid

from backend.db.database import get_db
from backend.models.availability import Availability, AvailabilityStatus
from backend.models.employee import Employee
from backend.ingestion.github_loader import load_github_stats

router = APIRouter(prefix="/availability", tags=["availability"])


class AvailabilityUpdateRequest(BaseModel):
    available_percentage: float = Field(ge=0.0, le=1.0)
    status: str
    free_from_date: Optional[datetime] = None
    is_soft_open: bool = False
    soft_open_note: Optional[str] = None
    preferred_domains: Optional[str] = None
    github_username: Optional[str] = None  # ← new field


@router.put("/{employee_id}")
async def update_availability(
    employee_id: str,
    body: AvailabilityUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    # ── Update availability record ─────────────────────────────────
    avail = await db.get(Availability, employee_id)
    if not avail:
        avail = Availability(id=str(uuid.uuid4()), employee_id=employee_id)
        db.add(avail)

    avail.available_percentage = body.available_percentage
    avail.status = AvailabilityStatus(body.status)
    avail.free_from_date = body.free_from_date
    avail.is_soft_open = body.is_soft_open
    avail.soft_open_note = body.soft_open_note
    avail.preferred_domains = body.preferred_domains
    avail.last_updated_by_employee = datetime.now(timezone.utc)
    avail.availability_score = avail.compute_availability_score()
    await db.flush()

    # ── Sync GitHub if username provided ──────────────────────────
    github_synced = False
    github_stats = None

    if body.github_username:
        emp = await db.get(Employee, employee_id)
        if emp:
            username_changed = emp.github_username != body.github_username

            # Save username regardless
            emp.github_username = body.github_username

            # Only fetch from GitHub API if username changed or stats are empty
            if username_changed or not emp.github_stats:
                try:
                    github_stats = load_github_stats(
                        username=body.github_username,
                        use_seed=False,
                    )
                    emp.github_stats = github_stats
                    emp.github_synced_at = datetime.now(timezone.utc)
                    github_synced = True
                except Exception as e:
                    # Don't fail the whole request if GitHub sync fails
                    github_stats = {"error": str(e)}
            else:
                # Username unchanged and stats exist — return cached stats
                github_stats = emp.github_stats
                github_synced = False

            await db.flush()

    return {
        "employee_id": employee_id,
        "availability_score": avail.availability_score,
        "status": avail.status.value,
        "available_percentage": avail.available_percentage,
        "github_synced": github_synced,
        "github_stats": github_stats,
    }


@router.get("/{employee_id}")
async def get_availability(employee_id: str, db: AsyncSession = Depends(get_db)):
    avail = await db.get(Availability, employee_id)
    if not avail:
        raise HTTPException(status_code=404, detail="Availability record not found")
    return {
        "employee_id": employee_id,
        "status": avail.status.value,
        "available_percentage": avail.available_percentage,
        "free_from_date": avail.free_from_date,
        "is_soft_open": avail.is_soft_open,
        "soft_open_note": avail.soft_open_note,
        "availability_score": avail.availability_score,
        "last_updated": avail.last_updated_by_employee,
    }
