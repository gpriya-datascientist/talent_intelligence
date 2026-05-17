"""
routers/availability.py — employee self-service availability endpoints.
The UI slider and free_from_date calendar call these routes.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uuid

from backend.db.database import get_db
from backend.models.availability import Availability, AvailabilityStatus

router = APIRouter(prefix="/availability", tags=["availability"])


class AvailabilityUpdateRequest(BaseModel):
    available_percentage: float = Field(ge=0.0, le=1.0)
    status: str
    free_from_date: Optional[datetime] = None
    is_soft_open: bool = False
    soft_open_note: Optional[str] = None
    preferred_domains: Optional[str] = None


@router.put("/{employee_id}")
async def update_availability(
    employee_id: str,
    body: AvailabilityUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Called when employee moves the availability slider in the UI.
    Creates or updates their availability record and recomputes score.
    """
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
    avail.last_updated_by_employee = datetime.utcnow()

    # Recompute and persist score immediately
    avail.availability_score = avail.compute_availability_score()
    await db.flush()

    return {
        "employee_id": employee_id,
        "availability_score": avail.availability_score,
        "status": avail.status.value,
        "available_percentage": avail.available_percentage,
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
        "availability_score": avail.availability_score,
        "last_updated": avail.last_updated_by_employee,
    }
