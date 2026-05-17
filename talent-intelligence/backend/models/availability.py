from sqlalchemy import String, Float, Boolean, DateTime, Text, ForeignKey, Enum as SAEnum, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime, date
from typing import Optional
import uuid
import enum

from backend.db.database import Base


class AvailabilityStatus(str, enum.Enum):
    AVAILABLE = "available"           # 100% free
    PARTIALLY_AVAILABLE = "partially_available"  # has bandwidth for more
    BUSY = "busy"                     # fully allocated
    ON_LEAVE = "on_leave"             # PTO / sick leave
    SOFT_OPEN = "soft_open"           # busy but open to being approached


class Availability(Base):
    __tablename__ = "availability"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True
    )
    # unique=True enforces one availability record per employee

    # Self-reported by employee via UI slider (0.0 - 1.0)
    available_percentage: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    # 1.0 = fully free, 0.4 = 40% bandwidth available, 0.0 = fully booked

    status: Mapped[AvailabilityStatus] = mapped_column(
        SAEnum(AvailabilityStatus), default=AvailabilityStatus.AVAILABLE, nullable=False
    )

    # Current project assignment
    current_project_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    current_project_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    current_project_end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # n8n watches this date — when it passes, status auto-resets to AVAILABLE

    # Future availability
    free_from_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Employee sets this: "I'll be free from June 15"

    # Soft open flag — employee is busy but willing to be approached
    is_soft_open: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    soft_open_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # e.g. "Finishing current sprint end of month, happy to discuss after"

    # Preferred next work type (employee sets via UI)
    preferred_domains: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # comma-separated: "backend,ml,audio_dsp"

    # Ranking modifier — computed field used directly by ranking engine
    # 1.0 = fully available, 0.0 = completely unavailable
    availability_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Timestamps
    last_updated_by_employee: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # tracks when employee last touched their availability — staleness signal
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship
    employee: Mapped["Employee"] = relationship("Employee", back_populates="availability")

    def compute_availability_score(self) -> float:
        """
        Computes a 0.0-1.0 score for the ranking engine.
        Called by n8n after any availability change.
        """
        if self.status == AvailabilityStatus.ON_LEAVE:
            return 0.0
        if self.status == AvailabilityStatus.AVAILABLE:
            return 1.0
        if self.status == AvailabilityStatus.SOFT_OPEN:
            return 0.3
        # Partially available — use the slider value directly
        base = self.available_percentage
        # Boost score if free_from_date is within 30 days
        if self.free_from_date:
            from datetime import timezone
            days_until_free = (self.free_from_date - datetime.now(timezone.utc)).days
            if days_until_free <= 30:
                base = min(base + 0.2, 1.0)
        return round(base, 4)

    def __repr__(self) -> str:
        return f"<Availability {self.employee_id} | {self.status.value} | {self.available_percentage*100:.0f}%>"
