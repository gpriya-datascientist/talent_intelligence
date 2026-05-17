from sqlalchemy import String, Text, Float, Boolean, DateTime, JSON, Integer, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional
import uuid
import enum

from backend.db.database import Base


class SeniorityLevel(str, enum.Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class EmploymentType(str, enum.Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACTOR = "contractor"


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    department: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    seniority_level: Mapped[SeniorityLevel] = mapped_column(
        SAEnum(SeniorityLevel), default=SeniorityLevel.MID, nullable=False
    )
    employment_type: Mapped[EmploymentType] = mapped_column(
        SAEnum(EmploymentType), default=EmploymentType.FULL_TIME, nullable=False
    )

    # Resume
    resume_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resume_uploaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # GitHub
    github_username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    github_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    github_stats: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # e.g. {"total_commits": 340, "top_languages": ["Python", "TypeScript"], "active_repos": 5}

    # Extracted skill vector (raw JSON from extraction chain)
    extracted_skills: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    skill_extraction_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    skills_extracted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Vector embedding reference (the actual vector lives in FAISS/Pinecone)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedding_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Profile flags
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_sme: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sme_domains: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # e.g. ["audio_dsp", "ux_design"]

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships (defined in later models)
    availability: Mapped[Optional["Availability"]] = relationship(
        "Availability", back_populates="employee", uselist=False, cascade="all, delete-orphan"
    )
    skills: Mapped[list["Skill"]] = relationship(
        "Skill", back_populates="employee", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Employee {self.full_name} | {self.title} | {self.seniority_level.value}>"

    @property
    def has_embedding(self) -> bool:
        return self.embedding_id is not None

    @property
    def needs_skill_extraction(self) -> bool:
        return self.extracted_skills is None or self.resume_text is not None and (
            self.skills_extracted_at is None or
            self.resume_uploaded_at > self.skills_extracted_at
        )
