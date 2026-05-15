from sqlalchemy import String, Text, Float, DateTime, JSON, Enum as SAEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional
import uuid
import enum

from backend.db.database import Base


class WishStatus(str, enum.Enum):
    PENDING = "pending"             # Just submitted by PO
    PARSING = "parsing"             # Wish parser chain running
    AWAITING_SME = "awaiting_sme"   # Waiting for SME consultation input
    ENRICHING = "enriching"         # Requirement builder running
    MATCHING = "matching"           # Ranking engine running
    COMPLETED = "completed"         # Results ready for PO
    FAILED = "failed"               # Something broke in the pipeline


class Wish(Base):
    __tablename__ = "wishes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    po_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    # po_id = the Product Owner's user ID (no User model yet, just store the ID)

    # Raw input from PO
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    # e.g. "We need software for tuning speaker frequency response on embedded devices"

    status: Mapped[WishStatus] = mapped_column(
        SAEnum(WishStatus), default=WishStatus.PENDING, nullable=False, index=True
    )

    # Output of Wish Parser chain
    parsed_intent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # e.g. "Build embedded DSP software for speaker EQ tuning"
    detected_domains: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # e.g. ["audio_dsp", "embedded_systems", "ux_design"]
    ambiguities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # e.g. [{"field": "platform", "question": "Is this for Linux or RTOS?"}]
    parser_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # SME consultation tracking
    required_sme_domains: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # domains that need SME input before matching
    sme_inputs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # collected SME responses keyed by domain
    # e.g. {"audio_dsp": {"constraints": "needs IIR filter experience", "sme_id": "uuid"}}
    sme_consultation_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Output of Enriched Requirement Builder
    enriched_requirements: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Full structured skill requirements fed into RAG + ranking
    # e.g. {
    #   "must_have": [{"skill": "DSP", "proficiency": "advanced", "hands_on": true}],
    #   "nice_to_have": [{"skill": "Python", "proficiency": "intermediate"}],
    #   "team_size": 3,
    #   "seniority_mix": {"senior": 1, "mid": 2}
    # }

    # Final matching results
    matched_candidates: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # ranked list of employee IDs with scores and explanations
    # e.g. [{"employee_id": "uuid", "score": 0.91, "explanation": "...", "rank": 1}]

    # Pipeline timing — useful for Langfuse observability
    parsing_duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    enrichment_duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    matching_duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    failed_at_stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<Wish {self.id[:8]} | {self.status.value} | '{self.raw_text[:40]}...'>"

    @property
    def is_ready_for_matching(self) -> bool:
        """True when all SME inputs are collected and requirements are enriched."""
        return (
            self.enriched_requirements is not None and
            self.status == WishStatus.ENRICHING
        )

    @property
    def total_pipeline_duration_ms(self) -> Optional[int]:
        durations = [self.parsing_duration_ms, self.enrichment_duration_ms, self.matching_duration_ms]
        valid = [d for d in durations if d is not None]
        return sum(valid) if valid else None
