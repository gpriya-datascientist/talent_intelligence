from sqlalchemy import String, Float, Boolean, DateTime, JSON, Integer, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional
import uuid
import enum

from backend.db.database import Base


class SkillType(str, enum.Enum):
    TECHNICAL = "technical"       # e.g. Python, React, DSP
    DOMAIN = "domain"             # e.g. Audio Engineering, UX Research
    SOFT = "soft"                 # e.g. Leadership, Communication
    TOOL = "tool"                 # e.g. Figma, Jira, LangChain


class SkillSource(str, enum.Enum):
    RESUME = "resume"             # Extracted from resume text
    GITHUB = "github"             # Inferred from GitHub repos
    SELF_DECLARED = "self_declared"  # Employee typed it themselves
    LLM_INFERRED = "llm_inferred"   # LLM inferred from project descriptions


class ProficiencyLevel(str, enum.Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id: Mapped[str] = mapped_column(String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)

    # Core skill info
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    # normalized_name lowercased + stripped, e.g. "react.js" -> "react"

    skill_type: Mapped[SkillType] = mapped_column(SAEnum(SkillType), nullable=False)
    source: Mapped[SkillSource] = mapped_column(SAEnum(SkillSource), nullable=False)
    proficiency: Mapped[ProficiencyLevel] = mapped_column(SAEnum(ProficiencyLevel), default=ProficiencyLevel.INTERMEDIATE)

    # Hands-on vs theoretical — key differentiator
    is_hands_on: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # True = used in real project/commit, False = mentioned in resume only

    # Recency signals
    last_used_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    years_of_experience: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recency_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # 0.0 - 1.0, computed by skill intelligence engine based on last_used_year

    # Confidence from LLM extraction
    extraction_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # 0.0 - 1.0, how confident the LLM was when extracting this skill

    # Supporting evidence — what the LLM saw that led to this skill
    evidence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # e.g. {"source_text": "built IIR filter in Python", "github_repo": "audio-eq"}

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship back to employee
    employee: Mapped["Employee"] = relationship("Employee", back_populates="skills")

    def __repr__(self) -> str:
        return f"<Skill {self.name} | {self.proficiency.value} | hands_on={self.is_hands_on}>"

    @property
    def weighted_score(self) -> float:
        """Quick score combining proficiency, recency and hands-on signal."""
        proficiency_map = {
            ProficiencyLevel.BEGINNER: 0.25,
            ProficiencyLevel.INTERMEDIATE: 0.50,
            ProficiencyLevel.ADVANCED: 0.75,
            ProficiencyLevel.EXPERT: 1.0,
        }
        base = proficiency_map.get(self.proficiency, 0.5)
        recency = self.recency_score or 0.5
        hands_on_bonus = 1.2 if self.is_hands_on else 1.0
        return round(min(base * recency * hands_on_bonus, 1.0), 4)
