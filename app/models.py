"""SQLAlchemy data models for FitBuddy."""

from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Index,
    sql,
)
from sqlalchemy.orm import relationship
from app.database import Base


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


class FitnessGoal(str, Enum):
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    GENERAL_WELLNESS = "general_wellness"


class WorkoutIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExperienceLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Equipment(str, Enum):
    BODYWEIGHT = "bodyweight"
    DUMBBELLS = "dumbbells"
    FULL_GYM = "full_gym"


class TipCategory(str, Enum):
    NUTRITION = "nutrition"
    RECOVERY = "recovery"


class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default=UserRole.USER.value, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    profile = relationship(
        "FitnessProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    workout_plans = relationship(
        "WorkoutPlan",
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    tips = relationship(
        "FitnessTip",
        back_populates="owner",
        cascade="all, delete-orphan",
    )


class UserSession(Base):
    """Server-controlled user session records for secure cookie authentication."""
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(128), unique=True, index=True, nullable=False)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="sessions")


class FitnessProfile(Base):
    """Fitness profile associated with an adult user (18+)."""
    __tablename__ = "fitness_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    display_name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)  # >= 18
    weight_kg = Column(Float, nullable=False)  # > 0.0
    fitness_goal = Column(String(50), nullable=False)
    workout_intensity = Column(String(50), nullable=False)
    experience_level = Column(String(50), nullable=False)
    equipment = Column(String(50), nullable=False)
    available_minutes = Column(Integer, nullable=False)
    exercise_limitations = Column(Text, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="profile")


class WorkoutPlan(Base):
    """Generated workout plan entity with versioning and parent reference."""
    __tablename__ = "workout_plans"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version = Column(Integer, default=1, nullable=False)
    parent_plan_id = Column(
        Integer,
        ForeignKey("workout_plans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    profile_snapshot = Column(Text, nullable=False)  # Stored JSON string
    validated_plan_json = Column(Text, nullable=False)  # Stored JSON string
    model_identifier = Column(String(100), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    owner = relationship("User", back_populates="workout_plans")
    parent_plan = relationship("WorkoutPlan", remote_side=[id], backref="revisions")
    feedbacks_received = relationship(
        "PlanFeedback",
        foreign_keys="PlanFeedback.source_plan_id",
        back_populates="source_plan",
        cascade="all, delete-orphan",
    )


class PlanFeedback(Base):
    """Feedback submitted on a workout plan triggering a revision."""
    __tablename__ = "plan_feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    source_plan_id = Column(
        Integer,
        ForeignKey("workout_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resulting_plan_id = Column(
        Integer,
        ForeignKey("workout_plans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    feedback_text = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    source_plan = relationship(
        "WorkoutPlan",
        foreign_keys=[source_plan_id],
        back_populates="feedbacks_received",
    )
    resulting_plan = relationship(
        "WorkoutPlan",
        foreign_keys=[resulting_plan_id],
    )


class FitnessTip(Base):
    """Nutrition or recovery tip generated for a user."""
    __tablename__ = "fitness_tips"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    goal = Column(String(50), nullable=False)
    category = Column(String(50), nullable=False)  # "nutrition" or "recovery"
    tip_text = Column(Text, nullable=False)
    model_identifier = Column(String(100), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    owner = relationship("User", back_populates="tips")
