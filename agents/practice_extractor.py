"""Agent 1 — Practice Extractor.

Décrit ce que l'étudiant·e fait concrètement avec (ou sans) IAG, situation par
situation. Volontairement descriptif et « aveugle » à la théorie : ses
consignes (prompts/practice_extractor.md) ne mentionnent aucun cadre
sociologique. Il ne reçoit jamais la sortie de l'Interaction Signal Reader.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.base import AgentSpec, Evidence, Explicitness
from core import config

AGENT_NAME = "practice_extractor"
PRACTICE_EXTRACTOR_VERSION = "1.0"
PRACTICE_SCHEMA_VERSION = "1.0"

UseStatus = Literal["use", "non_use", "refusal", "hypothetical", "past_use"]
AssessmentContext = Literal["graded", "ungraded", "exam", "class", "personal", "unknown"]


class Practice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Phrase descriptive au discours rapporté.")
    turn_start: str
    turn_end: str
    use_status: UseStatus
    academic_task: str | None
    discipline: str | None
    context: str
    ai_tool: list[str]
    student_action_before: list[str]
    ai_action: list[str]
    student_action_after: list[str]
    stated_reason: list[str] = Field(description="Raisons données par l'enquêté·e uniquement.")
    explicit_constraints: list[str]
    verification_or_control: list[str]
    stated_frequency: str | None
    assessment_context: AssessmentContext
    other_actors: list[str]
    evidence: list[Evidence] = Field(min_length=1)
    explicitness: Explicitness
    uncertainty_note: str | None


class PracticeExtractorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    practices: list[Practice]
    extraction_notes: str | None


SPEC = AgentSpec(
    name=AGENT_NAME,
    label="Practice Extractor",
    version=PRACTICE_EXTRACTOR_VERSION,
    schema_version=PRACTICE_SCHEMA_VERSION,
    prompt_path=config.PROMPTS_DIR / "practice_extractor.md",
    output_model=PracticeExtractorOutput,
    items_key="practices",
    id_letter="P",
    output_filename="practice_extractor.json",
    manifest_filename="practice_manifest.json",
    pipeline_step=config.PRACTICE_STEP,
)
