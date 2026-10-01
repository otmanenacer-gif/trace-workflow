"""Agent 1 — Practice Extractor.

Décrit ce que l'étudiant·e fait concrètement avec (ou sans) IAG, situation par
situation. Volontairement descriptif et « aveugle » à la théorie : ses
consignes (prompts/practice_extractor.md) ne mentionnent aucun cadre
sociologique. Il ne reçoit jamais la sortie de l'Interaction Signal Reader.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.base import AgentSpec, Evidence, Explicitness
from core import config

AGENT_NAME = "practice_extractor"
PRACTICE_EXTRACTOR_VERSION = "1.1"
PRACTICE_SCHEMA_VERSION = "1.1"  # 1.1 : ajout de scope_qualifier, stated_frequency réservé aux fréquences

UseStatus = Literal["use", "non_use", "refusal", "hypothetical", "past_use"]
AssessmentContext = Literal["graded", "ungraded", "exam", "class", "personal", "unknown"]

# Qualificatifs de portée (« je l'utilise surtout pour… ») : ce ne sont pas des fréquences.
# Leur présence dans stated_frequency rend la sortie non conforme au schéma.
SCOPE_QUALIFIER_TERMS = ("surtout", "principalement", "essentiellement", "notamment", "en particulier")
_SCOPE_QUALIFIER_RE = re.compile(r"\b(?:" + "|".join(SCOPE_QUALIFIER_TERMS) + r")\b", re.IGNORECASE)


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
    stated_frequency: str | None = Field(
        description="Fréquence uniquement (« parfois », « souvent », « rarement », « une fois », « jamais », "
                    "« toujours »…), sinon null. « surtout » n'est pas une fréquence : voir scope_qualifier.")
    scope_qualifier: str | None = Field(
        description="Qualificatif de portée employé par l'enquêté·e (« surtout », « principalement »…), sinon null.")
    assessment_context: AssessmentContext
    other_actors: list[str]
    evidence: list[Evidence] = Field(min_length=1)
    explicitness: Explicitness
    uncertainty_note: str | None

    @field_validator("stated_frequency")
    @classmethod
    def _frequency_is_not_scope_qualifier(cls, value: str | None) -> str | None:
        if value is not None and _SCOPE_QUALIFIER_RE.search(value):
            raise ValueError("stated_frequency contient un qualificatif de portée (à placer dans scope_qualifier)")
        return value


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
