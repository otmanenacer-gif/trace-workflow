"""Agent 1 — Practice Extractor.

Décrit ce que l'étudiant·e fait concrètement avec (ou sans) IAG, situation par
situation. Volontairement descriptif et « aveugle » à la théorie : ses
consignes (prompts/practice_extractor.md) ne mentionnent aucun cadre
sociologique. Il ne reçoit jamais la sortie de l'Interaction Signal Reader.

Entretien long (étape 3.7, voir core/practice_chunking.py) : le MÊME agent (mêmes
consignes, même schéma, même définition des catégories) lit l'entretien par blocs de
tours qui se chevauchent ; seul le message utilisateur (`CHUNK_USER_TEMPLATE`) annonce
un extrait. Les pratiques sont ensuite réunies de façon déterministe, sans LLM.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.base import TRANSCRIPT_DATA_NOTICE, AgentSpec, Evidence, Explicitness
from core import config

AGENT_NAME = "practice_extractor"
PRACTICE_EXTRACTOR_VERSION = "1.2"  # 1.2 : recherche explicite des non-usages et refus, usages hors études
# Schéma 1.1 : ajout de scope_qualifier, stated_frequency réservé aux fréquences.
# Schéma 1.2 : ajout de non_use_reason (forme du non-usage déclaré) et practice_domain (études, vie personnelle…).
PRACTICE_SCHEMA_VERSION = "1.2"

UseStatus = Literal["use", "non_use", "refusal", "hypothetical", "past_use"]
AssessmentContext = Literal["graded", "ungraded", "exam", "class", "personal", "unknown"]

# Statuts qui décrivent un non-usage : eux seuls portent un non_use_reason (sinon null).
NON_USE_STATUSES = ("non_use", "refusal")
# Ce sur quoi l'enquêté·e fait reposer le non-usage, tel qu'il ou elle le formule (jamais un motif supposé).
NON_USE_REASONS = ("not_stated", "preference", "personal_rule", "external_rule", "technical_limitation", "other")
NonUseReason = Literal[NON_USE_REASONS]  # type: ignore[valid-type]
# Domaine de la situation : les usages personnels et professionnels sont décrits aussi, pas filtrés.
PRACTICE_DOMAINS = ("academic", "personal", "professional", "mixed", "unknown")
PracticeDomain = Literal[PRACTICE_DOMAINS]  # type: ignore[valid-type]

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
    non_use_reason: NonUseReason | None = Field(
        description="Pour non_use / refusal uniquement : ce sur quoi l'enquêté·e fait reposer le non-usage "
                    "(not_stated si aucune raison n'est formulée) ; null pour les autres statuts.")
    practice_domain: PracticeDomain = Field(
        description="Domaine de la situation : academic, personal, professional, mixed ou unknown.")
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


# --- Entretien long : lecture par blocs (étape 3.7) ----------------------------------------

# Message utilisateur d'un bloc : mêmes consignes système, même schéma ; seul l'en-tête annonce un extrait.
# Il rappelle de décrire chaque conduite déclarée dans l'extrait (usages ET non-usages / refus) sans
# rien supposer de la suite : une conduite décrite ailleurs n'efface jamais celle de l'extrait.
CHUNK_USER_TEMPLATE = """Entretien à analyser : {interview_id}. Cet entretien est long : tu en reçois un EXTRAIT de {turn_count} tours consécutifs, du tour {first_turn_id} au tour {last_turn_id}.{overlap_note}

""" + TRANSCRIPT_DATA_NOTICE + """

<transcript>
{transcript_json}
</transcript>

Applique tes consignes à cet extrait et réponds avec l'objet JSON demandé. Décris chaque situation racontée dans l'extrait, y compris dans ses premiers et ses derniers tours, et en particulier chaque usage, non-usage, refus, usage passé ou hypothétique que l'enquêté·e y déclare, même s'il ou elle dit peut-être autre chose dans une autre partie de l'entretien : TRACE réunit ensuite les extraits sans jamais fusionner deux conduites différentes (un usage et un non-usage restent deux pratiques). Ne suppose rien sur les passages situés hors de l'extrait et n'y fais pas référence."""

CHUNK_OVERLAP_NOTE = (" Ses {overlap_turns} premiers tours (jusqu'au tour {overlap_last_turn_id}) figurent aussi à la fin "
                      "de l'extrait précédent : ils sont repris pour qu'une situation racontée à la jonction reste "
                      "lisible en entier.")
