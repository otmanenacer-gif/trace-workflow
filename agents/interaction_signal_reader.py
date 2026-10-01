"""Agent 2 — Interaction Signal Reader.

Relève des signaux discursifs OBSERVABLES dans la manière dont l'étudiant·e
raconte ses pratiques (hésitations, autocorrections, minimisations, affects
explicitement nommés, références au jugement d'autrui, contradictions…).
Il ne lit pas les pensées : aucun état psychologique n'est inféré.
Il ne reçoit jamais la sortie du Practice Extractor.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.base import AgentSpec, Evidence, Explicitness
from core import config

AGENT_NAME = "interaction_signal_reader"
INTERACTION_SIGNAL_READER_VERSION = "1.2"
# Schéma 1.1 : ajout du type preference_statement.
# Schéma 1.2 : ajout du type metadiscursive_self_evaluation.
INTERACTION_SCHEMA_VERSION = "1.2"

SIGNAL_TYPES = (
    "explicit_emotion",
    "hesitation",
    "self_correction",
    "self_reformulation",
    "minimization",
    "intensification",
    "restriction",
    "exception",
    "contrast",
    "cross_turn_contradiction",
    "vocabulary_shift",
    "modalization",
    "normative_formulation",
    "generalization",
    "reference_to_teacher_judgment",
    "reference_to_peer_judgment",
    "reference_to_rule",
    "distancing_from_own_practice",
    "attribution_to_others",
    "transcribed_laughter",
    "transcribed_silence",
    "significant_repetition",
    "pronoun_shift",
    "preference_statement",
    "metadiscursive_self_evaluation",
    "other",
)
SignalType = Literal[SIGNAL_TYPES]  # type: ignore[valid-type]


class InteractionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_ids: list[str] = Field(min_length=1)
    signal_type: SignalType
    surface_form: str = Field(description="Mots exacts qui portent le signal.")
    description: str = Field(description="Description purement textuelle, sans interprétation.")
    topic: str | None
    evidence: list[Evidence] = Field(min_length=1)
    explicit_affect: str | None = Field(description="Affect nommé par l'enquêté·e dans une citation, sinon null.")
    cross_turn_reference: str | None
    explicitness: Explicitness
    needs_human_review: bool


class InteractionSignalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signals: list[InteractionSignal]
    reading_notes: str | None


SPEC = AgentSpec(
    name=AGENT_NAME,
    label="Interaction Signal Reader",
    version=INTERACTION_SIGNAL_READER_VERSION,
    schema_version=INTERACTION_SCHEMA_VERSION,
    prompt_path=config.PROMPTS_DIR / "interaction_signal_reader.md",
    output_model=InteractionSignalOutput,
    items_key="signals",
    id_letter="S",
    output_filename="interaction_signals.json",
    manifest_filename="interaction_manifest.json",
    pipeline_step=config.INTERACTION_STEP,
)
