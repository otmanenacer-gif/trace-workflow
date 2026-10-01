"""Agent 2 — Interaction Signal Reader.

Relève des signaux discursifs OBSERVABLES dans la manière dont l'étudiant·e
raconte ses pratiques (hésitations, autocorrections, minimisations, affects
explicitement nommés, références au jugement d'autrui, contradictions…).
Il ne lit pas les pensées : aucun état psychologique n'est inféré.
Il ne reçoit jamais la sortie du Practice Extractor.

Entretien long (voir core/interaction_chunking.py) : le MÊME agent (mêmes consignes,
même schéma) lit l'entretien par blocs de tours qui se chevauchent (`CHUNK_USER_TEMPLATE` :
seul le message utilisateur annonce un extrait), puis une lecture restreinte à
longue distance (`LONG_DISTANCE_SPEC`) rapproche des passages de blocs différents
(contradictions, répétitions, changements de vocabulaire), sur une sélection compacte de tours.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.base import TRANSCRIPT_DATA_NOTICE, AgentSpec, Evidence, Explicitness
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


# --- Entretien long : lecture par blocs ---------------------------------------------------

# Message utilisateur d'un bloc : mêmes consignes système, même schéma ; seul l'en-tête annonce un extrait.
CHUNK_USER_TEMPLATE = """Entretien à analyser : {interview_id}. Cet entretien est long : tu en reçois un EXTRAIT de {turn_count} tours consécutifs, du tour {first_turn_id} au tour {last_turn_id}.{overlap_note}

""" + TRANSCRIPT_DATA_NOTICE + """

<transcript>
{transcript_json}
</transcript>

Applique tes consignes à cet extrait et réponds avec l'objet JSON demandé. Relève tous les signaux présents dans l'extrait, y compris dans ses premiers et ses derniers tours : TRACE réunit ensuite les extraits et supprime les doublons. Les contradictions, répétitions ou changements de vocabulaire avec des passages situés hors de l'extrait font l'objet d'une lecture séparée : ne les suppose pas."""

CHUNK_OVERLAP_NOTE = (" Ses {overlap_turns} premiers tours (jusqu'au tour {overlap_last_turn_id}) figurent aussi à la fin "
                      "de l'extrait précédent : ils sont repris pour que les phénomènes situés à la jonction restent lisibles.")


# --- Entretien long : lecture à longue distance -------------------------------------------

LONG_DISTANCE_AGENT_NAME = "interaction_signal_reader_long_distance"
LONG_DISTANCE_VERSION = "1.0"
LONG_DISTANCE_SIGNAL_TYPES = ("cross_turn_contradiction", "significant_repetition", "vocabulary_shift")
LongDistanceSignalType = Literal[LONG_DISTANCE_SIGNAL_TYPES]  # type: ignore[valid-type]


class LongDistanceSignal(InteractionSignal):
    """Même signal que InteractionSignal, restreint aux phénomènes entre passages éloignés."""

    signal_type: LongDistanceSignalType


class LongDistanceSignalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signals: list[LongDistanceSignal]
    reading_notes: str | None


LONG_DISTANCE_USER_TEMPLATE = """Entretien à analyser : {interview_id}. Cet entretien long a été lu en {chunk_count} blocs successifs ({chunk_ranges}). Tu reçois une SÉLECTION de {turn_count} tours non consécutifs ; le champ `blocks` de chaque tour indique le ou les blocs où il figure.

""" + TRANSCRIPT_DATA_NOTICE + """

<transcript>
{transcript_json}
</transcript>

Applique tes consignes à cette sélection : relève seulement les contradictions, répétitions remarquables ou changements de vocabulaire entre des tours de blocs différents, et réponds avec l'objet JSON demandé."""

LONG_DISTANCE_SPEC = AgentSpec(
    name=LONG_DISTANCE_AGENT_NAME,
    label="Interaction Signal Reader (lecture à longue distance)",
    version=LONG_DISTANCE_VERSION,
    schema_version=INTERACTION_SCHEMA_VERSION,
    prompt_path=config.PROMPTS_DIR / "interaction_long_distance_reader.md",
    output_model=LongDistanceSignalOutput,
    items_key="signals",
    id_letter="S",
    output_filename=SPEC.output_filename,      # contribue au même interaction_signals.json
    manifest_filename=SPEC.manifest_filename,
    pipeline_step=config.INTERACTION_STEP,
    user_template=LONG_DISTANCE_USER_TEMPLATE,
)
