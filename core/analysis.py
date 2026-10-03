"""Étape 3 — premier étage d'analyse par LLM.

Pour chaque entretien structuré :
0. Speaker Attribution Auditor (étape 3.5) → speaker_attribution_audit.json + speaker_audit_manifest.json
   présélection déterministe des tours à l'attribution douteuse ; UN appel LLM
   seulement s'il y a des candidats (aucun sinon). Le transcript n'est jamais modifié.
puis deux agents INDÉPENDANTS lancés EN PARALLÈLE :
- Practice Extractor           → practice_extractor.json  + practice_manifest.json
- Interaction Signal Reader    → interaction_signals.json + interaction_manifest.json
puis la validation déterministe des preuves → evidence_validation.json.

Indépendance : chaque agent reçoit seulement ses propres consignes, son propre
schéma et la même représentation compacte de l'entretien (avec, le cas échéant,
les `speaker_warning` de l'auditeur : le locuteur officiel reste inchangé).
Aucun des deux ne reçoit la sortie de l'autre ; aucune synthèse commune n'est produite.

Un appel = un entretien = un agent. Exception : un entretien LONG est lu par
chacun des deux agents en plusieurs blocs de tours qui se chevauchent (un appel
par bloc, cache par bloc) ; l'Interaction Signal Reader ajoute une lecture légère
à longue distance. Les sorties sont fusionnées de façon déterministe en UN
practice_extractor.json et UN interaction_signals.json (voir core/practice_chunking.py
et core/interaction_chunking.py). Un bloc en échec (réponse non conforme…) rend
l'analyse PARTIAL, sans effacer les blocs réussis. Un échec d'un agent n'efface
pas la sortie de l'autre ; un entretien en échec ne bloque pas les autres.

Avant la validation des preuves, les signaux passent par une étape déterministe de
cohérence et de sélectivité (core/signal_selectivity.py, étape 3.7) : les signaux
écartés restent visibles dans `set_aside_signals`, avec leur raison.

Sorties : data/outputs/<run_id>/interviews/<interview_id>/analysis/ (les
fichiers d'ingestion ne sont jamais modifiés).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import functools
import hashlib
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from agents import interaction_signal_reader, practice_extractor
from agents.base import AgentSpec, build_agent_input, render_user_message, serialize_agent_input, sha256_text
from core import config, evidence_validator, interaction_chunking, interpretation_guard, practice_chunking
from core import practice_selectivity, signal_selectivity
from core import speaker_attribution_auditor as speaker_audit
from core.analysis_cache import AnalysisCache, compute_cache_key, write_json_atomic
from core.llm_client import LLMError, LLMSettings
from core.run_manager import save_metadata
from core.schemas import STATUS_FAIL

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_SUCCESS_WITH_WARNINGS = "SUCCESS_WITH_WARNINGS"
STATUS_FAILED = "FAILED"
STATUS_CACHED = "CACHED"
STATUS_PARTIAL = "PARTIAL"      # entretien long : au moins un bloc (ou la lecture à longue distance) en échec
STATUS_TRUNCATED = "TRUNCATED"  # legacy : bloc tronqué (anciennes sorties par API), plus produit par le workflow
ANALYSIS_STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS,
                     STATUS_FAILED, STATUS_CACHED, STATUS_PARTIAL)
DONE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)

AGENTS: tuple[AgentSpec, ...] = (practice_extractor.SPEC, interaction_signal_reader.SPEC)
AUDITOR: AgentSpec = speaker_audit.SPEC
PRACTICE: AgentSpec = practice_extractor.SPEC
INTERACTION: AgentSpec = interaction_signal_reader.SPEC
LONG_DISTANCE: AgentSpec = interaction_signal_reader.LONG_DISTANCE_SPEC


CHUNK_USER_TEMPLATES = {PRACTICE.name: practice_extractor.CHUNK_USER_TEMPLATE,
                        INTERACTION.name: interaction_signal_reader.CHUNK_USER_TEMPLATE}


@functools.lru_cache(maxsize=None)
def chunk_spec(spec: AgentSpec) -> AgentSpec:
    """Le MÊME agent (consignes, schéma, version) ; seul le message utilisateur annonce un extrait."""
    return dataclasses.replace(spec, user_template=CHUNK_USER_TEMPLATES[spec.name])

StatusCallback = Callable[[str, str, str], None]  # (interview_id, agent, status)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- Préparation ------------------------------------------------------------------------

def eligible_files(metadata: dict) -> list[dict]:
    """Fichiers d'un run analysables : ingestion réussie et au moins un tour."""
    return [f for f in metadata.get("files", [])
            if "ingestion" in f and f["ingestion"]["status"] != STATUS_FAIL and f["ingestion"]["turn_count"] > 0]


@dataclass(frozen=True)
class PreparedInterview:
    interview_id: str
    transcript: dict            # transcript complet : réservé à la validation déterministe (jamais modifié)
    agent_input: dict           # représentation compacte envoyée aux agents
    transcript_json: str        # sérialisation exacte envoyée
    transcript_sha256: str
    source_sha256: str
    analysis_dir: Path
    speaker_warnings: dict = field(default_factory=dict)  # {turn_id: {suggested_speaker, confidence}}


def prepare_interview(ingestion: dict) -> PreparedInterview:
    interview_dir = Path(ingestion["output_dir"])
    transcript = json.loads((interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
    agent_input = build_agent_input(transcript)
    transcript_json = serialize_agent_input(agent_input)
    return PreparedInterview(
        interview_id=transcript["interview_id"],
        transcript=transcript,
        agent_input=agent_input,
        transcript_json=transcript_json,
        transcript_sha256=sha256_text(transcript_json),
        source_sha256=transcript["source"]["sha256"],
        analysis_dir=interview_dir / config.ANALYSIS_SUBDIR,
    )


def with_speaker_warnings(prepared: PreparedInterview, warnings: dict[str, dict]) -> PreparedInterview:
    """Même entretien, avec les avertissements de l'auditeur dans la représentation envoyée aux agents.

    Le transcript n'est pas touché. L'empreinte de la représentation (donc la clé de
    cache des agents) ne change que si les avertissements changent.
    """
    if not warnings and not prepared.speaker_warnings:
        return prepared
    agent_input = build_agent_input(prepared.transcript, warnings)
    transcript_json = serialize_agent_input(agent_input)
    return dataclasses.replace(prepared, agent_input=agent_input, transcript_json=transcript_json,
                               transcript_sha256=sha256_text(transcript_json), speaker_warnings=dict(warnings))


def cache_key_fields(spec: AgentSpec, prepared: PreparedInterview, settings: LLMSettings) -> dict:
    return {
        "source_sha256": prepared.source_sha256,
        "transcript_sha256": prepared.transcript_sha256,
        **{k: v for k, v in spec.identity().items()},
        "model": settings.model,
        "request_params": settings.request_params(),
    }


def audit_cache_key_fields(prepared: PreparedInterview, request: speaker_audit.AuditRequest,
                           settings: LLMSettings) -> dict:
    """Clé de cache de l'audit : l'extrait envoyé (et non l'entretien entier) + l'identité de l'auditeur."""
    return {
        "source_sha256": prepared.source_sha256,
        "transcript_sha256": request.audit_sha256,
        **AUDITOR.identity(),
        "model": settings.model,
        "request_params": settings.request_params(),
    }


def plan_analysis(metadata: dict, interview_ids: list[str], settings: LLMSettings,
                  cache: AnalysisCache | None = None, force: bool = False) -> dict:
    """Nombre d'appels LLM que provoquerait l'analyse (hors cache), sans rien appeler.

    L'audit des locuteurs ne coûte un appel que s'il existe des tours candidats. Tant que
    l'audit n'est pas en cache, les avertissements transmis aux agents (donc leur clé de
    cache) ne sont pas connus : leurs appels sont alors comptés comme probables
    (`exact: false`, le total est un maximum).
    """
    cache = cache or AnalysisCache()
    wanted = set(interview_ids)
    calls = cached = audit_calls = candidate_turns = 0
    exact = True
    for info in eligible_files(metadata):
        if info["ingestion"]["interview_id"] not in wanted:
            continue
        prepared = prepare_interview(info["ingestion"])
        request = speaker_audit.prepare_audit(prepared.transcript)
        candidate_turns += len(request.candidates)
        warnings: dict | None = {}
        if request.needs_llm:
            entry = None
            if not force and settings.model:
                entry = cache.load(audit_cache_key_fields(prepared, request, settings), AUDITOR.output_model)
            if entry is not None:
                cached += 1
                document = speaker_audit.build_audit_document(prepared.transcript, request, entry["output"])
                warnings = speaker_audit.agent_warnings(document, prepared.transcript)
            else:
                calls += 1
                audit_calls += 1
                warnings = None  # inconnus avant l'audit
        agent_prepared = with_speaker_warnings(prepared, warnings) if warnings is not None else None
        for spec in AGENTS:
            chunks = agent_chunks(spec, prepared, settings)
            if len(chunks) > 1:
                plan = _plan_chunked(spec, prepared, chunks, warnings, settings, cache, force)
                calls += plan["calls"]
                cached += plan["cached"]
                exact = exact and plan["exact"]
            elif agent_prepared is None:
                calls += 1
                exact = exact and force
            elif not force and settings.model and cache.contains(
                    spec.name, compute_cache_key(cache_key_fields(spec, agent_prepared, settings))):
                cached += 1
            else:
                calls += 1
    return {"interviews": len(wanted), "calls": calls, "cached": cached, "audit_calls": audit_calls,
            "candidate_turns": candidate_turns, "exact": exact}


def interaction_chunks(prepared: PreparedInterview, settings: LLMSettings) -> list:
    """Blocs de lecture de l'Interaction Signal Reader (un seul pour un entretien court)."""
    return interaction_chunking.plan_chunks(prepared.transcript, settings.interaction_chunk_tokens)


def practice_chunks(prepared: PreparedInterview, settings: LLMSettings) -> list:
    """Blocs de lecture du Practice Extractor (un seul pour un entretien court)."""
    return practice_chunking.plan_chunks(prepared.transcript, settings.practice_chunk_tokens)


def agent_chunks(spec: AgentSpec, prepared: PreparedInterview, settings: LLMSettings) -> list:
    return (practice_chunks if spec.name == PRACTICE.name else interaction_chunks)(prepared, settings)


def max_agent_calls(spec: AgentSpec, chunks: list) -> int:
    if spec.name == PRACTICE.name:
        return practice_chunking.max_practice_calls(chunks)
    return interaction_chunking.max_interaction_calls(chunks)


def chunk_request_for(spec: AgentSpec, transcript: dict, chunk, warnings: dict) -> dict:
    if spec.name == PRACTICE.name:
        return practice_chunking.chunk_request(transcript, chunk, warnings)
    return interaction_chunking.chunk_request(transcript, chunk, warnings)


def chunk_cache_key_fields(spec: AgentSpec, user_message: str, settings: LLMSettings) -> dict:
    """Clé de cache d'un bloc (ou de la lecture à longue distance) : le message EXACT envoyé.

    L'empreinte du message tient lieu d'empreinte de source : un bloc inchangé reste en
    cache même si une autre partie du fichier source a été modifiée.
    """
    digest = sha256_text(user_message)
    return {"source_sha256": digest, "transcript_sha256": digest, **spec.identity(), "model": settings.model,
            "request_params": settings.request_params()}


def _plan_chunked(spec: AgentSpec, prepared: PreparedInterview, chunks: list, warnings: dict | None,
                  settings: LLMSettings, cache: AnalysisCache, force: bool) -> dict:
    """Appels d'un entretien long : un par bloc absent du cache (plus, pour l'Interaction Reader,
    la lecture à longue distance)."""
    long_distance = spec.name == INTERACTION.name
    if warnings is None or force:  # avertissements inconnus avant l'audit, ou nouvelle analyse forcée
        return {"calls": max_agent_calls(spec, chunks), "cached": 0, "exact": False}
    calls = cached = 0
    outputs = []
    for chunk in chunks:
        request = chunk_request_for(spec, prepared.transcript, chunk, warnings)
        entry = cache.load(chunk_cache_key_fields(chunk_spec(spec), request["user_message"], settings),
                           spec.output_model) if settings.model else None
        if entry is None:
            calls += 1
        else:
            cached += 1
            outputs.append(entry["output"][spec.items_key])
    if not long_distance:
        return {"calls": calls, "cached": cached, "exact": True}
    if calls:  # la sélection à longue distance dépend des blocs : appel compté au plus
        return {"calls": calls + 1, "cached": cached, "exact": False}
    selection = interaction_chunking.select_long_distance_turns(prepared.transcript, chunks, outputs)
    if selection["needs_llm"]:
        request = interaction_chunking.long_distance_request(prepared.transcript, chunks, selection, warnings)
        if cache.contains(LONG_DISTANCE.name, compute_cache_key(
                chunk_cache_key_fields(LONG_DISTANCE, request["user_message"], settings))):
            cached += 1
        else:
            calls += 1
    return {"calls": calls, "cached": cached, "exact": True}


def _plan_agent_chunking(spec: AgentSpec, metadata: dict, interview_ids: list[str], settings: LLMSettings) -> list:
    wanted = set(interview_ids)
    plans = []
    for info in eligible_files(metadata):
        if info["ingestion"]["interview_id"] not in wanted:
            continue
        prepared = prepare_interview(info["ingestion"])
        chunks = agent_chunks(spec, prepared, settings)
        plans.append({"interview_id": prepared.interview_id, "chunk_count": len(chunks),
                      "chunking_used": len(chunks) > 1, "max_calls": max_agent_calls(spec, chunks),
                      "chunk_ranges": interaction_chunking.chunk_ranges(chunks)})
    return plans


def plan_interaction_chunking(metadata: dict, interview_ids: list[str], settings: LLMSettings) -> list[dict]:
    """Pour l'interface, AVANT exécution : blocs de lecture et nombre maximal d'appels Interaction Reader."""
    return _plan_agent_chunking(INTERACTION, metadata, interview_ids, settings)


def plan_practice_chunking(metadata: dict, interview_ids: list[str], settings: LLMSettings) -> list[dict]:
    """Pour l'interface, AVANT exécution : blocs de lecture et nombre maximal d'appels Practice Extractor."""
    return _plan_agent_chunking(PRACTICE, metadata, interview_ids, settings)


# --- Exécution d'un agent ---------------------------------------------------------------

def _validation_status(report: dict) -> str:
    return STATUS_SUCCESS_WITH_WARNINGS if evidence_validator.has_problems(report) else STATUS_SUCCESS


def _issue_count(report: dict | None) -> int:
    """Anomalies de validation (erreurs et avertissements ; les informations ne comptent pas)."""
    return (report.get("error_count") or 0) + (report.get("warning_count") or 0) if report else 0


# Post-traitement déterministe des objets d'un agent, AVANT la validation des preuves :
# (objets) -> (objets conservés, champs ajoutés au document, champs ajoutés au manifest).
PostProcess = Callable[[list[dict]], tuple[list[dict], dict, dict]]


def postprocess_for(spec: AgentSpec, prepared: PreparedInterview) -> PostProcess:
    if spec.name == INTERACTION.name:
        def select(items: list[dict]) -> tuple[list[dict], dict, dict]:
            result = signal_selectivity.apply(items, prepared.transcript, prepared.speaker_warnings)
            return (result["signals"], {"selectivity": result["summary"], "set_aside_signals": result["set_aside"]},
                    {"selectivity": result["summary"]})
        return select

    def cues(items: list[dict]) -> tuple[list[dict], dict, dict]:
        # normalisation formelle de non_use_reason et sélectivité (lien explicite avec une IAG), puis indices de
        # non-usage sur les pratiques conservées
        result = practice_selectivity.apply(items, prepared.transcript)
        items = result["practices"]
        found = practice_chunking.non_use_cues(prepared.transcript, items)
        return items, {"non_use_cues": found, "practice_selectivity": result["summary"],
                       "set_aside_practices": result["set_aside"]}, {
            "non_use_cues": {"cue_turn_count": found["cue_turn_count"],
                             "uncovered_count": len(found["uncovered_turn_ids"])},
            "practice_selectivity": result["summary"]}
    return cues


def _manifest_skeleton(spec: AgentSpec, prepared: PreparedInterview, settings: LLMSettings) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "interview_id": prepared.interview_id,
        **spec.identity(),
        "source_sha256": prepared.source_sha256,
        "transcript_sha256": prepared.transcript_sha256,
        "model": settings.model,
        "request_params": settings.request_params(),
        "cache_key": None,
        "cache_hit": False,
        "status": STATUS_PENDING,
        "run_at": _now(),
        "created_at": None,
        "api_calls": 0,
        "billed_this_run": False,
        "usage": None,
        "duration_seconds": None,
        "response_model": None,
        "request_id": None,
        "stop_reason": None,
        "validator_version": evidence_validator.VALIDATOR_VERSION,
        "guard_version": interpretation_guard.GUARD_VERSION,
        "speaker_warning_count": len(prepared.speaker_warnings),
        "item_count": 0,
        "invalid_evidence_count": 0,
        "needs_review_count": 0,
        "validation_issue_count": 0,
        "has_warnings": False,
        "output_file": None,
        "error": None,
    }


async def run_agent(spec: AgentSpec, prepared: PreparedInterview, client, cache: AnalysisCache,
                    settings: LLMSettings, force: bool = False, on_status: StatusCallback | None = None,
                    extra: dict | None = None, postprocess: PostProcess | None = None) -> dict:
    """Exécute UN agent sur UN entretien en UN appel. Ne lève pas d'exception : renvoie le manifest.

    L'agent ne reçoit que : ses consignes, son schéma, le transcript compact.
    `extra` : champs ajoutés au manifest et au document (ex. `chunking`).
    `postprocess` : traitement déterministe des objets avant validation (voir `postprocess_for`).
    """
    notify = on_status or (lambda *args: None)
    label = f"{prepared.interview_id}/{spec.name}"
    key_fields = cache_key_fields(spec, prepared, settings)
    manifest = {**_manifest_skeleton(spec, prepared, settings), "cache_key": compute_cache_key(key_fields),
                **(extra or {})}
    output_path = prepared.analysis_dir / spec.output_filename
    manifest_path = prepared.analysis_dir / spec.manifest_filename
    report = None
    try:
        entry = None if force else cache.load(key_fields, spec.output_model)
        if entry is not None:
            output = entry["output"]
            call = entry.get("call", {})
            manifest.update(cache_hit=True, created_at=entry.get("created_at"), usage=call.get("usage"),
                            duration_seconds=call.get("duration_seconds"), response_model=call.get("response_model"),
                            request_id=call.get("request_id"), stop_reason=call.get("stop_reason"))
            logger.info("Analyse %s : résultat en cache réutilisé (aucun appel API)", label)
        else:
            notify(prepared.interview_id, spec.name, STATUS_RUNNING)
            user_message = render_user_message(prepared.agent_input, prepared.transcript_json)
            result = await client.complete_json(
                system_prompt=spec.system_prompt, user_content=user_message,
                output_schema=spec.output_schema, response_model=spec.output_model, label=label,
            )
            output = result.data.model_dump(mode="json")
            cache.store(key_fields, output, _call_record(result))
            manifest.update(created_at=_now(), api_calls=result.attempts, billed_this_run=False,
                            usage=result.usage,
                            duration_seconds=result.duration_seconds, response_model=result.response_model,
                            request_id=result.request_id, stop_reason=result.stop_reason)

        items, doc_extra, manifest_extra = (postprocess or (lambda i: (i, {}, {})))(output[spec.items_key])
        validated = evidence_validator.validate_agent_output(
            spec.name, items, prepared.transcript, spec.id_letter, prepared.speaker_warnings)
        report = validated["report"]
        status = STATUS_CACHED if manifest["cache_hit"] else _validation_status(report)
        document = {
            "agent": spec.name,
            "agent_version": spec.version,
            "schema_version": spec.schema_version,
            "interview_id": prepared.interview_id,
            "model": settings.model,
            "status": status,
            "cache_hit": manifest["cache_hit"],
            "generated_at": manifest["created_at"],
            "item_count": len(validated["items"]),
            "needs_review_count": len(report["objects_needing_review"]),
            "invalid_evidence_count": report["invalid_evidence_count"],
            "validation_issue_count": _issue_count(report),
            **{k: v for k, v in output.items() if k != spec.items_key},
            **(extra or {}),
            **doc_extra,
            spec.items_key: validated["items"],
        }
        write_json_atomic(output_path, document)
        manifest.update(status=status, item_count=document["item_count"],
                        invalid_evidence_count=report["invalid_evidence_count"],
                        needs_review_count=document["needs_review_count"], output_file=spec.output_filename,
                        validation_issue_count=_issue_count(report),
                        has_warnings=evidence_validator.has_problems(report), **manifest_extra)
    except LLMError as error:
        manifest.update(status=STATUS_FAILED, error=error.to_dict(), api_calls=error.attempts,
                        usage=error.usage, billed_this_run=False,
                        duration_seconds=error.duration_seconds, request_id=error.request_id)
    except Exception as exc:  # noqa: BLE001 — isolé à cet agent, sans contenu d'entretien
        logger.error("Analyse %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        manifest.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())

    if manifest["status"] == STATUS_FAILED:
        output_path.unlink(missing_ok=True)  # pas de sortie périmée à côté d'un manifest en échec
    write_json_atomic(manifest_path, manifest)
    notify(prepared.interview_id, spec.name, manifest["status"])
    logger.info("Analyse %s : %s", label, manifest["status"])
    return {"manifest": manifest, "validation": report}


# --- Les deux agents : entretien court (1 appel) ou long (blocs) -------------------------

def _chunking_summary(spec: AgentSpec, chunks: list, settings: LLMSettings, transcript: dict) -> dict:
    practice = spec.name == PRACTICE.name
    overlap = practice_chunking.PRACTICE_OVERLAP_TURNS if practice else interaction_chunking.OVERLAP_TURNS
    return {
        "chunking_version": (practice_chunking.PRACTICE_CHUNKING_VERSION if practice
                             else interaction_chunking.CHUNKING_VERSION),
        "chunking_used": len(chunks) > 1,
        "chunk_count": len(chunks),
        "target_tokens": settings.practice_chunk_tokens if practice else settings.interaction_chunk_tokens,
        "estimated_tokens": sum(interaction_chunking.turn_costs(transcript)),
        "overlap_turns": overlap if len(chunks) > 1 else 0,
        "chunk_ranges": interaction_chunking.chunk_ranges(chunks),
    }


async def run_chunkable_agent(spec: AgentSpec, prepared: PreparedInterview, client,
                              cache: AnalysisCache, settings: LLMSettings, force: bool = False,
                              on_status: StatusCallback | None = None) -> dict:
    """Un des deux agents : un appel pour un entretien court (comportement inchangé), des blocs sinon."""
    try:
        chunks = agent_chunks(spec, prepared, settings)
    except Exception as exc:  # noqa: BLE001 — sans découpage, on garde le comportement historique
        logger.error("Découpage %s/%s : %s", prepared.interview_id, spec.name, type(exc).__name__)
        chunks = []
    if len(chunks) <= 1:
        extra = {"chunking": _chunking_summary(spec, chunks, settings, prepared.transcript)} if chunks else None
        return await run_agent(spec, prepared, client, cache, settings, force, on_status, extra,
                               postprocess_for(spec, prepared))
    return await run_chunked_agent(spec, prepared, chunks, client, cache, settings, force, on_status)


async def run_interaction_reader(spec: AgentSpec, prepared: PreparedInterview, client,
                                 cache: AnalysisCache, settings: LLMSettings, force: bool = False,
                                 on_status: StatusCallback | None = None) -> dict:
    """Interaction Signal Reader : un appel pour un entretien court, des blocs sinon."""
    return await run_chunkable_agent(spec, prepared, client, cache, settings, force, on_status)


def _call_record(result) -> dict:
    return {"usage": result.usage, "attempts": result.attempts, "duration_seconds": result.duration_seconds,
            "model_requested": result.model_requested, "response_model": result.response_model,
            "request_id": result.request_id, "stop_reason": result.stop_reason}


async def _cached_call(spec: AgentSpec, user_message: str, client, cache: AnalysisCache,
                       settings: LLMSettings, force: bool, label: str) -> dict:
    """UN appel (bloc ou lecture à longue distance), repris du cache s'il existe. Ne lève pas d'exception."""
    key_fields = chunk_cache_key_fields(spec, user_message, settings)
    record = {"cache_key": compute_cache_key(key_fields), "cache_hit": False, "status": STATUS_PENDING,
              "api_calls": 0, "billed_this_run": False, "usage": None, "duration_seconds": None,
              "response_model": None, "request_id": None, "stop_reason": None, "created_at": None, "error": None,
              "output": None}
    try:
        entry = None if force else cache.load(key_fields, spec.output_model)
        if entry is not None:
            call = entry.get("call", {})
            record.update(cache_hit=True, status=STATUS_CACHED, output=entry["output"],
                          created_at=entry.get("created_at"), usage=call.get("usage"),
                          duration_seconds=call.get("duration_seconds"), response_model=call.get("response_model"),
                          request_id=call.get("request_id"), stop_reason=call.get("stop_reason"))
            return record
        result = await client.complete_json(system_prompt=spec.system_prompt, user_content=user_message,
                                            output_schema=spec.output_schema, response_model=spec.output_model,
                                            label=label)
        output = result.data.model_dump(mode="json")
        cache.store(key_fields, output, _call_record(result))
        record.update(status=STATUS_SUCCESS, output=output, created_at=_now(), api_calls=result.attempts,
                      billed_this_run=False, usage=result.usage, duration_seconds=result.duration_seconds,
                      response_model=result.response_model, request_id=result.request_id,
                      stop_reason=result.stop_reason)
    except LLMError as error:
        record.update(status=STATUS_FAILED, error=error.to_dict(), api_calls=error.attempts, usage=error.usage,
                      duration_seconds=error.duration_seconds, billed_this_run=False, request_id=error.request_id)
    except Exception as exc:  # noqa: BLE001 — isolé à ce bloc, sans contenu d'entretien
        logger.error("Analyse %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        record.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())
    return record


CALL_OK = (STATUS_SUCCESS, STATUS_CACHED)


def _sum_usage(usages: list[dict]) -> dict | None:
    usages = [u for u in usages if u]
    if not usages:
        return None
    return {k: sum(u.get(k) or 0 for u in usages) for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


def _public_record(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "output"}


async def _long_distance_pass(prepared: PreparedInterview, chunks: list, succeeded: list, client,
                              cache: AnalysisCache, settings: LLMSettings, force: bool, label: str) -> tuple:
    """Interaction Reader : lecture à longue distance, si tous les blocs ont réussi. → (bilan, appel, signaux)."""
    transcript, warnings = prepared.transcript, prepared.speaker_warnings
    long_distance = {"status": "NOT_NEEDED", "llm_called": False, "selected_turn_count": 0,
                     "candidate_count": 0, "dropped_candidate_count": 0, "signals_returned": 0,
                     "signals_kept": 0, "signals_discarded_not_long_distance": 0}
    if len(succeeded) < len(chunks):
        long_distance["status"] = "SKIPPED_INCOMPLETE"  # refaite quand tous les blocs auront réussi
        return long_distance, None, []
    selection = interaction_chunking.select_long_distance_turns(
        transcript, chunks, [rec["output"]["signals"] for _, rec in succeeded])
    long_distance.update(selected_turn_count=len(selection["positions"]), candidate_count=selection["candidate_count"],
                         dropped_candidate_count=selection["dropped_count"])
    if not selection["needs_llm"]:
        return long_distance, None, []
    request = interaction_chunking.long_distance_request(transcript, chunks, selection, warnings)
    record = await _cached_call(LONG_DISTANCE, request["user_message"], client, cache, settings, force,
                                f"{label}/longue_distance")
    # lecture effectuée pendant cette exécution (modèle local : 0 appel API)
    llm_called = record["api_calls"] > 0 or record["status"] == STATUS_SUCCESS
    long_distance.update(_public_record(record), llm_called=llm_called, speaker_warning_count=len(request["warnings"]))
    kept: list[dict] = []
    if record["status"] in CALL_OK:
        position = {t["turn_id"]: i for i, t in enumerate(transcript["turns"])}
        returned = record["output"]["signals"]
        kept = [s for s in returned if interaction_chunking.is_long_distance(s, chunks, position)]
        long_distance.update(signals_returned=len(returned), signals_kept=len(kept),
                             signals_discarded_not_long_distance=len(returned) - len(kept))
    return long_distance, record, kept


async def run_chunked_agent(spec: AgentSpec, prepared: PreparedInterview, chunks: list, client,
                            cache: AnalysisCache, settings: LLMSettings, force: bool = False,
                            on_status: StatusCallback | None = None) -> dict:
    """Un des deux agents sur un entretien LONG. Ne lève pas d'exception : renvoie le manifest.

    1. un appel par bloc (en parallèle, concurrence bornée par le client), chaque bloc en cache ;
    2. Interaction Reader seulement : si tous les blocs ont réussi, lecture légère à longue distance ;
    3. fusion déterministe (pratiques ou signaux), post-traitement déterministe, puis validation des
       preuves sur le résultat fusionné (comme pour un appel unique).
    Un bloc en échec ou tronqué rend le statut PARTIAL : le résultat n'est jamais présenté comme complet.
    """
    notify = on_status or (lambda *args: None)
    label = f"{prepared.interview_id}/{spec.name}"
    interaction = spec.name == INTERACTION.name
    noun = "signals" if interaction else "practices"
    notes_key = "reading_notes" if interaction else "extraction_notes"
    transcript = prepared.transcript
    warnings = prepared.speaker_warnings
    chunking = _chunking_summary(spec, chunks, settings, transcript)
    # un entretien long a une clé de cache par bloc (chunking.chunks[].cache_key)
    manifest = {**_manifest_skeleton(spec, prepared, settings), "analysis_complete": False, "chunking": chunking}
    output_path = prepared.analysis_dir / spec.output_filename
    manifest_path = prepared.analysis_dir / spec.manifest_filename
    report = None
    try:
        notify(prepared.interview_id, spec.name, STATUS_RUNNING)
        requests = [chunk_request_for(spec, transcript, chunk, warnings) for chunk in chunks]
        records = await asyncio.gather(*(
            _cached_call(chunk_spec(spec), r["user_message"], client, cache, settings, force,
                         f"{label}/bloc{r['chunk'].index}") for r in requests))
        chunk_records = []
        for request, record in zip(requests, records):
            output = record["output"]
            chunk_records.append({**request["chunk"].describe(), "speaker_warning_count": len(request["warnings"]),
                                  f"{noun[:-1]}_count": len(output[spec.items_key]) if output else 0,
                                  **_public_record(record)})
        succeeded = [(r["chunk"], rec) for r, rec in zip(requests, records) if rec["status"] in CALL_OK]

        ld_record, kept_long = None, []
        if interaction:
            long_distance, ld_record, kept_long = await _long_distance_pass(
                prepared, chunks, succeeded, client, cache, settings, force, label)
            chunking["long_distance"] = long_distance

        calls = list(records) + ([ld_record] if ld_record else [])
        billed = [c for c in calls if c["billed_this_run"]]
        failed_chunks = [c for c in chunk_records if c["status"] not in CALL_OK]
        ld_failed = ld_record is not None and ld_record["status"] not in CALL_OK
        all_cached = all(c["cache_hit"] for c in calls)
        manifest.update(
            api_calls=sum(c["api_calls"] or 0 for c in calls), billed_this_run=bool(billed),
            usage=_sum_usage([c["usage"] for c in (billed or (calls if all_cached else []))]),
            duration_seconds=round(sum(c["duration_seconds"] or 0 for c in (billed or calls)), 3),
            response_model=next((c["response_model"] for c in calls if c["response_model"]), None),
            created_at=max((c["created_at"] for c in calls if c["created_at"]), default=None),
            cache_hit=all_cached and not failed_chunks and not ld_failed,
        )
        chunking.update(
            chunks=chunk_records, llm_calls_this_run=manifest["api_calls"],
            **{"interaction_calls_max" if interaction else "practice_calls_max": max_agent_calls(spec, chunks)},
            chunks_succeeded=len(succeeded), failed_chunks=[c["chunk"] for c in failed_chunks],
            truncated_chunks=[c["chunk"] for c in failed_chunks if c["status"] == STATUS_TRUNCATED])

        if not succeeded:
            first = failed_chunks[0]
            manifest.update(status=STATUS_FAILED, error=first["error"])
        else:
            if interaction:
                merged = interaction_chunking.merge_signals(
                    [{"label": f"bloc{chunk.index}", "chunk": chunk.index, "signals": rec["output"]["signals"]}
                     for chunk, rec in succeeded] + [{"label": "longue_distance", "chunk": None, "signals": kept_long}],
                    chunks, transcript)
                items = merged["signals"]
            else:
                merged = practice_chunking.merge_practices(
                    [{"chunk": chunk.index, "practices": rec["output"]["practices"]} for chunk, rec in succeeded],
                    chunks, transcript)
                items = merged["practices"]
            chunking.update({f"{noun}_before_dedup": merged["before"], f"{noun}_after_dedup": merged["after"],
                             "duplicates_removed": merged["duplicates_removed"]})
            items, doc_extra, manifest_extra = postprocess_for(spec, prepared)(items)
            validated = evidence_validator.validate_agent_output(
                spec.name, items, transcript, spec.id_letter, warnings)
            report = validated["report"]
            complete = not failed_chunks and not ld_failed
            report["analysis_complete"] = complete
            status = (STATUS_PARTIAL if not complete else STATUS_CACHED if manifest["cache_hit"]
                      else _validation_status(report))
            error = None
            if not complete:
                parts = [f"bloc {c['chunk']} ({c['first_turn_id']} → {c['last_turn_id']}) : {c['error']['message']}"
                         for c in failed_chunks]
                if ld_failed:
                    parts.append(f"lecture à longue distance : {ld_record['error']['message']}")
                error = {"code": "PARTIAL_ANALYSIS", "status_code": None, "request_id": None, "message": (
                    f"Analyse INCOMPLÈTE : {len(succeeded)}/{len(chunks)} bloc(s) réussi(s). " + " ; ".join(parts)
                    + ". Les blocs réussis sont conservés (cache TRACE) : relancez l'étape, seuls les blocs "
                    "manquants sont refaits par le modèle local.")}
            notes = [f"Bloc {chunk.index} : {rec['output'][notes_key]}" for chunk, rec in succeeded
                     if rec["output"].get(notes_key)]
            if ld_record is not None and ld_record["status"] in CALL_OK and ld_record["output"].get("reading_notes"):
                notes.append(f"Lecture à longue distance : {ld_record['output']['reading_notes']}")
            document = {
                "agent": spec.name,
                "agent_version": spec.version,
                "schema_version": spec.schema_version,
                "interview_id": prepared.interview_id,
                "model": settings.model,
                "status": status,
                "analysis_complete": complete,
                "cache_hit": manifest["cache_hit"],
                "generated_at": manifest["created_at"],
                "item_count": len(validated["items"]),
                "needs_review_count": len(report["objects_needing_review"]),
                "invalid_evidence_count": report["invalid_evidence_count"],
                "validation_issue_count": _issue_count(report),
                notes_key: "\n".join(notes) or None,
                "chunking": chunking,
                **doc_extra,
                spec.items_key: validated["items"],
            }
            write_json_atomic(output_path, document)
            manifest.update(status=status, item_count=document["item_count"], analysis_complete=complete,
                            invalid_evidence_count=report["invalid_evidence_count"], error=error,
                            needs_review_count=document["needs_review_count"], output_file=spec.output_filename,
                            validation_issue_count=_issue_count(report),
                            has_warnings=not complete or evidence_validator.has_problems(report), **manifest_extra)
    except Exception as exc:  # noqa: BLE001 — isolé à cet agent, sans contenu d'entretien
        logger.error("Analyse %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        report = None
        manifest.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())

    if manifest["status"] == STATUS_FAILED:
        output_path.unlink(missing_ok=True)
    write_json_atomic(manifest_path, manifest)
    notify(prepared.interview_id, spec.name, manifest["status"])
    logger.info("Analyse %s : %s (%d bloc(s))", label, manifest["status"], len(chunks))
    return {"manifest": manifest, "validation": report}


async def run_chunked_interaction(spec: AgentSpec, prepared: PreparedInterview, chunks: list, client,
                                  cache: AnalysisCache, settings: LLMSettings, force: bool = False,
                                  on_status: StatusCallback | None = None) -> dict:
    """Interaction Signal Reader sur un entretien LONG (voir `run_chunked_agent`)."""
    return await run_chunked_agent(spec, prepared, chunks, client, cache, settings, force, on_status)


# --- Audit de l'attribution des locuteurs (avant les deux agents) ------------------------

def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


async def run_speaker_audit(prepared: PreparedInterview, client, cache: AnalysisCache,
                            settings: LLMSettings, force: bool = False,
                            on_status: StatusCallback | None = None) -> dict:
    """Audite l'attribution des locuteurs d'UN entretien. Ne lève pas d'exception.

    A. présélection déterministe (gratuite) ; B. un seul appel LLM, et seulement s'il
    existe des candidats, sur un extrait (candidats + voisins). Le transcript n'est
    JAMAIS modifié : seuls speaker_attribution_audit.json et speaker_audit_manifest.json
    sont écrits. Renvoie {"manifest", "document", "warnings"} ; `warnings` alimente
    les `speaker_warning` transmis aux agents.
    """
    notify = on_status or (lambda *args: None)
    label = f"{prepared.interview_id}/{AUDITOR.name}"
    transcript_path = prepared.analysis_dir.parent / config.STRUCTURED_TRANSCRIPT_FILENAME
    request = speaker_audit.prepare_audit(prepared.transcript)
    key_fields = audit_cache_key_fields(prepared, request, settings) if request.needs_llm else None
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "interview_id": prepared.interview_id,
        **AUDITOR.identity(),
        "heuristics_version": speaker_audit.HEURISTICS_VERSION,
        "source_sha256": prepared.source_sha256,
        "structured_transcript_sha256": _file_sha256(transcript_path),
        "transcript_sha256": request.audit_sha256,  # empreinte de l'EXTRAIT envoyé (null sans appel)
        "model": settings.model,
        "request_params": settings.request_params(),
        "audit_mode": "heuristics_and_llm" if request.needs_llm else "heuristics_only",
        "llm_called": False,
        "cache_key": compute_cache_key(key_fields) if key_fields else None,
        "cache_hit": False,
        "status": STATUS_PENDING,
        "run_at": _now(),
        "created_at": None,
        "api_calls": 0,
        "billed_this_run": False,
        "usage": None,
        "duration_seconds": None,
        "response_model": None,
        "request_id": None,
        "stop_reason": None,
        "validator_version": evidence_validator.VALIDATOR_VERSION,
        "guard_version": interpretation_guard.GUARD_VERSION,
        "candidate_count": len(request.candidates),
        "sent_candidate_count": len(request.sent),
        "excerpt_turn_count": len(request.excerpt_turn_ids),
        "review_count": 0,
        "speaker_warning_count": 0,
        "invalid_evidence_count": 0,
        "has_warnings": False,
        "output_file": None,
        "error": None,
    }
    output = None
    if request.needs_llm:
        try:
            entry = None if force else cache.load(key_fields, AUDITOR.output_model)
            if entry is not None:
                output = entry["output"]
                call = entry.get("call", {})
                manifest.update(cache_hit=True, created_at=entry.get("created_at"), usage=call.get("usage"),
                                duration_seconds=call.get("duration_seconds"),
                                response_model=call.get("response_model"), request_id=call.get("request_id"),
                                stop_reason=call.get("stop_reason"))
                logger.info("Audit %s : résultat en cache réutilisé (aucun appel API)", label)
            else:
                notify(prepared.interview_id, AUDITOR.name, STATUS_RUNNING)
                result = await client.complete_json(
                    system_prompt=AUDITOR.system_prompt,
                    user_content=speaker_audit.render_audit_message(request.audit_input, request.audit_json),
                    output_schema=AUDITOR.output_schema, response_model=AUDITOR.output_model, label=label,
                )
                output = result.data.model_dump(mode="json")
                call = {"usage": result.usage, "attempts": result.attempts,
                        "duration_seconds": result.duration_seconds, "model_requested": result.model_requested,
                        "response_model": result.response_model, "request_id": result.request_id,
                        "stop_reason": result.stop_reason}
                cache.store(key_fields, output, call)
                manifest.update(llm_called=True, created_at=_now(), api_calls=result.attempts,
                                billed_this_run=False,
                                usage=result.usage, duration_seconds=result.duration_seconds,
                                response_model=result.response_model, request_id=result.request_id,
                                stop_reason=result.stop_reason)
        except LLMError as error:
            manifest.update(status=STATUS_FAILED, error=error.to_dict(), api_calls=error.attempts,
                            llm_called=bool(error.attempts), usage=error.usage,
                            billed_this_run=False,
                            duration_seconds=error.duration_seconds, request_id=error.request_id)
        except Exception as exc:  # noqa: BLE001 — isolé à l'audit, sans contenu d'entretien
            logger.error("Audit %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                         "".join(traceback.format_tb(exc.__traceback__)))
            output = None
            manifest.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())
    else:
        manifest["created_at"] = _now()
        logger.info("Audit %s : aucun tour candidat, aucun appel API", label)

    # Même en cas d'échec de l'appel, les candidats déterministes sont signalés (sans suggestion).
    document = speaker_audit.build_audit_document(prepared.transcript, request, output)
    if manifest["status"] != STATUS_FAILED:
        manifest["status"] = (STATUS_CACHED if manifest["cache_hit"] else
                              STATUS_SUCCESS_WITH_WARNINGS if document["error_count"] or document["warning_count"]
                              else STATUS_SUCCESS)
    document = {"interview_id": document["interview_id"], "status": manifest["status"], **document}
    document.update(model=settings.model if request.needs_llm else None, llm_called=manifest["llm_called"],
                    cache_hit=manifest["cache_hit"], generated_at=manifest["created_at"] or _now(),
                    structured_transcript_sha256=manifest["structured_transcript_sha256"])
    warnings = speaker_audit.agent_warnings(document, prepared.transcript)
    manifest.update(review_count=document["review_count"], speaker_warning_count=len(warnings),
                    invalid_evidence_count=document["invalid_evidence_count"],
                    has_warnings=bool(document["error_count"] or document["warning_count"]),
                    output_file=AUDITOR.output_filename)
    write_json_atomic(prepared.analysis_dir / AUDITOR.output_filename, document)
    write_json_atomic(prepared.analysis_dir / AUDITOR.manifest_filename, manifest)
    notify(prepared.interview_id, AUDITOR.name, manifest["status"])
    logger.info("Audit %s : %s (%d candidat(s), %d à vérifier)", label, manifest["status"],
                document["candidate_count"], document["review_count"])
    return {"manifest": manifest, "document": document, "warnings": warnings}


# --- Un entretien, puis un corpus -------------------------------------------------------

def _validation_document(prepared: PreparedInterview, results: dict[str, dict], audit: dict | None = None) -> dict:
    agents = {}
    for spec in AGENTS:
        result = results[spec.name]
        if result["validation"] is not None:
            agents[spec.name] = result["validation"]
        else:
            agents[spec.name] = {"available": False, "status": result["manifest"]["status"],
                                 "reason": (result["manifest"]["error"] or {}).get("message")}
    available = [a for a in agents.values() if a.get("available")]
    document = {
        "interview_id": prepared.interview_id,
        "validator_version": evidence_validator.VALIDATOR_VERSION,
        "guard_version": interpretation_guard.GUARD_VERSION,
        "validated_at": _now(),
        "total_evidence": sum(a["evidence_count"] for a in available),
        "total_invalid_evidence": sum(a["invalid_evidence_count"] for a in available),
        "total_objects_needing_review": sum(len(a["objects_needing_review"]) for a in available),
        "agents": agents,
    }
    if audit is not None:  # les totaux ci-dessus portent sur les deux agents ; l'audit a son propre bilan
        audit_doc = audit["document"]
        document["speaker_audit"] = {
            "status": audit_doc["status"],
            "candidate_count": audit_doc["candidate_count"],
            "review_count": audit_doc["review_count"],
            "evidence_count": audit_doc["evidence_count"],
            "invalid_evidence_count": audit_doc["invalid_evidence_count"],
            "error_count": audit_doc["error_count"],
            "warning_count": audit_doc["warning_count"],
            "issues": audit_doc["issues"],
        }
    return document


def _agent_summary(manifest: dict) -> dict:
    keys = ("status", "cache_hit", "item_count", "invalid_evidence_count", "needs_review_count", "api_calls",
            "billed_this_run", "usage", "duration_seconds", "error", "has_warnings", "speaker_warning_count",
            "validation_issue_count", "selectivity", "non_use_cues")
    summary = {k: manifest.get(k) for k in keys}
    chunking = manifest.get("chunking")
    if chunking:  # lecture en un appel ou par blocs
        summary["chunking"] = {k: chunking.get(k) for k in (
            "chunking_used", "chunk_count", "chunks_succeeded", "failed_chunks", "truncated_chunks",
            "signals_before_dedup", "signals_after_dedup", "practices_before_dedup", "practices_after_dedup",
            "llm_calls_this_run", "interaction_calls_max", "practice_calls_max")}
        summary["chunking"]["long_distance_status"] = (chunking.get("long_distance") or {}).get("status")
        summary["analysis_complete"] = manifest.get("analysis_complete", manifest.get("status") != STATUS_FAILED)
    return summary


def _audit_summary(manifest: dict) -> dict:
    keys = ("status", "audit_mode", "llm_called", "cache_hit", "candidate_count", "review_count",
            "speaker_warning_count", "invalid_evidence_count", "api_calls", "billed_this_run", "usage",
            "duration_seconds", "error", "has_warnings")
    return {k: manifest.get(k) for k in keys}


async def analyze_interview(prepared: PreparedInterview, client, cache: AnalysisCache,
                            settings: LLMSettings, force: bool = False,
                            on_status: StatusCallback | None = None) -> dict:
    """Audit des locuteurs, puis les deux agents EN PARALLÈLE sur un entretien, puis evidence_validation.json.

    Les agents reçoivent les avertissements de l'audit comme métadonnées secondaires ;
    ils ne voient jamais la sortie l'un de l'autre.
    """
    prepared.analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        audit = await run_speaker_audit(prepared, client, cache, settings, force, on_status)
    except Exception as exc:  # noqa: BLE001 — filet de sécurité : l'audit ne bloque jamais les agents
        logger.error("Audit %s : %s", prepared.interview_id, type(exc).__name__)
        audit = None
        for filename in (AUDITOR.output_filename, AUDITOR.manifest_filename):  # pas de sortie périmée
            (prepared.analysis_dir / filename).unlink(missing_ok=True)
    if audit is not None:
        prepared = with_speaker_warnings(prepared, audit["warnings"])
    outcomes = await asyncio.gather(
        *(run_chunkable_agent(spec, prepared, client, cache, settings, force, on_status) for spec in AGENTS),
        return_exceptions=True,
    )
    results = {}
    for spec, outcome in zip(AGENTS, outcomes):
        if isinstance(outcome, BaseException):  # filet de sécurité : run_agent ne lève pas
            logger.error("Analyse %s/%s : %s", prepared.interview_id, spec.name, type(outcome).__name__)
            outcome = {"manifest": {"status": STATUS_FAILED, "error": LLMError(
                "UNEXPECTED_ERROR", type(outcome).__name__).to_dict()}, "validation": None}
        results[spec.name] = outcome
    validation = _validation_document(prepared, results, audit)
    write_json_atomic(prepared.analysis_dir / config.EVIDENCE_VALIDATION_FILENAME, validation)
    audit_summary = _audit_summary(audit["manifest"]) if audit is not None else {
        "status": STATUS_FAILED, "error": LLMError("UNEXPECTED_ERROR").to_dict()}
    return {
        "interview_id": prepared.interview_id,
        "analysis_dir": str(prepared.analysis_dir),
        "analyzed_at": _now(),
        "invalid_evidence_count": validation["total_invalid_evidence"],
        "speaker_audit": audit_summary,
        "agents": {name: _agent_summary(r["manifest"]) for name, r in results.items()},
    }


async def analyze_interviews(prepared_list: list[PreparedInterview], settings: LLMSettings, *, client,
                             cache: AnalysisCache | None = None, force: bool = False,
                             on_status: StatusCallback | None = None) -> list[dict]:
    """Analyse plusieurs entretiens.

    `client` : client des agents (`complete_json`, `aclose`), injecté par le pipeline local
    (core/local_agent_runner.LocalAgentRunner : modèle local via Ollama, aucune API) ou par les tests (agents simulés).
    """
    cache = cache or AnalysisCache()
    try:
        for prepared in prepared_list:
            for spec in (AUDITOR, *AGENTS):
                (on_status or (lambda *a: None))(prepared.interview_id, spec.name, STATUS_PENDING)
        return list(await asyncio.gather(
            *(analyze_interview(p, client, cache, settings, force, on_status) for p in prepared_list)))
    finally:
        await client.aclose()


def _run_coroutine(coroutine):
    """asyncio.run, y compris si une boucle tourne déjà dans ce fil (exécution dans un autre fil)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def summarize_usage(summaries: list[dict]) -> dict:
    """Consommation réelle de CETTE exécution (les résultats en cache ne coûtent rien)."""
    total = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "cached_results": 0, "failed": 0, "partial": 0,
             "duration_seconds": 0.0}
    for summary in summaries:
        steps = list(summary["agents"].values())
        if summary.get("speaker_audit"):
            steps.append(summary["speaker_audit"])
        for agent in steps:
            total["api_calls"] += agent.get("api_calls") or 0
            total["cached_results"] += 1 if agent.get("cache_hit") else 0
            total["failed"] += 1 if agent.get("status") == STATUS_FAILED else 0
            total["partial"] += 1 if agent.get("status") == STATUS_PARTIAL else 0
            if agent.get("billed_this_run") and agent.get("usage"):
                for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
                    total[key] += agent["usage"].get(key) or 0
                total["duration_seconds"] += agent.get("duration_seconds") or 0.0
    total["duration_seconds"] = round(total["duration_seconds"], 1)
    return total


def _step_state(metadata: dict, agent: str) -> str:
    eligible = eligible_files(metadata)
    analyzed = [f["analysis"]["agents"][agent]["status"] for f in eligible if "analysis" in f]
    if not analyzed:
        return "inactive"
    done = sum(s in DONE_STATUSES for s in analyzed)
    failed = sum(s == STATUS_FAILED for s in analyzed)
    partial = sum(s == STATUS_PARTIAL for s in analyzed)
    state = f"terminé ({done}/{len(eligible)} entretien(s)"
    state += f", {partial} incomplet(s)" if partial else ""
    return state + (f", {failed} échec(s))" if failed else ")")


def analyze_run(metadata: dict, interview_ids: list[str] | None = None, *, client,
                settings: LLMSettings | None = None, cache: AnalysisCache | None = None, force: bool = False,
                on_status: StatusCallback | None = None) -> dict:
    """Analyse les entretiens choisis d'un run ingéré et met à jour metadata.json.

    `interview_ids=None` : tous les entretiens analysables. `client` : client des agents, injecté par le
    pipeline local (modèle local via Ollama, aucune clé, aucune API).
    """
    settings = settings or LLMSettings.from_env()
    files = eligible_files(metadata)
    if interview_ids is not None:
        wanted = set(interview_ids)
        files = [f for f in files if f["ingestion"]["interview_id"] in wanted]
    prepared = [prepare_interview(f["ingestion"]) for f in files]
    summaries = _run_coroutine(analyze_interviews(prepared, settings, client=client, cache=cache, force=force,
                                                   on_status=on_status))

    by_id = {s["interview_id"]: s for s in summaries}
    for info in files:
        info["analysis"] = by_id[info["ingestion"]["interview_id"]]
    for spec in AGENTS:
        metadata["pipeline"][spec.pipeline_step] = _step_state(metadata, spec.name)
    metadata["last_analysis"] = {
        "analyzed_at": _now(),
        "model": settings.model,
        "interview_ids": [p.interview_id for p in prepared],
        "forced": force,
        "max_concurrency": None,  # legacy (anciens appels API simultanés) : sans objet dans le workflow
        "usage": summarize_usage(summaries),
    }
    save_metadata(metadata, Path(metadata["output_dir"]))
    return metadata
