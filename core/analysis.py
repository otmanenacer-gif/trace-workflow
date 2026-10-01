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

Un appel = un entretien = un agent. Un échec d'un agent n'efface pas la
sortie de l'autre ; un entretien en échec ne bloque pas les autres.

Sorties : data/outputs/<run_id>/interviews/<interview_id>/analysis/ (les
fichiers d'ingestion ne sont jamais modifiés).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
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
from core import config, evidence_validator, interpretation_guard
from core import speaker_attribution_auditor as speaker_audit
from core.analysis_cache import AnalysisCache, compute_cache_key, write_json_atomic
from core.llm_client import LLMClient, LLMError, LLMSettings, Transport
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
ANALYSIS_STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS,
                     STATUS_FAILED, STATUS_CACHED)
DONE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)

AGENTS: tuple[AgentSpec, ...] = (practice_extractor.SPEC, interaction_signal_reader.SPEC)
AUDITOR: AgentSpec = speaker_audit.SPEC

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
            if agent_prepared is None:
                calls += 1
                exact = exact and force
            elif not force and settings.model and cache.contains(
                    spec.name, compute_cache_key(cache_key_fields(spec, agent_prepared, settings))):
                cached += 1
            else:
                calls += 1
    return {"interviews": len(wanted), "calls": calls, "cached": cached, "audit_calls": audit_calls,
            "candidate_turns": candidate_turns, "exact": exact}


# --- Exécution d'un agent ---------------------------------------------------------------

def _validation_status(report: dict) -> str:
    return STATUS_SUCCESS_WITH_WARNINGS if evidence_validator.has_problems(report) else STATUS_SUCCESS


async def run_agent(spec: AgentSpec, prepared: PreparedInterview, client: LLMClient, cache: AnalysisCache,
                    settings: LLMSettings, force: bool = False, on_status: StatusCallback | None = None) -> dict:
    """Exécute UN agent sur UN entretien. Ne lève pas d'exception : renvoie le manifest.

    L'agent ne reçoit que : ses consignes, son schéma, le transcript compact.
    """
    notify = on_status or (lambda *args: None)
    label = f"{prepared.interview_id}/{spec.name}"
    key_fields = cache_key_fields(spec, prepared, settings)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "interview_id": prepared.interview_id,
        **spec.identity(),
        "source_sha256": prepared.source_sha256,
        "transcript_sha256": prepared.transcript_sha256,
        "model": settings.model,
        "request_params": settings.request_params(),
        "cache_key": compute_cache_key(key_fields),
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
        "has_warnings": False,
        "output_file": None,
        "error": None,
    }
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
            call = {"usage": result.usage, "attempts": result.attempts, "duration_seconds": result.duration_seconds,
                    "model_requested": result.model_requested, "response_model": result.response_model,
                    "request_id": result.request_id, "stop_reason": result.stop_reason}
            cache.store(key_fields, output, call)
            manifest.update(created_at=_now(), api_calls=result.attempts, billed_this_run=True, usage=result.usage,
                            duration_seconds=result.duration_seconds, response_model=result.response_model,
                            request_id=result.request_id, stop_reason=result.stop_reason)

        validated = evidence_validator.validate_agent_output(
            spec.name, output[spec.items_key], prepared.transcript, spec.id_letter, prepared.speaker_warnings)
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
            **{k: v for k, v in output.items() if k != spec.items_key},
            spec.items_key: validated["items"],
        }
        write_json_atomic(output_path, document)
        manifest.update(status=status, item_count=document["item_count"],
                        invalid_evidence_count=report["invalid_evidence_count"],
                        needs_review_count=document["needs_review_count"], output_file=spec.output_filename,
                        has_warnings=evidence_validator.has_problems(report))
    except LLMError as error:
        manifest.update(status=STATUS_FAILED, error=error.to_dict(), api_calls=error.attempts,
                        usage=error.usage, billed_this_run=bool(error.usage and error.usage.get("input_tokens")),
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


# --- Audit de l'attribution des locuteurs (avant les deux agents) ------------------------

def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


async def run_speaker_audit(prepared: PreparedInterview, client: LLMClient, cache: AnalysisCache,
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
                manifest.update(llm_called=True, created_at=_now(), api_calls=result.attempts, billed_this_run=True,
                                usage=result.usage, duration_seconds=result.duration_seconds,
                                response_model=result.response_model, request_id=result.request_id,
                                stop_reason=result.stop_reason)
        except LLMError as error:
            manifest.update(status=STATUS_FAILED, error=error.to_dict(), api_calls=error.attempts,
                            llm_called=bool(error.attempts), usage=error.usage,
                            billed_this_run=bool(error.usage and error.usage.get("input_tokens")),
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
            "billed_this_run", "usage", "duration_seconds", "error", "has_warnings", "speaker_warning_count")
    return {k: manifest.get(k) for k in keys}


def _audit_summary(manifest: dict) -> dict:
    keys = ("status", "audit_mode", "llm_called", "cache_hit", "candidate_count", "review_count",
            "speaker_warning_count", "invalid_evidence_count", "api_calls", "billed_this_run", "usage",
            "duration_seconds", "error", "has_warnings")
    return {k: manifest.get(k) for k in keys}


async def analyze_interview(prepared: PreparedInterview, client: LLMClient, cache: AnalysisCache,
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
        *(run_agent(spec, prepared, client, cache, settings, force, on_status) for spec in AGENTS),
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


async def analyze_interviews(prepared_list: list[PreparedInterview], settings: LLMSettings,
                             transport: Transport | None = None, cache: AnalysisCache | None = None,
                             force: bool = False, on_status: StatusCallback | None = None) -> list[dict]:
    """Analyse plusieurs entretiens ; la concurrence des appels API est bornée par le client."""
    cache = cache or AnalysisCache()
    client = LLMClient(settings, transport=transport)
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
             "cache_read_input_tokens": 0, "cached_results": 0, "failed": 0, "duration_seconds": 0.0}
    for summary in summaries:
        steps = list(summary["agents"].values())
        if summary.get("speaker_audit"):
            steps.append(summary["speaker_audit"])
        for agent in steps:
            total["api_calls"] += agent.get("api_calls") or 0
            total["cached_results"] += 1 if agent.get("cache_hit") else 0
            total["failed"] += 1 if agent.get("status") == STATUS_FAILED else 0
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
    state = f"terminé ({done}/{len(eligible)} entretien(s)"
    return state + (f", {failed} échec(s))" if failed else ")")


def analyze_run(metadata: dict, interview_ids: list[str] | None = None, *, settings: LLMSettings | None = None,
                transport: Transport | None = None, cache: AnalysisCache | None = None, force: bool = False,
                on_status: StatusCallback | None = None) -> dict:
    """Analyse les entretiens choisis d'un run ingéré et met à jour metadata.json.

    `interview_ids=None` : tous les entretiens analysables. Lève LLMError
    (NOT_CONFIGURED) si la clé ou le modèle manque ; aucune analyse n'est
    alors lancée.
    """
    settings = settings or LLMSettings.from_env()
    if not settings.enabled:
        raise LLMError("NOT_CONFIGURED", ", ".join(settings.missing))
    files = eligible_files(metadata)
    if interview_ids is not None:
        wanted = set(interview_ids)
        files = [f for f in files if f["ingestion"]["interview_id"] in wanted]
    prepared = [prepare_interview(f["ingestion"]) for f in files]
    summaries = _run_coroutine(analyze_interviews(prepared, settings, transport, cache, force, on_status))

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
        "max_concurrency": settings.max_concurrency,
        "usage": summarize_usage(summaries),
    }
    save_metadata(metadata, Path(metadata["output_dir"]))
    return metadata
