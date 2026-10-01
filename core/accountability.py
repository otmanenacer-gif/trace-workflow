"""Étape 4 — épisodes d'accountability (orchestration).

    sorties de l'étape 3  →  Candidate Episode Builder (déterministe)
                          →  Accountability Episode Builder (LLM, au plus 1 appel par entretien)
                          →  Episode Validator (déterministe)
                          →  accountability_episodes.json
                             accountability_episode_manifest.json
                             accountability_episode_validation.json

L'étape 4 LIT les sorties de l'étape 3 sur le disque (practice_extractor.json,
interaction_signals.json, speaker_attribution_audit.json et leurs manifests) ; elle ne les
modifie jamais et ne relance jamais l'étape 3.

État de l'étape 3 :
- FAILED / absente (un des deux agents en échec, sortie manquante) → étape 4 BLOCKED :
  aucun appel, aucune sortie d'épisodes, le manifest dit pourquoi ;
- PARTIAL (un bloc en échec, `analysis_complete: false`) → étape 4 exécutée, mais son statut
  est PARTIAL et `analysis_complete: false` : jamais présentée comme complète.

Coût : 0 appel si aucun candidat ; sinon 1 appel par entretien (tous les candidats dans une
seule requête). Cache propre à l'étape 4 (data/cache/analysis/accountability_episode_builder/) :
la clé couvre la requête exacte (candidats, citations, tours, avertissements), l'agent, sa
version, son prompt, son schéma, le modèle et les paramètres. Modifier l'étape 4 ne relance
que l'étape 4 ; une étape 3 reprise du cache (mêmes objets) ne change pas la clé.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agents import accountability_episode_builder as builder
from agents.base import sha256_text
from core import accountability_candidates as candidates_mod
from core import accountability_episode_validator as episode_validator
from core import config
from core import speaker_attribution_auditor as speaker_audit
from core.analysis import (PRACTICE, INTERACTION, AUDITOR, STATUS_CACHED, STATUS_FAILED, STATUS_PARTIAL,
                           STATUS_PENDING, STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, _run_coroutine,
                           eligible_files)
from core.analysis_cache import AnalysisCache, compute_cache_key, write_json_atomic
from core.llm_client import LLMClient, LLMError, LLMSettings, Transport
from core.run_manager import save_metadata

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"
SPEC = builder.SPEC

STATUS_BLOCKED = "BLOCKED"  # étape 3 en échec ou absente : aucune analyse de l'étape 4
DONE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)

STAGE3_COMPLETE = "COMPLETE"
STAGE3_PARTIAL = "PARTIAL"
STAGE3_FAILED = "FAILED"
STAGE3_NOT_RUN = "NOT_RUN"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- État des sorties de l'étape 3 ---------------------------------------------------------

def stage3_state(analysis_dir: Path) -> dict:
    """État des deux agents de l'étape 3, lu sur le disque : COMPLETE, PARTIAL, FAILED ou NOT_RUN."""
    agents, reasons = {}, []
    for spec in (PRACTICE, INTERACTION):
        manifest = _read_json(analysis_dir / spec.manifest_filename)
        output = _read_json(analysis_dir / spec.output_filename)
        if manifest is None and output is None:
            agents[spec.name] = "MISSING"
            reasons.append(f"{spec.label} : aucune sortie.")
            continue
        status = (manifest or {}).get("status") or (output or {}).get("status")
        if output is None or status == STATUS_FAILED:
            agents[spec.name] = STATUS_FAILED
            message = ((manifest or {}).get("error") or {}).get("message")
            reasons.append(f"{spec.label} : échec" + (f" ({message})" if message else "") + ".")
            continue
        complete = output.get("analysis_complete", (manifest or {}).get("analysis_complete", True))
        if status == STATUS_PARTIAL or complete is False:
            agents[spec.name] = STATUS_PARTIAL
            reasons.append(f"{spec.label} : analyse incomplète (PARTIAL).")
        else:
            agents[spec.name] = status
    values = set(agents.values())
    if values == {"MISSING"}:
        state = STAGE3_NOT_RUN
    elif values & {"MISSING", STATUS_FAILED}:
        state = STAGE3_FAILED
    elif STATUS_PARTIAL in values:
        state = STAGE3_PARTIAL
    else:
        state = STAGE3_COMPLETE
    return {"status": state, "agents": agents, "reasons": reasons}


# --- Préparation (déterministe, gratuite) ----------------------------------------------------

@dataclass(frozen=True)
class Stage4Input:
    interview_id: str
    analysis_dir: Path
    transcript: dict
    source_sha256: str
    stage3: dict
    source_hashes: dict
    practices: list
    signals: list
    speaker_warnings: dict
    built: dict | None           # candidats (None si l'étape 3 est inutilisable)
    payload_json: str | None
    user_message: str | None

    @property
    def blocked(self) -> bool:
        return self.stage3["status"] in (STAGE3_FAILED, STAGE3_NOT_RUN)

    @property
    def candidate_count(self) -> int:
        return len(self.built["candidates"]) if self.built else 0

    @property
    def needs_llm(self) -> bool:
        return not self.blocked and self.candidate_count > 0


def prepare_stage4(ingestion: dict) -> Stage4Input:
    interview_dir = Path(ingestion["output_dir"])
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    transcript_path = interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    stage3 = stage3_state(analysis_dir)
    source_hashes = {
        "source_sha256": transcript["source"]["sha256"],
        "structured_transcript_sha256": _file_sha256(transcript_path),
        "practice_extractor_sha256": _file_sha256(analysis_dir / PRACTICE.output_filename),
        "interaction_signals_sha256": _file_sha256(analysis_dir / INTERACTION.output_filename),
        "speaker_audit_sha256": _file_sha256(analysis_dir / AUDITOR.output_filename),
    }
    base = dict(interview_id=transcript["interview_id"], analysis_dir=analysis_dir, transcript=transcript,
                source_sha256=transcript["source"]["sha256"], stage3=stage3, source_hashes=source_hashes)
    if stage3["status"] in (STAGE3_FAILED, STAGE3_NOT_RUN):
        return Stage4Input(**base, practices=[], signals=[], speaker_warnings={}, built=None, payload_json=None,
                           user_message=None)
    practices = (_read_json(analysis_dir / PRACTICE.output_filename) or {}).get("practices", [])
    signals = (_read_json(analysis_dir / INTERACTION.output_filename) or {}).get("signals", [])
    audit = _read_json(analysis_dir / AUDITOR.output_filename)
    warnings = speaker_audit.agent_warnings(audit, transcript) if audit and "items" in audit else {}
    built = candidates_mod.build_candidates(transcript, practices, signals, warnings)
    payload_json = user_message = None
    if built["candidates"]:
        payload = candidates_mod.build_payload(transcript, built, warnings)
        payload_json = candidates_mod.serialize_payload(payload)
        user_message = SPEC.user_template.format(
            interview_id=transcript["interview_id"], candidate_count=len(built["candidates"]),
            turn_count=len(payload["turns"]), payload_json=payload_json)
    return Stage4Input(**base, practices=practices, signals=signals, speaker_warnings=warnings, built=built,
                       payload_json=payload_json, user_message=user_message)


def cache_key_fields(prepared: Stage4Input, settings: LLMSettings) -> dict:
    return {"source_sha256": prepared.source_sha256, "transcript_sha256": sha256_text(prepared.user_message or ""),
            **SPEC.identity(), "model": settings.model, "request_params": settings.request_params()}


def plan_stage4(metadata: dict, interview_ids: list[str], settings: LLMSettings,
                cache: AnalysisCache | None = None, force: bool = False) -> dict:
    """Appels que provoquerait l'étape 4 (hors cache), sans rien appeler : 0 ou 1 par entretien."""
    cache = cache or AnalysisCache()
    wanted = set(interview_ids)
    plan = {"interviews": 0, "calls": 0, "cached": 0, "blocked": 0, "partial": 0, "no_candidate": 0,
            "candidates": 0, "per_interview": []}
    for info in eligible_files(metadata):
        if info["ingestion"]["interview_id"] not in wanted:
            continue
        prepared = prepare_stage4(info["ingestion"])
        plan["interviews"] += 1
        plan["candidates"] += prepared.candidate_count
        row = {"interview_id": prepared.interview_id, "stage3_status": prepared.stage3["status"],
               "candidate_count": prepared.candidate_count, "call": 0, "cached": False,
               "estimated_input_tokens": (candidates_mod.payload_estimate(prepared.user_message)
                                          if prepared.user_message else 0)}
        if prepared.blocked:
            plan["blocked"] += 1
        elif not prepared.needs_llm:
            plan["no_candidate"] += 1
        elif not force and settings.model and cache.contains(SPEC.name, compute_cache_key(
                cache_key_fields(prepared, settings))):
            plan["cached"] += 1
            row["cached"] = True
        else:
            plan["calls"] += 1
            row["call"] = 1
        plan["partial"] += prepared.stage3["status"] == STAGE3_PARTIAL
        plan["per_interview"].append(row)
    return plan


# --- Exécution sur un entretien ------------------------------------------------------------

def _counts(episodes: list[dict]) -> dict:
    kept = [e for e in episodes if e["validation_status"] != "rejected"]
    return {
        "episode_count": len(episodes),
        "accountability_episode_count": sum(e["episode_status"] == "accountability_episode" for e in kept),
        "ordinary_practice_count": sum(e["episode_status"] == "ordinary_practice" for e in kept),
        "uncertain_count": sum(e["episode_status"] == "uncertain" for e in kept),
        "rejected_episode_count": len(episodes) - len(kept),
        "needs_review_count": sum(e["needs_review"] for e in episodes),
    }


def _unmarked(prepared: Stage4Input) -> list[dict]:
    by_id = {p["practice_id"]: p for p in prepared.practices}
    return [{"practice_id": pid, "episode_status": "ordinary_practice",
             "classification": "deterministic_no_marker", "use_status": by_id[pid].get("use_status"),
             "practice_domain": by_id[pid].get("practice_domain"), "summary": by_id[pid].get("summary"),
             "turn_start": by_id[pid].get("turn_start"), "turn_end": by_id[pid].get("turn_end")}
            for pid in (prepared.built or {}).get("unmarked_practice_ids", [])]


async def run_stage4_interview(prepared: Stage4Input, client: LLMClient | None, cache: AnalysisCache,
                               settings: LLMSettings, force: bool = False) -> dict:
    """Étape 4 sur UN entretien. Ne lève pas d'exception : renvoie le résumé (manifest)."""
    label = f"{prepared.interview_id}/{SPEC.name}"
    out_dir = prepared.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / config.ACCOUNTABILITY_EPISODES_FILENAME
    estimated = candidates_mod.payload_estimate(prepared.user_message) if prepared.user_message else 0
    over = (estimated > candidates_mod.SINGLE_CALL_MAX_INPUT_TOKENS
            or prepared.candidate_count > candidates_mod.SINGLE_CALL_MAX_CANDIDATES)
    key_fields = cache_key_fields(prepared, settings) if prepared.needs_llm else None
    complete = prepared.stage3["status"] == STAGE3_COMPLETE
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "interview_id": prepared.interview_id,
        **SPEC.identity(),
        "candidate_builder_version": candidates_mod.CANDIDATE_BUILDER_VERSION,
        "validator_version": episode_validator.VALIDATOR_VERSION,
        "source_hashes": prepared.source_hashes,
        "payload_sha256": sha256_text(prepared.user_message) if prepared.user_message else None,
        "model": settings.model,
        "request_params": settings.request_params(),
        "cache_key": compute_cache_key(key_fields) if key_fields else None,
        "cache_hit": False,
        "llm_called": False,
        "status": STATUS_PENDING,
        "analysis_complete": False,
        "stage3_status": prepared.stage3["status"],
        "stage3_agents": prepared.stage3["agents"],
        "run_at": _now(),
        "created_at": None,
        "api_calls": 0,
        "billed_this_run": False,
        "usage": None,
        "duration_seconds": None,
        "response_model": None,
        "request_id": None,
        "stop_reason": None,
        "estimated_input_tokens": estimated,
        "single_call_threshold_tokens": candidates_mod.SINGLE_CALL_MAX_INPUT_TOKENS,
        "single_call_max_candidates": candidates_mod.SINGLE_CALL_MAX_CANDIDATES,
        "over_single_call_threshold": over,
        "candidate_count": prepared.candidate_count,
        "episode_count": 0, "accountability_episode_count": 0, "ordinary_practice_count": 0, "uncertain_count": 0,
        "rejected_episode_count": 0, "needs_review_count": 0,
        "unmarked_practice_count": len((prepared.built or {}).get("unmarked_practice_ids", [])),
        "validation_error_count": 0, "validation_warning_count": 0,
        "has_warnings": False,
        "output_file": None,
        "error": None,
    }
    validation_doc = {"interview_id": prepared.interview_id, "validator_version": episode_validator.VALIDATOR_VERSION,
                      "validated_at": _now(), "stage3": prepared.stage3}

    if prepared.blocked:
        manifest.update(status=STATUS_BLOCKED, error={
            "code": "STAGE3_UNAVAILABLE", "status_code": None, "request_id": None,
            "message": "Étape 4 non exécutée : sorties de l'étape 3 en échec ou absentes. "
                       + " ".join(prepared.stage3["reasons"])})
        episodes_path.unlink(missing_ok=True)  # jamais d'épisodes périmés à côté d'un blocage
        validation_doc.update(status=STATUS_BLOCKED, analysis_complete=False, available=False,
                              reason=manifest["error"]["message"])
        write_json_atomic(out_dir / config.ACCOUNTABILITY_VALIDATION_FILENAME, validation_doc)
        write_json_atomic(out_dir / config.ACCOUNTABILITY_MANIFEST_FILENAME, manifest)
        logger.info("Étape 4 %s : BLOCKED (étape 3 %s)", label, prepared.stage3["status"])
        return manifest

    output = {"episodes": [], "builder_notes": None}
    try:
        if prepared.needs_llm:
            entry = None if force else cache.load(key_fields, SPEC.output_model)
            if entry is not None:
                output = entry["output"]
                call = entry.get("call", {})
                manifest.update(cache_hit=True, created_at=entry.get("created_at"), usage=call.get("usage"),
                                duration_seconds=call.get("duration_seconds"), response_model=call.get("response_model"),
                                request_id=call.get("request_id"), stop_reason=call.get("stop_reason"))
                logger.info("Étape 4 %s : résultat en cache réutilisé (aucun appel API)", label)
            else:
                if client is None:
                    raise LLMError("NOT_CONFIGURED")
                if over:
                    logger.warning("Étape 4 %s : requête au-delà du seuil d'un appel unique (%d tokens estimés)",
                                   label, estimated)
                result = await client.complete_json(system_prompt=SPEC.system_prompt, user_content=prepared.user_message,
                                                    output_schema=SPEC.output_schema, response_model=SPEC.output_model,
                                                    label=label)
                output = result.data.model_dump(mode="json")
                cache.store(key_fields, output, {
                    "usage": result.usage, "attempts": result.attempts, "duration_seconds": result.duration_seconds,
                    "model_requested": result.model_requested, "response_model": result.response_model,
                    "request_id": result.request_id, "stop_reason": result.stop_reason})
                manifest.update(llm_called=True, created_at=_now(), api_calls=result.attempts, billed_this_run=True,
                                usage=result.usage, duration_seconds=result.duration_seconds,
                                response_model=result.response_model, request_id=result.request_id,
                                stop_reason=result.stop_reason)
        else:
            manifest["created_at"] = _now()
            logger.info("Étape 4 %s : aucun candidat, aucun appel API", label)

        validated = episode_validator.validate_episodes(
            output["episodes"], prepared.transcript, prepared.practices, prepared.signals,
            prepared.built["candidates"], prepared.speaker_warnings)
        report = validated["report"]
        if over:
            report["issues"].append({"object_id": prepared.interview_id, **episode_validator.make_issue(
                "PAYLOAD_OVER_THRESHOLD", estimated_input_tokens=estimated)})
            report["warning_count"] += 1
        if not complete:
            report["issues"].append({"object_id": prepared.interview_id, **episode_validator.make_issue(
                "STAGE3_INCOMPLETE", stage3_status=prepared.stage3["status"])})
            report["warning_count"] += 1
        counts = _counts(validated["episodes"])
        status = (STATUS_PARTIAL if not complete else STATUS_CACHED if manifest["cache_hit"]
                  else STATUS_SUCCESS_WITH_WARNINGS if episode_validator.has_problems(report) else STATUS_SUCCESS)
        document = {
            "agent": SPEC.name,
            "agent_version": SPEC.version,
            "schema_version": SPEC.schema_version,
            "prompt_sha256": SPEC.prompt_sha256,
            "candidate_builder_version": candidates_mod.CANDIDATE_BUILDER_VERSION,
            "validator_version": episode_validator.VALIDATOR_VERSION,
            "interview_id": prepared.interview_id,
            "model": settings.model if prepared.needs_llm else None,
            "status": status,
            "analysis_complete": complete,
            "stage3_status": prepared.stage3["status"],
            "cache_hit": manifest["cache_hit"],
            "llm_called": manifest["llm_called"],
            "generated_at": manifest["created_at"],
            "candidate_count": prepared.candidate_count,
            **counts,
            "unmarked_practice_count": manifest["unmarked_practice_count"],
            "validation_error_count": report["error_count"],
            "validation_warning_count": report["warning_count"],
            "builder_notes": output.get("builder_notes"),
            "source_hashes": prepared.source_hashes,
            "candidates": candidates_mod.public(prepared.built),
            "unmarked_practices": _unmarked(prepared),
            "episodes": validated["episodes"],
        }
        write_json_atomic(episodes_path, document)
        validation_doc.update(status=status, analysis_complete=complete, available=True,
                              candidate_summary=prepared.built["summary"], **report)
        manifest.update(status=status, analysis_complete=complete, **counts,
                        validation_error_count=report["error_count"], validation_warning_count=report["warning_count"],
                        has_warnings=not complete or episode_validator.has_problems(report),
                        output_file=config.ACCOUNTABILITY_EPISODES_FILENAME)
    except LLMError as error:
        manifest.update(status=STATUS_FAILED, error=error.to_dict(), api_calls=error.attempts,
                        llm_called=bool(error.attempts), usage=error.usage,
                        billed_this_run=bool(error.usage and error.usage.get("input_tokens")),
                        duration_seconds=error.duration_seconds, request_id=error.request_id)
    except Exception as exc:  # noqa: BLE001 — isolé à cet entretien, sans contenu d'entretien
        logger.error("Étape 4 %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        manifest.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())

    if manifest["status"] == STATUS_FAILED:
        episodes_path.unlink(missing_ok=True)
        validation_doc.update(status=STATUS_FAILED, analysis_complete=False, available=False,
                              reason=manifest["error"]["message"])
    write_json_atomic(out_dir / config.ACCOUNTABILITY_VALIDATION_FILENAME, validation_doc)
    write_json_atomic(out_dir / config.ACCOUNTABILITY_MANIFEST_FILENAME, manifest)
    logger.info("Étape 4 %s : %s (%d candidat(s))", label, manifest["status"], prepared.candidate_count)
    return manifest


# --- Un run ------------------------------------------------------------------------------------

SUMMARY_KEYS = ("status", "stage3_status", "analysis_complete", "candidate_count", "episode_count",
                "accountability_episode_count", "ordinary_practice_count", "uncertain_count", "rejected_episode_count",
                "unmarked_practice_count", "needs_review_count", "validation_error_count", "validation_warning_count",
                "has_warnings", "cache_hit", "llm_called", "api_calls", "billed_this_run", "usage", "duration_seconds",
                "estimated_input_tokens", "over_single_call_threshold", "error")


def _summary(prepared: Stage4Input, manifest: dict) -> dict:
    return {"interview_id": prepared.interview_id, "analysis_dir": str(prepared.analysis_dir), "run_at": _now(),
            **{k: manifest.get(k) for k in SUMMARY_KEYS}}


async def run_stage4_interviews(prepared_list: list[Stage4Input], settings: LLMSettings,
                                transport: Transport | None = None, cache: AnalysisCache | None = None,
                                force: bool = False) -> list[dict]:
    cache = cache or AnalysisCache()
    client = LLMClient(settings, transport=transport) if any(p.needs_llm for p in prepared_list) else None
    try:
        manifests = await asyncio.gather(*(run_stage4_interview(p, client, cache, settings, force)
                                           for p in prepared_list))
    finally:
        if client is not None:
            await client.aclose()
    return [_summary(p, m) for p, m in zip(prepared_list, manifests)]


def summarize_usage(summaries: list[dict]) -> dict:
    total = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_results": 0, "blocked": 0, "failed": 0,
             "partial": 0, "no_candidate": 0}
    for s in summaries:
        total["api_calls"] += s.get("api_calls") or 0
        total["cached_results"] += 1 if s.get("cache_hit") else 0
        total["blocked"] += s.get("status") == STATUS_BLOCKED
        total["failed"] += s.get("status") == STATUS_FAILED
        total["partial"] += s.get("status") == STATUS_PARTIAL
        total["no_candidate"] += s.get("status") != STATUS_BLOCKED and not s.get("candidate_count")
        if s.get("billed_this_run") and s.get("usage"):
            total["input_tokens"] += s["usage"].get("input_tokens") or 0
            total["output_tokens"] += s["usage"].get("output_tokens") or 0
    return total


def _step_state(metadata: dict) -> str:
    done = [f["accountability"]["status"] for f in eligible_files(metadata) if "accountability" in f]
    if not done:
        return "inactive"
    ok = sum(s in DONE_STATUSES for s in done)
    partial = sum(s == STATUS_PARTIAL for s in done)
    blocked = sum(s == STATUS_BLOCKED for s in done)
    failed = sum(s == STATUS_FAILED for s in done)
    state = f"terminé ({ok}/{len(eligible_files(metadata))} entretien(s)"
    state += f", {partial} incomplet(s)" if partial else ""
    state += f", {blocked} bloqué(s)" if blocked else ""
    return state + (f", {failed} échec(s))" if failed else ")")


def analyze_run_stage4(metadata: dict, interview_ids: list[str] | None = None, *, settings: LLMSettings | None = None,
                       transport: Transport | None = None, cache: AnalysisCache | None = None,
                       force: bool = False) -> dict:
    """Étape 4 sur les entretiens choisis d'un run (étape 3 déjà exécutée). Met à jour metadata.json.

    Lève LLMError (NOT_CONFIGURED) si la clé ou le modèle manque.
    """
    settings = settings or LLMSettings.from_env()
    if not settings.enabled:
        raise LLMError("NOT_CONFIGURED", ", ".join(settings.missing))
    files = eligible_files(metadata)
    if interview_ids is not None:
        wanted = set(interview_ids)
        files = [f for f in files if f["ingestion"]["interview_id"] in wanted]
    prepared = [prepare_stage4(f["ingestion"]) for f in files]
    summaries = _run_coroutine(run_stage4_interviews(prepared, settings, transport, cache, force))
    by_id = {s["interview_id"]: s for s in summaries}
    for info in files:
        info["accountability"] = by_id[info["ingestion"]["interview_id"]]
    metadata["pipeline"][config.ACCOUNTABILITY_STEP] = _step_state(metadata)
    metadata["last_accountability"] = {"analyzed_at": _now(), "model": settings.model,
                                       "interview_ids": [p.interview_id for p in prepared], "forced": force,
                                       "usage": summarize_usage(summaries)}
    save_metadata(metadata, Path(metadata["output_dir"]))
    return metadata
