"""Étape 5 — configuration et trajectoire intra-entretien (orchestration).

    sorties de l'étape 4  →  préparation déterministe (core/trajectory_candidates.py)
                          →  Trajectory Mapper (LLM, au plus 1 appel par entretien)
                          →  validateur déterministe (core/trajectory_validator.py)
                          →  student_trajectory.json
                             student_trajectory_validation.json
                             student_trajectory_manifest.json

L'étape 5 LIT les sorties de l'étape 4 (accountability_episodes.json et sa validation) et, pour vérifier
un ancrage temporel ou contextuel, le minimum de l'étape 3 (pratiques, types de signaux, audit des
locuteurs) ; elle ne modifie et ne relance jamais ni l'étape 3 ni l'étape 4. Elle travaille sur UN
entretien à la fois : aucune comparaison entre entretiens.

État de l'étape 4 :
- absente, en échec, bloquée, ou PÉRIMÉE (calculée sur d'autres sorties de l'étape 3 que celles du dossier)
  → étape 5 BLOCKED : aucun appel, aucune sortie de trajectoire, le manifest dit pourquoi ;
- PARTIAL → étape 5 exécutée, statut PARTIAL, `analysis_complete: false`.

Coût : 0 appel si moins de `MIN_ITEMS_FOR_LLM` éléments utilisables (configuration `no_clear_pattern`
déterministe) ; sinon 1 appel par entretien, représentation normalisée. Pas de découpage : un entretien
au-delà du seuil est signalé (`over_single_call_threshold`), jamais traité en augmentant max_tokens.
Cache propre à l'étape 5 (data/cache/analysis/trajectory_mapper/) : la clé couvre la requête exacte,
l'agent, sa version, son prompt, son schéma, le modèle et les paramètres. Modifier l'étape 5 n'invalide
que l'étape 5 ; une étape 4 restaurée depuis fichiers (mêmes épisodes) donne la même clé.
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

from agents import trajectory_mapper as mapper
from agents.base import sha256_text
from core import accountability, config
from core import speaker_attribution_auditor as speaker_audit
from core import trajectory_candidates as tc
from core import trajectory_validator as validator
from core.analysis import (AUDITOR, INTERACTION, PRACTICE, STATUS_CACHED, STATUS_FAILED, STATUS_PARTIAL,
                           STATUS_PENDING, STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, _run_coroutine,
                           eligible_files)
from core.analysis_cache import AnalysisCache, compute_cache_key, write_json_atomic
from core.llm_client import LLMClient, LLMError, LLMSettings, Transport
from core.run_manager import save_metadata

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"
SPEC = mapper.SPEC
STATUS_BLOCKED = accountability.STATUS_BLOCKED
DONE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)
STAGE4_USABLE_STATUSES = DONE_STATUSES

STAGE4_COMPLETE = "COMPLETE"
STAGE4_PARTIAL = "PARTIAL"
STAGE4_FAILED = "FAILED"
STAGE4_STALE = "STALE"
STAGE4_NOT_RUN = "NOT_RUN"

INSUFFICIENT_SUMMARY = ("Matériau insuffisant pour décrire une configuration : moins de deux épisodes ou pratiques "
                        "utilisables dans cet entretien. Aucun appel au modèle.")


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


def stage3_hashes(interview_dir: Path) -> dict:
    """Empreintes actuelles des fichiers que l'étape 4 a consommés (mêmes clés que son `source_hashes`)."""
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    transcript_path = interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME
    transcript = _read_json(transcript_path) or {}
    return {
        "source_sha256": (transcript.get("source") or {}).get("sha256"),
        "structured_transcript_sha256": _file_sha256(transcript_path),
        "practice_extractor_sha256": _file_sha256(analysis_dir / PRACTICE.output_filename),
        "interaction_signals_sha256": _file_sha256(analysis_dir / INTERACTION.output_filename),
        "speaker_audit_sha256": _file_sha256(analysis_dir / AUDITOR.output_filename),
    }


def stage4_state(interview_dir: Path) -> dict:
    """État de l'étape 4 lu sur le disque : COMPLETE, PARTIAL, FAILED (dont BLOCKED), STALE ou NOT_RUN."""
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    manifest = _read_json(analysis_dir / config.ACCOUNTABILITY_MANIFEST_FILENAME)
    document = _read_json(analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME)
    restored = bool((manifest or {}).get("restored_from_file"))
    if manifest is None and document is None:
        return {"status": STAGE4_NOT_RUN, "reasons": ["Étape 4 : aucune sortie."], "restored": False}
    status = (document or {}).get("status") or (manifest or {}).get("status")
    if document is None or status in (STATUS_FAILED, STATUS_BLOCKED):
        message = ((manifest or {}).get("error") or {}).get("message")
        return {"status": STAGE4_FAILED, "restored": restored, "stage4_run_status": status,
                "reasons": [f"Étape 4 : {status or 'sortie absente'}" + (f" ({message})" if message else "") + "."]}
    current = stage3_hashes(interview_dir)
    recorded = document.get("source_hashes") or {}
    changed = sorted(k for k, v in current.items() if recorded.get(k) != v)
    if changed:
        return {"status": STAGE4_STALE, "restored": restored, "stage4_run_status": status,
                "reasons": ["Étape 4 périmée : elle a été calculée sur d'autres sorties de l'étape 3 ou une autre "
                            f"transcription ({', '.join(changed)}). Relancez ou restaurez l'étape 4."]}
    if status == STATUS_PARTIAL or document.get("analysis_complete") is False:
        return {"status": STAGE4_PARTIAL, "restored": restored, "stage4_run_status": status,
                "reasons": ["Étape 4 incomplète (PARTIAL)."]}
    return {"status": STAGE4_COMPLETE, "restored": restored, "stage4_run_status": status, "reasons": []}


# --- Préparation (déterministe, gratuite) ----------------------------------------------------------

@dataclass(frozen=True)
class Stage5Input:
    interview_id: str
    analysis_dir: Path
    transcript: dict
    source_sha256: str
    stage4: dict
    source_hashes: dict
    speaker_warnings: dict
    material: dict | None          # None si l'étape 4 est inutilisable
    request: dict | None = None    # UN message (jamais de découpage)

    @property
    def blocked(self) -> bool:
        return self.stage4["status"] not in (STAGE4_COMPLETE, STAGE4_PARTIAL)

    @property
    def item_count(self) -> int:
        return self.material["summary"]["item_count"] if self.material else 0

    @property
    def needs_llm(self) -> bool:
        return not self.blocked and self.item_count >= tc.MIN_ITEMS_FOR_LLM

    @property
    def estimated_input_tokens(self) -> int:
        return self.request["estimated_tokens"] if self.request else 0


def build_request(transcript: dict, material: dict, warnings: dict) -> dict:
    payload = tc.build_payload(transcript, material, warnings)
    payload_json = tc.serialize_payload(payload)
    summary = material["summary"]
    message = SPEC.user_template.format(
        interview_id=transcript["interview_id"], episode_count=summary["usable_episode_count"],
        unmarked_count=summary["unmarked_practice_count"], anchor_count=summary["temporal_anchor_count"],
        payload_json=payload_json)
    return {"user_message": message, "payload_chars": len(payload_json),
            "estimated_tokens": tc.payload_estimate(message)}


def prepare_stage5(ingestion: dict) -> Stage5Input:
    interview_dir = Path(ingestion["output_dir"])
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    transcript = json.loads((interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
    stage4 = stage4_state(interview_dir)
    source_hashes = {**stage3_hashes(interview_dir),
                     "accountability_episodes_sha256": _file_sha256(analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME),
                     "accountability_validation_sha256": _file_sha256(
                         analysis_dir / config.ACCOUNTABILITY_VALIDATION_FILENAME)}
    base = dict(interview_id=transcript["interview_id"], analysis_dir=analysis_dir, transcript=transcript,
                source_sha256=transcript["source"]["sha256"], stage4=stage4, source_hashes=source_hashes)
    if stage4["status"] not in (STAGE4_COMPLETE, STAGE4_PARTIAL):
        return Stage5Input(**base, speaker_warnings={}, material=None)
    episodes_doc = _read_json(analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME) or {}
    practices = (_read_json(analysis_dir / PRACTICE.output_filename) or {}).get("practices", [])
    signals = (_read_json(analysis_dir / INTERACTION.output_filename) or {}).get("signals", [])
    audit = _read_json(analysis_dir / AUDITOR.output_filename)
    warnings = speaker_audit.agent_warnings(audit, transcript) if audit and "items" in audit else {}
    material = tc.build_material(transcript, episodes_doc, practices, signals, warnings)
    request = build_request(transcript, material, warnings) if material["summary"]["item_count"] >= \
        tc.MIN_ITEMS_FOR_LLM else None
    return Stage5Input(**base, speaker_warnings=warnings, material=material, request=request)


def cache_key_fields(prepared: Stage5Input, settings: LLMSettings) -> dict:
    return {"source_sha256": prepared.source_sha256, "transcript_sha256": sha256_text(prepared.request["user_message"]),
            **SPEC.identity(), "model": settings.model, "request_params": settings.request_params()}


def plan_stage5(metadata: dict, interview_ids: list[str], settings: LLMSettings,
                cache: AnalysisCache | None = None, force: bool = False) -> dict:
    """Appels que provoquerait l'étape 5 (hors cache), sans rien appeler : 0 ou 1 par entretien."""
    cache = cache or AnalysisCache()
    wanted = set(interview_ids)
    plan = {"interviews": 0, "calls": 0, "cached": 0, "blocked": 0, "partial": 0, "no_material": 0, "per_interview": []}
    for info in eligible_files(metadata):
        if info["ingestion"]["interview_id"] not in wanted:
            continue
        prepared = prepare_stage5(info["ingestion"])
        plan["interviews"] += 1
        summary = (prepared.material or {}).get("summary", {})
        row = {"interview_id": prepared.interview_id, "stage4_status": prepared.stage4["status"],
               "stage4_restored": prepared.stage4.get("restored", False),
               "usable_episode_count": summary.get("usable_episode_count", 0),
               "unmarked_practice_count": summary.get("unmarked_practice_count", 0),
               "temporal_anchor_count": summary.get("temporal_anchor_count", 0),
               "call": 0, "cached": False, "estimated_input_tokens": prepared.estimated_input_tokens,
               "payload_chars": (prepared.request or {}).get("payload_chars", 0)}
        if prepared.blocked:
            plan["blocked"] += 1
        elif not prepared.needs_llm:
            plan["no_material"] += 1
        elif not force and settings.model and cache.contains(SPEC.name, compute_cache_key(
                cache_key_fields(prepared, settings))):
            plan["cached"] += 1
            row["cached"] = True
        else:
            plan["calls"] += 1
            row["call"] = 1
        plan["partial"] += prepared.stage4["status"] == STAGE4_PARTIAL
        plan["per_interview"].append(row)
    return plan


# --- Exécution sur un entretien ----------------------------------------------------------------------

async def _call(prepared: Stage5Input, client: LLMClient | None, cache: AnalysisCache, settings: LLMSettings,
                force: bool, label: str) -> dict:
    """L'appel unique, repris du cache s'il existe. Ne lève pas d'exception : renvoie un bilan."""
    key_fields = cache_key_fields(prepared, settings)
    record = {"cache_key": compute_cache_key(key_fields), "cache_hit": False, "status": STATUS_PENDING,
              "api_calls": 0, "billed_this_run": False, "usage": None, "duration_seconds": None,
              "response_model": None, "request_id": None, "stop_reason": None, "created_at": None, "error": None,
              "output": None}
    try:
        entry = None if force else cache.load(key_fields, SPEC.output_model)
        if entry is not None:
            call = entry.get("call", {})
            record.update(cache_hit=True, status=STATUS_CACHED, output=entry["output"], created_at=entry.get("created_at"),
                          usage=call.get("usage"), duration_seconds=call.get("duration_seconds"),
                          response_model=call.get("response_model"), request_id=call.get("request_id"),
                          stop_reason=call.get("stop_reason"))
            return record
        if client is None:
            raise LLMError("NOT_CONFIGURED")
        result = await client.complete_json(system_prompt=SPEC.system_prompt,
                                            user_content=prepared.request["user_message"],
                                            output_schema=SPEC.output_schema, response_model=SPEC.output_model,
                                            label=label)
        output = result.data.model_dump(mode="json")
        cache.store(key_fields, output, {
            "usage": result.usage, "attempts": result.attempts, "duration_seconds": result.duration_seconds,
            "model_requested": result.model_requested, "response_model": result.response_model,
            "request_id": result.request_id, "stop_reason": result.stop_reason})
        record.update(status=STATUS_SUCCESS, output=output, created_at=_now(), api_calls=result.attempts,
                      billed_this_run=result.attempts > 0, usage=result.usage, duration_seconds=result.duration_seconds,
                      response_model=result.response_model, request_id=result.request_id,
                      stop_reason=result.stop_reason)
    except LLMError as error:
        record.update(status=STATUS_FAILED, error=error.to_dict(), api_calls=error.attempts, usage=error.usage,
                      billed_this_run=bool(error.usage and error.usage.get("input_tokens")),
                      duration_seconds=error.duration_seconds, request_id=error.request_id)
    except Exception as exc:  # noqa: BLE001 — isolé à cet entretien, sans contenu d'entretien
        logger.error("Étape 5 %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        record.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())
    return record


def insufficient_output() -> dict:
    return {"configuration_type": "no_clear_pattern", "claims": [], "student_role_criteria": [],
            "trajectory_summary": INSUFFICIENT_SUMMARY, "confidence": "low", "needs_review": False, "mapper_notes": None}


def _counts(document: dict) -> dict:
    claims = document["trajectory_claims"]
    kept = [c for c in claims if c["usable_for_next_stages"]]
    criteria = [c for c in document["student_role_criteria"] if c["usable_for_next_stages"]]
    return {
        "claim_count": len(claims), "kept_claim_count": len(kept),
        "rejected_claim_count": len(claims) - len(kept),
        "requalified_claim_count": sum("model_claim_type" in c for c in claims),
        **{f"{key}_count": len(document[key]) for key in validator.LIST_KEYS.values()},
        "student_role_criteria_count": len(criteria),
        "claims_needing_review_count": sum(c["needs_review"] for c in kept),
    }


async def run_stage5_interview(prepared: Stage5Input, client: LLMClient | None, cache: AnalysisCache,
                               settings: LLMSettings, force: bool = False) -> dict:
    """Étape 5 sur UN entretien. Ne lève pas d'exception : renvoie le résumé (manifest)."""
    label = f"{prepared.interview_id}/{SPEC.name}"
    out_dir = prepared.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_dir / config.STUDENT_TRAJECTORY_FILENAME
    estimated = prepared.estimated_input_tokens
    over = estimated > tc.SINGLE_CALL_MAX_INPUT_TOKENS
    summary = (prepared.material or {}).get("summary", {})
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "interview_id": prepared.interview_id,
        **SPEC.identity(),
        "preprocessor_version": tc.PREPROCESSOR_VERSION,
        "payload_format": tc.PAYLOAD_FORMAT_VERSION,
        "validator_version": validator.VALIDATOR_VERSION,
        "source_hashes": prepared.source_hashes,
        "payload_sha256": sha256_text(prepared.request["user_message"]) if prepared.request else None,
        "model": settings.model,
        "request_params": settings.request_params(),
        "cache_key": None, "cache_hit": False, "llm_called": False,
        "status": STATUS_PENDING, "analysis_complete": False,
        "stage4_status": prepared.stage4["status"], "stage4_restored": prepared.stage4.get("restored", False),
        "run_at": _now(), "created_at": None,
        "api_calls": 0, "billed_this_run": False, "usage": None, "duration_seconds": None, "response_model": None,
        "request_id": None, "stop_reason": None,
        "estimated_input_tokens": estimated,
        "payload_chars": (prepared.request or {}).get("payload_chars", 0),
        "single_call_threshold_tokens": tc.SINGLE_CALL_MAX_INPUT_TOKENS,
        "over_single_call_threshold": over,
        "usable_episode_count": summary.get("usable_episode_count", 0),
        "excluded_episode_count": summary.get("excluded_episode_count", 0),
        "unmarked_practice_count": summary.get("unmarked_practice_count", 0),
        "temporal_anchor_count": summary.get("temporal_anchor_count", 0),
        "configuration_type": None,
        "validation_error_count": 0, "validation_warning_count": 0, "has_warnings": False,
        "output_file": None, "error": None,
    }
    validation_doc = {"interview_id": prepared.interview_id, "validator_version": validator.VALIDATOR_VERSION,
                      "validated_at": _now(), "stage4": prepared.stage4}

    if prepared.blocked:
        manifest.update(status=STATUS_BLOCKED, error={
            "code": "STAGE4_UNAVAILABLE", "status_code": None, "request_id": None,
            "message": "Étape 5 non exécutée : sorties de l'étape 4 absentes, en échec ou périmées. "
                       + " ".join(prepared.stage4["reasons"])})
        trajectory_path.unlink(missing_ok=True)  # jamais de trajectoire périmée à côté d'un blocage
        validation_doc.update(status=STATUS_BLOCKED, analysis_complete=False, available=False,
                              reason=manifest["error"]["message"])
        write_json_atomic(out_dir / config.STUDENT_TRAJECTORY_VALIDATION_FILENAME, validation_doc)
        write_json_atomic(out_dir / config.STUDENT_TRAJECTORY_MANIFEST_FILENAME, manifest)
        logger.info("Étape 5 %s : BLOCKED (étape 4 %s)", label, prepared.stage4["status"])
        return manifest

    complete = prepared.stage4["status"] == STAGE4_COMPLETE
    try:
        record = None
        if prepared.needs_llm:
            if over:
                logger.warning("Étape 5 %s : représentation au-delà du seuil d'un appel (%d tokens estimés)",
                               label, estimated)
            record = await _call(prepared, client, cache, settings, force, label)
            manifest.update(
                # appel API, ou réponse d'agent du workflow Claude Code (0 appel API, statut SUCCESS)
                cache_key=record["cache_key"], cache_hit=record["cache_hit"],
                llm_called=bool(record["api_calls"]) or record["status"] == STATUS_SUCCESS,
                api_calls=record["api_calls"] or 0, billed_this_run=record["billed_this_run"], usage=record["usage"],
                duration_seconds=record["duration_seconds"], response_model=record["response_model"],
                request_id=record["request_id"], stop_reason=record["stop_reason"],
                created_at=record["created_at"] or _now())
            if record["status"] not in (STATUS_SUCCESS, STATUS_CACHED):
                manifest.update(status=STATUS_FAILED, error=record["error"])
                raise _CallFailed()
            output = tc.expand_ids(record["output"], prepared.interview_id)
        else:
            logger.info("Étape 5 %s : matériau insuffisant, aucun appel API", label)
            manifest["created_at"] = _now()
            output = insufficient_output()

        validated = validator.validate_trajectory(output, prepared.material, prepared.transcript,
                                                  prepared.speaker_warnings)
        report, fields = validated["report"], validated["document"]
        if over:
            report["issues"].append({"object_id": prepared.interview_id, **validator.make_issue(
                "PAYLOAD_OVER_THRESHOLD", estimated_input_tokens=estimated)})
            report["warning_count"] += 1
        if not complete:
            report["issues"].append({"object_id": prepared.interview_id, **validator.make_issue(
                "STAGE4_INCOMPLETE", stage4_status=prepared.stage4["status"])})
            report["warning_count"] += 1
        fields["needs_review"] = fields["needs_review"] or validator.has_problems(report)
        cache_hit = bool(record and record["cache_hit"])
        status = (STATUS_PARTIAL if not complete else STATUS_CACHED if cache_hit
                  else STATUS_SUCCESS_WITH_WARNINGS if validator.has_problems(report) else STATUS_SUCCESS)
        counts = _counts(fields)
        document = {
            "agent": SPEC.name, "agent_version": SPEC.version, "schema_version": SPEC.schema_version,
            "prompt_sha256": SPEC.prompt_sha256, "preprocessor_version": tc.PREPROCESSOR_VERSION,
            "payload_format": tc.PAYLOAD_FORMAT_VERSION, "validator_version": validator.VALIDATOR_VERSION,
            "interview_id": prepared.interview_id,
            "model": settings.model if prepared.needs_llm else None,
            "status": status, "analysis_complete": complete,
            "stage4_status": prepared.stage4["status"], "stage4_restored": prepared.stage4.get("restored", False),
            "cache_hit": cache_hit, "llm_called": manifest["llm_called"], "generated_at": manifest["created_at"],
            "source_hashes": prepared.source_hashes,
            **counts,
            "validation_error_count": report["error_count"], "validation_warning_count": report["warning_count"],
            **fields,
            "material": tc.public(prepared.material),
        }
        write_json_atomic(trajectory_path, document)
        validation_doc.update(status=status, analysis_complete=complete, available=True, **report)
        manifest.update(status=status, analysis_complete=complete, configuration_type=fields["configuration_type"],
                        **counts, validation_error_count=report["error_count"],
                        validation_warning_count=report["warning_count"],
                        has_warnings=not complete or validator.has_problems(report),
                        output_file=config.STUDENT_TRAJECTORY_FILENAME)
    except _CallFailed:
        pass
    except Exception as exc:  # noqa: BLE001 — isolé à cet entretien, sans contenu d'entretien
        logger.error("Étape 5 %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        manifest.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())

    if manifest["status"] == STATUS_FAILED:
        trajectory_path.unlink(missing_ok=True)
        validation_doc.update(status=STATUS_FAILED, analysis_complete=False, available=False,
                              reason=manifest["error"]["message"])
    write_json_atomic(out_dir / config.STUDENT_TRAJECTORY_VALIDATION_FILENAME, validation_doc)
    write_json_atomic(out_dir / config.STUDENT_TRAJECTORY_MANIFEST_FILENAME, manifest)
    logger.info("Étape 5 %s : %s (%d élément(s), %s)", label, manifest["status"], prepared.item_count,
                manifest["configuration_type"])
    return manifest


class _CallFailed(Exception):
    """L'appel a échoué : le manifest porte déjà l'erreur."""


# --- Un run ----------------------------------------------------------------------------------------

SUMMARY_KEYS = ("status", "stage4_status", "stage4_restored", "analysis_complete", "configuration_type", "claim_count",
                "kept_claim_count", "rejected_claim_count", "requalified_claim_count",
                *(f"{key}_count" for key in validator.LIST_KEYS.values()), "student_role_criteria_count",
                "claims_needing_review_count", "usable_episode_count", "excluded_episode_count",
                "unmarked_practice_count", "temporal_anchor_count", "validation_error_count",
                "validation_warning_count", "has_warnings", "cache_hit", "llm_called", "api_calls", "billed_this_run",
                "usage", "duration_seconds", "estimated_input_tokens", "payload_chars", "over_single_call_threshold",
                "error")


def _summary(prepared: Stage5Input, manifest: dict) -> dict:
    return {"interview_id": prepared.interview_id, "analysis_dir": str(prepared.analysis_dir), "run_at": _now(),
            **{k: manifest.get(k) for k in SUMMARY_KEYS}}


async def run_stage5_interviews(prepared_list: list[Stage5Input], settings: LLMSettings,
                                transport: Transport | None = None, cache: AnalysisCache | None = None,
                                force: bool = False, client=None) -> list[dict]:
    """`client` : client déjà construit, de même interface que LLMClient ; le workflow Claude Code
    (core/claude_code_workflow.py) y passe un client SANS appel réseau. Sinon, un LLMClient (API) est construit
    seulement si un entretien a assez de matériau."""
    cache = cache or AnalysisCache()
    if client is None and any(p.needs_llm for p in prepared_list):
        client = LLMClient(settings, transport=transport)
    try:
        manifests = await asyncio.gather(*(run_stage5_interview(p, client, cache, settings, force)
                                           for p in prepared_list))
    finally:
        if client is not None:
            await client.aclose()
    return [_summary(p, m) for p, m in zip(prepared_list, manifests)]


def summarize_usage(summaries: list[dict]) -> dict:
    total = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_results": 0, "blocked": 0, "failed": 0,
             "partial": 0, "no_material": 0}
    for s in summaries:
        total["api_calls"] += s.get("api_calls") or 0
        total["cached_results"] += 1 if s.get("cache_hit") else 0
        total["blocked"] += s.get("status") == STATUS_BLOCKED
        total["failed"] += s.get("status") == STATUS_FAILED
        total["partial"] += s.get("status") == STATUS_PARTIAL
        total["no_material"] += s.get("status") not in (STATUS_BLOCKED, STATUS_FAILED) and not s.get("llm_called") \
            and not s.get("cache_hit")
        if s.get("billed_this_run") and s.get("usage"):
            total["input_tokens"] += s["usage"].get("input_tokens") or 0
            total["output_tokens"] += s["usage"].get("output_tokens") or 0
    return total


def _step_state(metadata: dict) -> str:
    done = [f["trajectory"]["status"] for f in eligible_files(metadata) if "trajectory" in f]
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


def analyze_run_stage5(metadata: dict, interview_ids: list[str] | None = None, *, settings: LLMSettings | None = None,
                       transport: Transport | None = None, cache: AnalysisCache | None = None,
                       force: bool = False, client=None) -> dict:
    """Étape 5 sur les entretiens choisis d'un run (étape 4 déjà disponible). Met à jour metadata.json.

    Sans `client`, lève LLMError (NOT_CONFIGURED) si la clé ou le modèle manque. Avec `client` (workflow
    Claude Code, sans API), aucune clé n'est demandée.
    """
    settings = settings or LLMSettings.from_env()
    if client is None and not settings.enabled:
        raise LLMError("NOT_CONFIGURED", ", ".join(settings.missing))
    files = eligible_files(metadata)
    if interview_ids is not None:
        wanted = set(interview_ids)
        files = [f for f in files if f["ingestion"]["interview_id"] in wanted]
    prepared = [prepare_stage5(f["ingestion"]) for f in files]
    summaries = _run_coroutine(run_stage5_interviews(prepared, settings, transport, cache, force, client))
    by_id = {s["interview_id"]: s for s in summaries}
    for info in files:
        info["trajectory"] = by_id[info["ingestion"]["interview_id"]]
    metadata["pipeline"][config.TRAJECTORY_STEP] = _step_state(metadata)
    metadata["last_trajectory"] = {"analyzed_at": _now(), "model": settings.model,
                                   "interview_ids": [p.interview_id for p in prepared], "forced": force,
                                   "usage": summarize_usage(summaries)}
    save_metadata(metadata, Path(metadata["output_dir"]))
    return metadata


def trajectory_current(summary: dict) -> bool:
    """Le résultat affiché décrit-il encore l'étape 4 actuelle ? (l'étape 4 relancée ou restaurée le périme)."""
    document = _read_json(Path(summary["analysis_dir"]) / config.STUDENT_TRAJECTORY_FILENAME)
    if document is None:
        return summary.get("status") in (STATUS_BLOCKED, STATUS_FAILED)
    recorded = document.get("source_hashes") or {}
    analysis_dir = Path(summary["analysis_dir"])
    return recorded.get("accountability_episodes_sha256") == _file_sha256(
        analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME) and recorded.get("practice_extractor_sha256") == \
        _file_sha256(analysis_dir / PRACTICE.output_filename)
