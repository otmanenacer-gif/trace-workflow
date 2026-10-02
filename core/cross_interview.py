"""Étape 6 — comparaison inter-entretiens (orchestration).

    triplets de l'étape 5 (importés ou repris du run)  →  contrôle du corpus (core/cross_interview_corpus.py)
                                                       →  préparation déterministe (core/cross_interview_material.py)
                                                       →  Cross-Interview Comparator (LLM, 1 appel par corpus)
                                                       →  validateur déterministe (core/cross_interview_validator.py)
                                                       →  analysis/cross_interview_comparison.json
                                                          analysis/cross_interview_validation.json
                                                          analysis/cross_interview_manifest.json

L'étape 6 ne lit QUE les triplets de l'étape 5 ; elle n'appelle et ne modifie jamais les étapes 3, 4 ou 5, et
n'écrit rien dans les dossiers des entretiens. Ses sorties vont dans
data/outputs/cross_interview/<corpus_id>/ (copie octet pour octet des triplets exploitables dans stage5/, bilan
de l'import, puis analysis/).

Seuil : moins de 2 entretiens exploitables → BLOCKED, aucun appel ; 2 → comparaison EXPLORATOIRE ; 3 et plus →
comparaison normale.

Coût : 1 appel pour tout le corpus (0 si bloqué ou repris du cache). Pas de découpage MAP → REDUCE : la
représentation normalisée de 17 entretiens synthétiques fait ≈ 9 000 tokens estimés (≈ 22 000 si les 17 avaient
la taille de l'entretien long « OTMANE-like »), sous le seuil d'un appel unique (24 000, celui de l'étape 5). Au-delà
du seuil, l'appel unique a lieu quand même, signalé (`PAYLOAD_OVER_THRESHOLD`) ; max_tokens n'est jamais augmenté.

Cache propre à l'étape 6 (data/cache/analysis/cross_interview_comparator/) : la clé couvre la requête exacte
(donc chaque entretien du corpus et son contenu), l'agent, sa version, son prompt, son schéma, le modèle et les
paramètres. Même corpus + même version → 0 appel ; ajouter, retirer ou modifier un entretien ne change que la
requête de l'étape 6 : les étapes 3 à 5 restent intactes.
"""

from __future__ import annotations

import json
import logging
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agents import cross_interview_comparator as comparator
from agents.base import canonical_json, sha256_text
from core import config
from core import cross_interview_corpus as corpus
from core import cross_interview_material as cm
from core import cross_interview_validator as validator
from core.analysis import (STATUS_CACHED, STATUS_FAILED, STATUS_PENDING, STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS,
                           _run_coroutine)
from core.analysis_cache import AnalysisCache, compute_cache_key, write_json_atomic
from core.llm_client import LLMClient, LLMError, LLMSettings, Transport

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"
SPEC = comparator.SPEC
STATUS_BLOCKED = "BLOCKED"
DONE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)
# Même seuil d'appel unique qu'à l'étape 5 ; mesuré sur 17 entretiens synthétiques : ≈ 9 000 tokens estimés.
SINGLE_CALL_MAX_INPUT_TOKENS = 24_000
ANALYSIS_SUBDIR = config.ANALYSIS_SUBDIR

BLOCKED_MESSAGE = ("Étape 6 bloquée : moins de deux entretiens exploitables ({n_usable} sur {n_total} importé(s)). "
                   "Aucun appel API.")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Stage6Input:
    checked: dict
    corpus_id: str
    material: dict | None
    request: dict | None

    @property
    def blocked(self) -> bool:
        return self.checked["mode"] == corpus.MODE_BLOCKED

    @property
    def estimated_input_tokens(self) -> int:
        return self.request["estimated_tokens"] if self.request else 0

    @property
    def excluded_ids(self) -> set[str]:
        return {r["interview_id"] for r in self.checked["rows"] if r["status"] == corpus.STATUS_EXCLUDED}

    @property
    def input_hashes(self) -> dict:
        return {r["interview_id"]: {kind: f["sha256"] for kind, f in r["files"].items()}
                for r in self.checked["rows"] if r["status"] == corpus.STATUS_USABLE}


def build_request(material: dict) -> dict:
    payload_json = cm.serialize_payload(cm.build_payload(material))
    summary = material["summary"]
    message = SPEC.user_template.format(
        n_usable=summary["corpus_n_usable"], n_total=summary["corpus_n_total"],
        mode_label=comparator.MODE_LABELS.get(summary["mode"], summary["mode"]),
        claim_count=summary["claim_count"], criterion_count=summary["criterion_count"],
        review_count=summary["review_claim_count"] + summary["review_criterion_count"], payload_json=payload_json)
    return {"user_message": message, "payload_chars": len(payload_json),
            "estimated_tokens": cm.payload_estimate(message)}


def prepare_stage6(uploads: list[tuple[str, bytes]]) -> Stage6Input:
    """Contrôle du corpus et préparation déterministe (gratuites). Aucun appel."""
    checked = corpus.check_corpus(uploads)
    corpus_id = corpus.corpus_id(checked)
    if checked["mode"] == corpus.MODE_BLOCKED:
        return Stage6Input(checked, corpus_id, None, None)
    usable = {iid: files[corpus.KIND_TRAJECTORY][2] for iid, files in checked["usable"].items()}
    material = cm.build_material(usable, checked["n_total"], checked["mode"])
    return Stage6Input(checked, corpus_id, material, build_request(material))


def cache_key_fields(prepared: Stage6Input, settings: LLMSettings) -> dict:
    sources = {iid: (files[corpus.KIND_TRAJECTORY][2].get("source_hashes") or {})
               for iid, files in sorted(prepared.checked["usable"].items())}
    return {"source_sha256": sha256_text(canonical_json(sources)),
            "transcript_sha256": sha256_text(prepared.request["user_message"]),
            **SPEC.identity(), "model": settings.model, "request_params": settings.request_params()}


def plan_stage6(prepared: Stage6Input, settings: LLMSettings, cache: AnalysisCache | None = None,
                force: bool = False) -> dict:
    """Appels que provoquerait l'étape 6 (hors cache), sans rien appeler : 0 ou 1."""
    cache = cache or AnalysisCache()
    plan = {"corpus_id": prepared.corpus_id, "n_total": prepared.checked["n_total"],
            "n_usable": prepared.checked["n_usable"], "mode": prepared.checked["mode"], "calls": 0, "cached": False,
            "blocked": prepared.blocked, "estimated_input_tokens": prepared.estimated_input_tokens,
            "payload_chars": (prepared.request or {}).get("payload_chars", 0),
            "raw_stage5_chars": cm.raw_size(prepared.checked["usable"]),
            "over_single_call_threshold": prepared.estimated_input_tokens > SINGLE_CALL_MAX_INPUT_TOKENS,
            "map_reduce": False}
    if prepared.blocked:
        return plan
    key = compute_cache_key(cache_key_fields(prepared, settings)) if settings.model else None
    if not force and key and cache.contains(SPEC.name, key):
        plan["cached"] = True
    else:
        plan["calls"] = 1
    return plan


async def _call(prepared: Stage6Input, client: LLMClient | None, cache: AnalysisCache, settings: LLMSettings,
                force: bool) -> dict:
    """L'appel unique, repris du cache s'il existe. Ne lève pas d'exception : renvoie un bilan."""
    key_fields = cache_key_fields(prepared, settings)
    record = {"cache_key": compute_cache_key(key_fields), "cache_hit": False, "status": STATUS_PENDING,
              "api_calls": 0, "billed_this_run": False, "usage": None, "duration_seconds": None,
              "response_model": None, "request_id": None, "stop_reason": None, "created_at": None, "error": None,
              "output": None}
    label = f"{prepared.corpus_id}/{SPEC.name}"
    try:
        entry = None if force else cache.load(key_fields, SPEC.output_model)
        if entry is not None:
            call = entry.get("call", {})
            record.update(cache_hit=True, status=STATUS_CACHED, output=entry["output"],
                          created_at=entry.get("created_at"), usage=call.get("usage"),
                          duration_seconds=call.get("duration_seconds"),
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
    except Exception as exc:  # noqa: BLE001 — sans contenu d'entretien
        logger.error("Étape 6 %s : erreur inattendue %s\n%s", label, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        record.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())
    return record


def _counts(fields: dict) -> dict:
    claims = fields["cross_case_claims"]
    kept = [c for c in claims if c["usable_for_next_stages"]]
    return {"cross_claim_count": len(claims), "kept_cross_claim_count": len(kept),
            "rejected_cross_claim_count": len(claims) - len(kept),
            "requalified_cross_claim_count": sum("model_claim_type" in c for c in claims),
            "cross_claims_needing_review_count": sum(c["needs_review"] for c in kept),
            "negative_case_count": len(fields["negative_cases"]),
            "student_role_criterion_pattern_count": len(fields["student_role_criterion_patterns"]),
            **{f"{key}_count": len(fields[key]) for key in validator.LIST_KEYS.values()}}


def corpus_dir_for(prepared: Stage6Input, base_dir: Path | None = None) -> Path:
    return Path(base_dir or config.CROSS_INTERVIEW_DIR) / prepared.corpus_id


async def run_stage6_corpus(prepared: Stage6Input, client: LLMClient | None, cache: AnalysisCache,
                            settings: LLMSettings, base_dir: Path | None = None, force: bool = False) -> dict:
    """Étape 6 sur un corpus. Ne lève pas d'exception : renvoie le manifest."""
    corpus_dir = corpus_dir_for(prepared, base_dir)
    out_dir = corpus_dir / ANALYSIS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stage5_dir = corpus_dir / "stage5"
    if stage5_dir.exists():
        shutil.rmtree(stage5_dir)  # recopie complète : jamais un triplet périmé à côté du corpus
    corpus.install(prepared.checked, corpus_dir)
    write_json_atomic(corpus_dir / config.CROSS_INTERVIEW_CORPUS_FILENAME,
                      {"corpus_id": prepared.corpus_id, "checked_at": _now(), **corpus.public(prepared.checked)})
    comparison_path = out_dir / config.CROSS_INTERVIEW_COMPARISON_FILENAME
    checked = prepared.checked
    estimated = prepared.estimated_input_tokens
    over = estimated > SINGLE_CALL_MAX_INPUT_TOKENS
    excluded = [{"interview_id": r["interview_id"], "reasons": r["reasons"]} for r in checked["rows"]
                if r["status"] == corpus.STATUS_EXCLUDED]
    manifest = {
        "manifest_version": MANIFEST_VERSION, "corpus_id": prepared.corpus_id, **SPEC.identity(),
        "preprocessor_version": cm.PREPROCESSOR_VERSION, "payload_format": cm.PAYLOAD_FORMAT_VERSION,
        "validator_version": validator.VALIDATOR_VERSION, "corpus_check_version": corpus.CORPUS_CHECK_VERSION,
        "corpus_n_total": checked["n_total"], "corpus_n_usable": checked["n_usable"], "mode": checked["mode"],
        "exploratory": checked["mode"] == corpus.MODE_EXPLORATORY,
        "interview_ids": sorted(checked["usable"]), "excluded_interviews": excluded,
        "unrecognized_files": checked["unrecognized"], "input_hashes": prepared.input_hashes,
        "payload_sha256": sha256_text(prepared.request["user_message"]) if prepared.request else None,
        "model": settings.model, "request_params": settings.request_params(),
        "cache_key": None, "cache_hit": False, "llm_called": False, "status": STATUS_PENDING,
        "analysis_complete": False, "run_at": _now(), "created_at": None,
        "api_calls": 0, "billed_this_run": False, "usage": None, "duration_seconds": None, "response_model": None,
        "request_id": None, "stop_reason": None,
        "planned_calls": 0 if prepared.blocked else 1, "map_reduce_used": False, "upstream_stage_calls": 0,
        "estimated_input_tokens": estimated, "payload_chars": (prepared.request or {}).get("payload_chars", 0),
        "raw_stage5_chars": cm.raw_size(checked["usable"]),
        "single_call_threshold_tokens": SINGLE_CALL_MAX_INPUT_TOKENS, "over_single_call_threshold": over,
        "validation_error_count": 0, "validation_warning_count": 0, "has_warnings": False,
        "output_file": None, "error": None,
    }
    validation_doc = {"corpus_id": prepared.corpus_id, "validator_version": validator.VALIDATOR_VERSION,
                      "validated_at": _now(), "corpus_n_total": checked["n_total"],
                      "corpus_n_usable": checked["n_usable"], "mode": checked["mode"], "excluded_interviews": excluded}

    if prepared.blocked:
        message = BLOCKED_MESSAGE.format(n_usable=checked["n_usable"], n_total=checked["n_total"])
        manifest.update(status=STATUS_BLOCKED, created_at=_now(), error={
            "code": "CORPUS_TOO_SMALL", "status_code": None, "request_id": None, "message": message})
        comparison_path.unlink(missing_ok=True)
        validation_doc.update(status=STATUS_BLOCKED, analysis_complete=False, available=False, reason=message)
        write_json_atomic(out_dir / config.CROSS_INTERVIEW_VALIDATION_FILENAME, validation_doc)
        write_json_atomic(out_dir / config.CROSS_INTERVIEW_MANIFEST_FILENAME, manifest)
        return {**manifest, "corpus_dir": str(corpus_dir)}

    try:
        if over:
            logger.warning("Étape 6 %s : représentation au-delà du seuil d'un appel (%d tokens estimés)",
                           prepared.corpus_id, estimated)
        record = await _call(prepared, client, cache, settings, force)
        # appel API, ou réponse d'agent du workflow Claude Code (0 appel API, statut SUCCESS)
        manifest.update(cache_key=record["cache_key"], cache_hit=record["cache_hit"],
                        llm_called=bool(record["api_calls"]) or record["status"] == STATUS_SUCCESS,
                        api_calls=record["api_calls"] or 0,
                        billed_this_run=record["billed_this_run"], usage=record["usage"],
                        duration_seconds=record["duration_seconds"], response_model=record["response_model"],
                        request_id=record["request_id"], stop_reason=record["stop_reason"],
                        created_at=record["created_at"] or _now())
        if record["status"] not in (STATUS_SUCCESS, STATUS_CACHED):
            manifest.update(status=STATUS_FAILED, error=record["error"])
        else:
            output = cm.expand_ids(record["output"], prepared.material)
            validated = validator.validate_comparison(output, prepared.material, prepared.excluded_ids)
            report, fields = validated["report"], validated["document"]
            if over:
                report["issues"].append({"object_id": "corpus", **validator.make_issue(
                    "PAYLOAD_OVER_THRESHOLD", estimated_input_tokens=estimated)})
                report["warning_count"] += 1
            fields["needs_review"] = fields["needs_review"] or validator.has_problems(report)
            status = (STATUS_CACHED if record["cache_hit"] else
                      STATUS_SUCCESS_WITH_WARNINGS if validator.has_problems(report) else STATUS_SUCCESS)
            counts = _counts(fields)
            document = {
                "agent": SPEC.name, "agent_version": SPEC.version, "schema_version": SPEC.schema_version,
                "prompt_sha256": SPEC.prompt_sha256, "preprocessor_version": cm.PREPROCESSOR_VERSION,
                "payload_format": cm.PAYLOAD_FORMAT_VERSION, "validator_version": validator.VALIDATOR_VERSION,
                "corpus_id": prepared.corpus_id, "model": settings.model, "status": status, "analysis_complete": True,
                "cache_hit": record["cache_hit"], "llm_called": manifest["llm_called"],
                "generated_at": manifest["created_at"],
                "corpus_n_total": checked["n_total"], "corpus_n_usable": checked["n_usable"],
                "mode": checked["mode"], "exploratory": checked["mode"] == corpus.MODE_EXPLORATORY,
                "interview_ids": sorted(checked["usable"]), "excluded_interviews": excluded,
                "input_hashes": prepared.input_hashes, **counts,
                "validation_error_count": report["error_count"], "validation_warning_count": report["warning_count"],
                **fields,
                "material": cm.public(prepared.material),
            }
            write_json_atomic(comparison_path, document)
            validation_doc.update(status=status, analysis_complete=True, available=True, **report)
            manifest.update(status=status, analysis_complete=True, **counts,
                            validation_error_count=report["error_count"],
                            validation_warning_count=report["warning_count"],
                            has_warnings=validator.has_problems(report),
                            output_file=config.CROSS_INTERVIEW_COMPARISON_FILENAME)
    except Exception as exc:  # noqa: BLE001 — sans contenu d'entretien
        logger.error("Étape 6 %s : erreur inattendue %s\n%s", prepared.corpus_id, type(exc).__name__,
                     "".join(traceback.format_tb(exc.__traceback__)))
        manifest.update(status=STATUS_FAILED, error=LLMError("UNEXPECTED_ERROR", type(exc).__name__).to_dict())

    if manifest["status"] == STATUS_FAILED:
        comparison_path.unlink(missing_ok=True)  # jamais de synthèse COMPLETE si l'appel manque
        validation_doc.update(status=STATUS_FAILED, analysis_complete=False, available=False,
                              reason=(manifest["error"] or {}).get("message"))
    write_json_atomic(out_dir / config.CROSS_INTERVIEW_VALIDATION_FILENAME, validation_doc)
    write_json_atomic(out_dir / config.CROSS_INTERVIEW_MANIFEST_FILENAME, manifest)
    logger.info("Étape 6 %s : %s (%d/%d entretiens exploitables)", prepared.corpus_id, manifest["status"],
                checked["n_usable"], checked["n_total"])
    return {**manifest, "corpus_dir": str(corpus_dir)}


def run_stage6(uploads: list[tuple[str, bytes]], *, settings: LLMSettings | None = None,
               transport: Transport | None = None, cache: AnalysisCache | None = None, force: bool = False,
               base_dir: Path | None = None, client=None) -> dict:
    """Étape 6 sur les fichiers importés. Renvoie le manifest (+ corpus_dir).

    Sans `client`, lève LLMError (NOT_CONFIGURED) si la clé ou le modèle manque alors qu'une comparaison est
    possible. `client` : client déjà construit, de même interface que LLMClient ; le workflow Claude Code
    (core/claude_code_workflow.py) y passe un client SANS appel réseau, et aucune clé n'est demandée.
    """
    settings = settings or LLMSettings.from_env()
    prepared = prepare_stage6(uploads)
    if client is None and not prepared.blocked and not settings.enabled:
        raise LLMError("NOT_CONFIGURED", ", ".join(settings.missing))
    cache = cache or AnalysisCache()

    async def run(client=client):
        if client is None and not prepared.blocked:
            client = LLMClient(settings, transport=transport)
        try:
            return await run_stage6_corpus(prepared, client, cache, settings, base_dir, force)
        finally:
            if client is not None:
                await client.aclose()

    return _run_coroutine(run())


def read_outputs(corpus_dir: Path) -> dict:
    """Les trois sorties de l'étape 6 (texte JSON brut, ou None) et le bilan de l'import."""
    corpus_dir = Path(corpus_dir)
    out = corpus_dir / ANALYSIS_SUBDIR

    def read(path: Path):
        return path.read_text(encoding="utf-8") if path.is_file() else None

    return {"comparison": read(out / config.CROSS_INTERVIEW_COMPARISON_FILENAME),
            "validation": read(out / config.CROSS_INTERVIEW_VALIDATION_FILENAME),
            "manifest": read(out / config.CROSS_INTERVIEW_MANIFEST_FILENAME),
            "corpus": json.loads(read(corpus_dir / config.CROSS_INTERVIEW_CORPUS_FILENAME) or "null")}
