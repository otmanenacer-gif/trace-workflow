"""Étapes 3 à 6 exécutées LOCALEMENT, de bout en bout, par TRACE lui-même.

    Streamlit (ou la CLI scripts/trace_local.py)
      → local_pipeline.run_stage("3" | "4" | "5") / run_until / run_stage6
      → orchestrateurs habituels (core/analysis.py, accountability.py, trajectory.py, cross_interview.py)
      → LocalAgentRunner → Ollama local (core/local_agent_runner.py)
      → validateurs habituels → sorties habituelles (analysis/, cross_interview/<corpus_id>/analysis/)

Aucune API externe, aucune clé, aucune intervention extérieure : un clic (ou une commande) exécute toute l'étape.
Avant toute exécution, le runtime est vérifié (Ollama joignable sur la boucle locale, modèle installé) ; sinon une
erreur claire est levée et rien n'est écrit — jamais de repli vers un autre fournisseur.

Garde contre les réexécutions : une étape n'est rejouée QUE pour les entretiens dont elle n'est pas complète et à jour
sur le disque (étape 3 COMPLETE ; étape 4 COMPLETE et calculée sur l'étape 3 actuelle ; étape 5 terminée, complète et
calculée sur les étapes 3 et 4 actuelles). Rejouer une étape complète réécrirait ses fichiers et rendrait l'étape
suivante « périmée » : seule une demande explicite (`force=True`) le fait. `run_until` enchaîne 3 → 4 → 5 avec cette
garde et s'arrête à la première étape non terminée. Étape 6 : un corpus déjà analysé à l'identique (mêmes fichiers de
l'étape 5, même paquet, mêmes prompt, schéma et versions) n'est pas rejoué.

Le cache TRACE (core/analysis_cache.py) reprend les réponses validées d'une requête identique (même modèle) : après
un échec partiel, seuls les appels manquants sont refaits. Chaque appel est journalisé dans
<run>/local_runs/stage<N>/ (ou <corpus>/local_runs/stage6/).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from core import accountability, analysis, config, cross_interview
from core import cross_interview_corpus as corpus
from core import cross_interview_material as cm
from core import cross_interview_validator as cross_validator
from core import trajectory
from core.agent_checks import AGENT_SPECS, MethodChecker
from core.analysis_cache import AnalysisCache, write_json_atomic
from core.llm_client import LLMError, LLMSettings
from core.local_agent_runner import EXECUTION, EXTERNAL_DATA, RUNTIME_NAME, LocalAgentRunner, check_runtime
from core.run_manager import save_metadata
from agents.base import sha256_text

PIPELINE_VERSION = "1.0"
STAGES = ("3", "4", "5")   # étapes par entretien (dans un run) ; l'étape 6 porte sur un corpus
CORPUS_STAGE = "6"
LOCAL_RUNS_SUBDIR = "local_runs"
STATUS_FILENAME = "status.json"

# État d'une étape après une exécution
STAGE_COMPLETE = "COMPLETE"
STAGE_FAILED = "FAILED"
STAGE_BLOCKED = "BLOCKED"   # étape précédente en échec, absente ou périmée (règles de accountability / trajectory)
STATUS_LABELS = {
    STAGE_COMPLETE: "terminée",
    STAGE_FAILED: "échec ou analyse incomplète (voir les statuts des agents)",
    STAGE_BLOCKED: "bloquée (étape précédente en échec, absente ou périmée)",
}
PIPELINE_STEPS_BY_STAGE = {"3": (config.PRACTICE_STEP, config.INTERACTION_STEP), "4": (config.ACCOUNTABILITY_STEP,),
                           "5": (config.TRAJECTORY_STEP,)}
SUMMARY_KEY_BY_STAGE = {"3": "analysis", "4": "accountability", "5": "trajectory"}
DONE_BY_STAGE = {"3": analysis.DONE_STATUSES, "4": accountability.DONE_STATUSES, "5": trajectory.DONE_STATUSES}
SPEC_BY_STAGE = {"4": accountability.SPEC, "5": trajectory.SPEC}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


# --- Runtime -----------------------------------------------------------------------------------------

def runtime_settings(env: dict | None = None) -> LLMSettings:
    """Paramètres du runtime local (TRACE_LOCAL_MODEL, TRACE_OLLAMA_URL…), lus dans l'environnement."""
    return LLMSettings.from_env(env)


def make_runner(settings: LLMSettings, *, journal_dir: Path | None = None, checker=None) -> LocalAgentRunner:
    """Le runner des agents (Ollama local). Point d'injection unique : les tests le remplacent par un runner dont
    le transport est simulé (tests/fake_llm.FakeLocalAgentRunner)."""
    return LocalAgentRunner(settings, checker=checker, journal_dir=journal_dir, specs=AGENT_SPECS)


def runtime_status(settings: LLMSettings | None = None) -> dict:
    """État du runtime pour l'interface : exécution locale, Ollama, modèle, données externes : aucune."""
    settings = settings or runtime_settings()
    return check_runtime(settings, make_runner(settings).transport)


def runtime_info(settings: LLMSettings) -> dict:
    return {"execution": EXECUTION, "runtime": RUNTIME_NAME, "model": settings.model, "url": settings.ollama_url,
            "external_data": EXTERNAL_DATA}


# --- Garde : une étape complète et à jour n'est pas rejouée ------------------------------------------

STAGE5_COMPLETE, STAGE5_NOT_RUN, STAGE5_INCOMPLETE, STAGE5_STALE = "COMPLETE", "NOT_RUN", "INCOMPLETE", "STALE"


def _stage5_source_hashes(interview_dir: Path) -> dict:
    """Empreintes des sources de l'étape 5, calculées comme core/trajectory.prepare_stage5."""
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    return {**trajectory.stage3_hashes(interview_dir),
            "accountability_episodes_sha256": trajectory._file_sha256(
                analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME),
            "accountability_validation_sha256": trajectory._file_sha256(
                analysis_dir / config.ACCOUNTABILITY_VALIDATION_FILENAME)}


def stage5_state(interview_dir: Path) -> dict:
    """État de l'étape 5 sur le disque : COMPLETE (terminée, complète, à jour), NOT_RUN, INCOMPLETE ou STALE."""
    interview_dir = Path(interview_dir)
    document = trajectory._read_json(interview_dir / config.ANALYSIS_SUBDIR / config.STUDENT_TRAJECTORY_FILENAME)
    if document is None:
        return {"status": STAGE5_NOT_RUN, "reasons": ["Étape 5 : aucune sortie."]}
    if document.get("status") not in trajectory.DONE_STATUSES or document.get("analysis_complete") is not True:
        return {"status": STAGE5_INCOMPLETE, "reasons": [f"Étape 5 : {document.get('status')}, analysis_complete = "
                                                         f"{document.get('analysis_complete')}."]}
    if (document.get("source_hashes") or {}) != _stage5_source_hashes(interview_dir):
        return {"status": STAGE5_STALE, "reasons": ["Étape 5 périmée : les étapes 3 ou 4 ont changé depuis."]}
    return {"status": STAGE5_COMPLETE, "reasons": []}


def stage_state(stage: str, info: dict) -> dict:
    """État d'une étape pour un entretien, lu sur le disque (règles existantes des étapes 4 et 5)."""
    interview_dir = Path(info["ingestion"]["output_dir"])
    if stage == "3":
        return accountability.stage3_state(interview_dir / config.ANALYSIS_SUBDIR)
    if stage == "4":
        return trajectory.stage4_state(interview_dir)
    return stage5_state(interview_dir)


def stage_complete(stage: str, info: dict) -> bool:
    """Étape terminée, complète et calculée sur les sorties actuelles de l'étape précédente : non rejouée."""
    return stage_state(stage, info)["status"] == "COMPLETE"


def _selected_files(metadata: dict, interview_ids: list[str] | None) -> list[dict]:
    files = analysis.eligible_files(metadata)
    if interview_ids is not None:
        wanted = set(interview_ids)
        unknown = wanted - {f["ingestion"]["interview_id"] for f in files}
        if unknown:
            raise ValueError("Entretien(s) inconnu(s) ou non analysable(s) dans ce run : "
                             + ", ".join(sorted(unknown)))
        files = [f for f in files if f["ingestion"]["interview_id"] in wanted]
    return files


def stage_dir(metadata: dict, stage: str) -> Path:
    return Path(metadata["output_dir"]) / LOCAL_RUNS_SUBDIR / f"stage{stage}"


# --- Une étape d'un run ----------------------------------------------------------------------------------

def _execute(stage: str, metadata: dict, ids: list[str], runner, settings: LLMSettings, force: bool) -> dict:
    cache = AnalysisCache()
    if stage == "3":
        return analysis.analyze_run(metadata, ids, settings=settings, cache=cache, force=force, client=runner)
    if stage == "4":
        return accountability.analyze_run_stage4(metadata, ids, settings=settings, cache=cache, force=force,
                                                 client=runner)
    return trajectory.analyze_run_stage5(metadata, ids, settings=settings, cache=cache, force=force, client=runner)


def run_stage(stage: str, metadata: dict, interview_ids: list[str] | None = None, *,
              settings: LLMSettings | None = None, force: bool = False, runner=None) -> dict:
    """Exécute UNE étape (« 3 », « 4 » ou « 5 ») avec le modèle local. Renvoie {"metadata", "status"}.

    Seulement pour les entretiens dont l'étape n'est pas déjà complète et à jour (garde), sauf `force=True`.
    Lève LLMError (OLLAMA_UNAVAILABLE, MODEL_NOT_FOUND, NON_LOCAL_URL) avant toute écriture si le runtime local
    n'est pas prêt ; aucun repli.
    """
    if stage not in STAGES:
        raise ValueError(f"Étape inconnue : {stage} (étapes d'un run : {', '.join(STAGES)}).")
    settings = settings or runtime_settings()
    files = _selected_files(metadata, interview_ids)
    todo = files if force else [f for f in files if not stage_complete(stage, f)]
    todo_ids = [f["ingestion"]["interview_id"] for f in todo]
    skipped = [f["ingestion"]["interview_id"] for f in files if f not in todo]
    records: dict = {}
    started = time.monotonic()
    if todo:
        if runner is None:
            checker = MethodChecker({f["ingestion"]["interview_id"]: f["ingestion"]["output_dir"] for f in todo})
            runner = make_runner(settings, journal_dir=stage_dir(metadata, stage), checker=checker)
        runner.require_ready()  # Ollama absent ou modèle absent : erreur claire, rien n'est écrit
        metadata = _execute(stage, metadata, todo_ids, runner, settings, force)
        records = runner.records
    status = _status(metadata, stage, todo_ids, skipped, records, settings, round(time.monotonic() - started, 1))
    if todo or (metadata.get(f"stage{stage}_local") or {}).get("status") not in (None, STAGE_COMPLETE):
        _record(metadata, status)
    return {"metadata": metadata, "status": status}


def run_stage3(metadata: dict, interview_ids: list[str] | None = None, **kwargs) -> dict:
    """Étape 3 : audit des locuteurs, puis Practice Extractor et Interaction Signal Reader (voir `run_stage`)."""
    return run_stage("3", metadata, interview_ids, **kwargs)


def run_stage4(metadata: dict, interview_ids: list[str] | None = None, **kwargs) -> dict:
    """Étape 4 : candidats déterministes, Accountability Episode Builder (un appel par bloc), validation, fusion."""
    return run_stage("4", metadata, interview_ids, **kwargs)


def run_stage5(metadata: dict, interview_ids: list[str] | None = None, **kwargs) -> dict:
    """Étape 5 : préparation déterministe, Trajectory Mapper (un appel par entretien, jamais découpé), validation."""
    return run_stage("5", metadata, interview_ids, **kwargs)


def run_until(metadata: dict, until: str, interview_ids: list[str] | None = None, *,
              settings: LLMSettings | None = None, runner=None) -> dict:
    """Étapes 3 → `until` (« 3 », « 4 » ou « 5 ») avec la garde ; s'arrête à la première étape non terminée.
    Renvoie {"metadata", "status", "stage"}."""
    if until not in STAGES:
        raise ValueError(f"Étape inconnue : {until} (étapes d'un run : {', '.join(STAGES)}).")
    settings = settings or runtime_settings()
    for stage in STAGES[:STAGES.index(until) + 1]:
        result = run_stage(stage, metadata, interview_ids, settings=settings, runner=runner)
        metadata = result["metadata"]
        if result["status"]["status"] != STAGE_COMPLETE:
            return {**result, "stage": stage}
    return {**result, "stage": until}


def _agent_statuses(info: dict, stage: str) -> dict:
    if stage == "3":
        summaries = (info.get("analysis") or {}).get("agents") or {}
        audit = (info.get("analysis") or {}).get("speaker_audit")
        if audit:
            summaries = {analysis.AUDITOR.name: audit, **summaries}
    else:
        summary = info.get(SUMMARY_KEY_BY_STAGE[stage])
        summaries = {SPEC_BY_STAGE[stage].name: summary} if summary else {}
    return {name: (a or {}).get("status") for name, a in summaries.items()}


def _errors(info: dict, stage: str) -> list[str]:
    if stage == "3":
        agents = {**((info.get("analysis") or {}).get("agents") or {}),
                  "audit": (info.get("analysis") or {}).get("speaker_audit") or {}}
        return [f"{name} : {a['error']['message']}" for name, a in agents.items()
                if (a or {}).get("error") and (a or {}).get("status") not in analysis.DONE_STATUSES]
    summary = info.get(SUMMARY_KEY_BY_STAGE[stage]) or {}
    return [summary["error"]["message"]] if summary.get("error") else []


def _size_warning(summary: dict, stage: str) -> str | None:
    """Requête au-delà du seuil d'un appel unique : signalée, jamais découpée (étape 5) ni coupée (étape 4)."""
    if not summary.get("over_single_call_threshold"):
        return None
    return (f"Requête au-delà du seuil d'un appel unique ({summary.get('estimated_input_tokens')} tokens estimés) : "
            "conservée telle quelle, avertissement PAYLOAD_OVER_THRESHOLD, vérification humaine recommandée. "
            "Vérifiez que la fenêtre de contexte du modèle local (TRACE_OLLAMA_NUM_CTX) la contient.")


def _status(metadata: dict, stage: str, interview_ids: list[str], skipped: list[str], records: dict,
            settings: LLMSettings, duration: float) -> dict:
    by_interview = {f["ingestion"]["interview_id"]: f for f in metadata.get("files", []) if "ingestion" in f}
    order = [f["ingestion"]["interview_id"] for f in metadata.get("files", []) if "ingestion" in f]
    done = DONE_BY_STAGE[stage]
    interviews = []
    for interview_id in sorted({*interview_ids, *skipped}, key=lambda i: order.index(i) if i in order else len(order)):
        info = by_interview.get(interview_id, {})
        summary = info.get(SUMMARY_KEY_BY_STAGE[stage]) or {}
        statuses = _agent_statuses(info, stage)
        entry = {"interview_id": interview_id, "agent_statuses": statuses, "skipped": interview_id in skipped,
                 "errors": [], "estimated_input_tokens": summary.get("estimated_input_tokens"),
                 "over_single_call_threshold": bool(summary.get("over_single_call_threshold")),
                 "warnings": [w for w in (_size_warning(summary, stage),) if w]}
        if interview_id in skipped:
            interviews.append({**entry, "status": STAGE_COMPLETE, "phase": "déjà terminée — non rejouée"})
            continue
        agent_values = [s for name, s in statuses.items() if stage != "3" or name != analysis.AUDITOR.name]
        if agent_values and all(s in done for s in agent_values):
            state = STAGE_COMPLETE
        elif accountability.STATUS_BLOCKED in statuses.values():
            state = STAGE_BLOCKED
        else:
            state = STAGE_FAILED
        interviews.append({**entry, "status": state, "phase": "exécutée", "errors": _errors(info, stage)})
    states = {i["status"] for i in interviews}
    overall = next((s for s in (STAGE_FAILED, STAGE_BLOCKED) if s in states), STAGE_COMPLETE)
    calls = sorted(records.values(), key=lambda r: r["label"])
    return {
        "pipeline_version": PIPELINE_VERSION, "stage": stage, "run_id": metadata.get("run_id"), "status": overall,
        "updated_at": _now(), "duration_seconds": duration, **runtime_info(settings), "api_calls": 0,
        "skipped": list(skipped), "interviews": interviews,
        "call_count": len(calls), "corrected_calls": sum(c["attempts"] > 1 for c in calls),
        "calls": calls,
    }


def _record(metadata: dict, status: dict) -> None:
    """status.json de l'étape, résumé dans metadata.json et état lisible du pipeline."""
    stage = status["stage"]
    write_json_atomic(stage_dir(metadata, stage) / STATUS_FILENAME, status)
    metadata[f"stage{stage}_local"] = {k: v for k, v in status.items() if k != "calls"}
    if status["status"] != STAGE_COMPLETE:
        for step in PIPELINE_STEPS_BY_STAGE[stage]:
            if metadata["pipeline"].get(step) == "inactive":
                metadata["pipeline"][step] = f"incomplet — {STATUS_LABELS[status['status']]}"
    save_metadata(metadata, Path(metadata["output_dir"]))


def read_status(run_dir: Path, stage: str) -> dict | None:
    path = Path(run_dir) / LOCAL_RUNS_SUBDIR / f"stage{stage}" / STATUS_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# --- Étape 6 : un corpus -----------------------------------------------------------------------------

STAGE6_COMPLETE, STAGE6_NOT_RUN, STAGE6_INCOMPLETE, STAGE6_STALE = "COMPLETE", "NOT_RUN", "INCOMPLETE", "STALE"


def corpus_runs_dir(corpus_dir: Path) -> Path:
    return Path(corpus_dir) / LOCAL_RUNS_SUBDIR / f"stage{CORPUS_STAGE}"


def stage6_identity(prepared) -> dict:
    """Identité d'une analyse de l'étape 6 : fichiers de l'étape 5 inclus (empreintes), requête exacte, agent,
    prompt, schéma et versions des préparateur, validateur et contrôle du corpus. Aucun horodatage."""
    return {"corpus_id": prepared.corpus_id, "input_hashes": prepared.input_hashes,
            "payload_sha256": sha256_text(prepared.request["user_message"]) if prepared.request else None,
            **cross_interview.SPEC.identity(), "preprocessor_version": cm.PREPROCESSOR_VERSION,
            "validator_version": cross_validator.VALIDATOR_VERSION,
            "corpus_check_version": corpus.CORPUS_CHECK_VERSION}


def stage6_state(prepared, corpus_dir: Path) -> dict:
    """État de l'analyse du corpus sur le disque : COMPLETE (exploitable et de même identité), NOT_RUN,
    INCOMPLETE ou STALE (même corpus, mais autre prompt, schéma, version ou requête)."""
    outputs = cross_interview.read_outputs(Path(corpus_dir))
    if not outputs["comparison"] or not outputs["manifest"]:
        return {"status": STAGE6_NOT_RUN, "reasons": ["Étape 6 : aucune comparaison pour ce corpus."]}
    document, manifest = json.loads(outputs["comparison"]), json.loads(outputs["manifest"])
    if document.get("status") not in cross_interview.DONE_STATUSES or document.get("analysis_complete") is not True:
        return {"status": STAGE6_INCOMPLETE, "reasons": [f"Étape 6 : {document.get('status')}."]}
    changed = sorted(k for k, v in stage6_identity(prepared).items() if manifest.get(k) != v)
    if changed:
        return {"status": STAGE6_STALE,
                "reasons": [f"Étape 6 calculée avec une autre identité ({', '.join(changed)})."]}
    return {"status": STAGE6_COMPLETE, "reasons": []}


def run_stage6(uploads: list[tuple[str, bytes]], *, settings: LLMSettings | None = None, base_dir: Path | None = None,
               force: bool = False, runner=None) -> dict:
    """Étape 6 sur les triplets de l'étape 5 importés, avec le modèle local.

    Contrôle du corpus et préparation habituels, UN appel du Cross-Interview Comparator, validation et sorties
    habituelles dans <cross_interview>/<corpus_id>/. Garde : un corpus déjà analysé à l'identique n'est pas rejoué
    (sauf `force=True`). Moins de deux entretiens exploitables : BLOCKED, aucun appel. Lève LLMError avant toute
    écriture si le runtime local n'est pas prêt (corpus exploitable seulement).
    Renvoie {"status", "manifest", "corpus_dir", "prepared"}.
    """
    settings = settings or runtime_settings()
    prepared = cross_interview.prepare_stage6(uploads)
    corpus_dir = cross_interview.corpus_dir_for(prepared, base_dir)
    started = time.monotonic()
    if not force and not prepared.blocked and stage6_state(prepared, corpus_dir)["status"] == STAGE6_COMPLETE:
        manifest = {**json.loads(cross_interview.read_outputs(corpus_dir)["manifest"]), "corpus_dir": str(corpus_dir)}
        status = _corpus_status(prepared, corpus_dir, {}, manifest, settings, 0.0, skipped=True)
        return {"status": status, "manifest": manifest, "corpus_dir": str(corpus_dir), "prepared": prepared}
    if runner is None:
        runner = make_runner(settings, journal_dir=corpus_runs_dir(corpus_dir), checker=MethodChecker(corpus=prepared))
    if not prepared.blocked:
        runner.require_ready()  # Ollama absent ou modèle absent : erreur claire, rien n'est écrit
    manifest = cross_interview.run_stage6(uploads, settings=settings, cache=AnalysisCache(), base_dir=base_dir,
                                          force=force, client=runner)
    status = _corpus_status(prepared, corpus_dir, runner.records, manifest, settings,
                            round(time.monotonic() - started, 1))
    write_json_atomic(corpus_runs_dir(corpus_dir) / STATUS_FILENAME, status)
    return {"status": status, "manifest": manifest, "corpus_dir": str(corpus_dir), "prepared": prepared}


def _corpus_status(prepared, corpus_dir: Path, records: dict, manifest: dict, settings: LLMSettings,
                   duration: float, skipped: bool = False) -> dict:
    if manifest.get("status") == cross_interview.STATUS_BLOCKED:
        state = STAGE_BLOCKED
    elif manifest.get("status") in cross_interview.DONE_STATUSES:
        state = STAGE_COMPLETE
    else:
        state = STAGE_FAILED
    estimated = prepared.estimated_input_tokens
    warnings = []
    if estimated > cross_interview.SINGLE_CALL_MAX_INPUT_TOKENS:
        warnings.append(f"Requête au-delà du seuil d'un appel unique ({estimated} tokens estimés, seuil "
                        f"{cross_interview.SINGLE_CALL_MAX_INPUT_TOKENS}) : requête unique conservée, avertissement "
                        "PAYLOAD_OVER_THRESHOLD ; vérifiez la fenêtre de contexte du modèle local (TRACE_OLLAMA_NUM_CTX).")
    checked = prepared.checked
    calls = sorted(records.values(), key=lambda r: r["label"])
    return {
        "pipeline_version": PIPELINE_VERSION, "stage": CORPUS_STAGE, "corpus_id": prepared.corpus_id,
        "corpus_dir": display_path(Path(corpus_dir)), "status": state, "updated_at": _now(),
        "duration_seconds": duration, **runtime_info(settings), "api_calls": 0, "skipped": skipped,
        "phase": "déjà terminée — non rejouée" if skipped else "comparaison",
        "mode": checked["mode"], "n_total": checked["n_total"], "n_usable": checked["n_usable"],
        "rows": [{"interview_id": r["interview_id"], "status": r["status"], "reasons": r["reasons"]}
                 for r in checked["rows"]],
        "unrecognized": checked["unrecognized"], "estimated_input_tokens": estimated, "warnings": warnings,
        "reason": (manifest.get("error") or {}).get("message") if state in (STAGE_BLOCKED, STAGE_FAILED) else None,
        "manifest_status": manifest.get("status"),
        "call_count": len(calls), "corrected_calls": sum(c["attempts"] > 1 for c in calls), "calls": calls,
    }


def read_corpus_status(corpus_dir: Path) -> dict | None:
    path = corpus_runs_dir(corpus_dir) / STATUS_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


__all__ = ["LLMError", "run_stage", "run_stage3", "run_stage4", "run_stage5", "run_until", "run_stage6",
           "runtime_status", "runtime_settings", "make_runner", "stage_state", "stage6_state"]
