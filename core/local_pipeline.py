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
from core import accountability_episode_validator as episode_validator
from core.final_report import STAGE6_NOT_APPLICABLE, STAGE6_NOT_APPLICABLE_REASON
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
# Étape 6 avec un seul entretien exploitable : comparaison impossible par définition, ni erreur ni appel
STAGE_NOT_APPLICABLE = STAGE6_NOT_APPLICABLE
STATUS_LABELS = {
    STAGE_COMPLETE: "terminée",
    STAGE_FAILED: "échec ou analyse incomplète (voir les statuts des agents)",
    STAGE_BLOCKED: "bloquée (étape précédente en échec, absente ou périmée)",
    STAGE_NOT_APPLICABLE: "non applicable : " + STAGE6_NOT_APPLICABLE_REASON,
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


def stage4_guard_state(interview_dir: Path) -> dict:
    """État de l'étape 4 pour la garde : celui de trajectory.stage4_state, et PÉRIMÉE si elle a été validée par une
    autre version du validateur des épisodes (étape 4.3 : réparations ciblées, invariants bloquants) — la rejouer
    reprend les blocs du cache et ne demande que les réparations ciblées."""
    state = trajectory.stage4_state(interview_dir)
    if state["status"] != trajectory.STAGE4_COMPLETE:
        return state
    document = trajectory._read_json(interview_dir / config.ANALYSIS_SUBDIR / config.ACCOUNTABILITY_EPISODES_FILENAME)
    version = (document or {}).get("validator_version")
    if version != episode_validator.VALIDATOR_VERSION:
        return {**state, "status": trajectory.STAGE4_STALE, "reasons": [
            f"Étape 4 périmée : validée par la version {version} du validateur des épisodes (actuelle : "
            f"{episode_validator.VALIDATOR_VERSION}). Relancez l'étape 4 : blocs repris du cache, seules les "
            "réparations ciblées sont demandées au modèle."]}
    return state


def stage_state(stage: str, info: dict) -> dict:
    """État d'une étape pour un entretien, lu sur le disque (règles existantes des étapes 4 et 5)."""
    interview_dir = Path(info["ingestion"]["output_dir"])
    if stage == "3":
        return accountability.stage3_state(interview_dir / config.ANALYSIS_SUBDIR)
    if stage == "4":
        return stage4_guard_state(interview_dir)
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

class ReportingCache(AnalysisCache):
    """Cache TRACE habituel qui signale à la progression chaque résultat validé repris (aucun appel au modèle) et
    chaque réponse validée enregistrée."""

    def __init__(self, on_event, root: Path | None = None):
        super().__init__(root)
        self.on_event = on_event

    def load(self, key_fields: dict, output_model):
        entry = super().load(key_fields, output_model)
        if entry is not None and self.on_event is not None:
            self.on_event({"type": "cached", "agent": key_fields["agent"], "cache_key": entry.get("cache_key"),
                           "at": time.time()})
        return entry

    def store(self, key_fields: dict, output: dict, call: dict) -> Path:
        path = super().store(key_fields, output, call)
        if self.on_event is not None:  # réponse validée enregistrée : elle ne sera jamais recalculée
            self.on_event({"type": "stored", "request_id": (call or {}).get("request_id"),
                           "agent": key_fields["agent"], "at": time.time()})
        return path


def _attach_progress(runner, progress) -> None:
    if progress is not None and hasattr(runner, "on_event"):
        runner.on_event = progress


def _execute(stage: str, metadata: dict, ids: list[str], runner, settings: LLMSettings, force: bool,
             progress=None) -> dict:
    cache = ReportingCache(progress) if progress is not None else AnalysisCache()
    if stage == "3":
        return analysis.analyze_run(metadata, ids, settings=settings, cache=cache, force=force, client=runner)
    if stage == "4":
        return accountability.analyze_run_stage4(metadata, ids, settings=settings, cache=cache, force=force,
                                                 client=runner)
    return trajectory.analyze_run_stage5(metadata, ids, settings=settings, cache=cache, force=force, client=runner)


def run_stage(stage: str, metadata: dict, interview_ids: list[str] | None = None, *,
              settings: LLMSettings | None = None, force: bool = False, runner=None, progress=None) -> dict:
    """Exécute UNE étape (« 3 », « 4 » ou « 5 ») avec le modèle local. Renvoie {"metadata", "status"}.

    Seulement pour les entretiens dont l'étape n'est pas déjà complète et à jour (garde), sauf `force=True`.
    Lève LLMError (OLLAMA_UNAVAILABLE, MODEL_NOT_FOUND, NON_LOCAL_URL) avant toute écriture si le runtime local
    n'est pas prêt ; aucun repli. `progress(event)` : progression (core/local_jobs.py) — début d'étape, appels,
    tokens générés, résultats repris du cache.
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
        _attach_progress(runner, progress)
        if progress is not None:
            progress({"type": "stage_started", "stage": stage, "interview_ids": todo_ids, "metadata": metadata,
                      "settings": settings, "force": force, "at": time.time()})
        metadata = _execute(stage, metadata, todo_ids, runner, settings, force, progress)
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
              settings: LLMSettings | None = None, runner=None, progress=None) -> dict:
    """Étapes 3 → `until` (« 3 », « 4 » ou « 5 ») avec la garde ; s'arrête à la première étape non terminée.
    Renvoie {"metadata", "status", "stage"}."""
    if until not in STAGES:
        raise ValueError(f"Étape inconnue : {until} (étapes d'un run : {', '.join(STAGES)}).")
    settings = settings or runtime_settings()
    for stage in STAGES[:STAGES.index(until) + 1]:
        result = run_stage(stage, metadata, interview_ids, settings=settings, runner=runner, progress=progress)
        metadata = result["metadata"]
        if result["status"]["status"] != STAGE_COMPLETE:
            return {**result, "stage": stage}
    return {**result, "stage": until}


# --- Étape 3 : resélection sans modèle ------------------------------------------------------------------

REFILTER_REPORT_FILENAME = "refilter_report.json"


class NoModelClient:
    """Client des agents qui refuse tout appel : la resélection n'utilise que les réponses déjà validées du cache."""

    records: dict = {}

    async def complete_json(self, **kwargs):
        raise LLMError("UNEXPECTED_ERROR", "resélection de l'étape 3 : aucun appel au modèle n'est autorisé")

    async def aclose(self) -> None:
        return None


def _read_analysis(info: dict, filename: str) -> dict | None:
    path = Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR / filename
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _object_key(item: dict) -> tuple:
    """Identité d'un objet indépendante de sa numérotation : type, résumé ou forme, citations."""
    evidence = tuple((e.get("turn_id"), e.get("quote")) for e in item.get("evidence", []))
    return (item.get("signal_type") or item.get("summary"), evidence)


def _fate(before: dict | None, after: dict | None, items_key: str, set_aside_key: str, id_key: str) -> list[dict]:
    """Devenir de chaque objet de la sortie précédente : conservé (nouvel identifiant) ou écarté (raison)."""
    if not before:
        return []
    kept = {_object_key(i): i for i in (after or {}).get(items_key, [])}
    aside = {_object_key(i): i for i in (after or {}).get(set_aside_key, [])}
    fates = []
    for item in before.get(items_key, []):
        key = _object_key(item)
        row = {"previous_id": item.get(id_key), "label": (item.get("signal_type") or "") + (
            f" « {item.get('surface_form')} »" if item.get("surface_form") else f" {item.get('summary', '')[:90]}"),
               "turn_ids": item.get("turn_ids") or [e.get("turn_id") for e in item.get("evidence", [])],
               "quotes": [e.get("quote") for e in item.get("evidence", [])]}
        if key in kept:
            row.update(fate="kept", new_id=kept[key].get(id_key))
        elif key in aside:
            row.update(fate="set_aside", reason=aside[key].get("set_aside_reason"))
        else:
            row.update(fate="absent")
        fates.append(row)
    return fates


def refilter_stage3(metadata: dict, interview_ids: list[str] | None = None, *,
                    settings: LLMSettings | None = None) -> dict:
    """Réapplique à l'étape 3 le post-traitement déterministe actuel (sélectivité des pratiques et des signaux,
    normalisations formelles) et la validation habituelle, à partir des réponses DÉJÀ VALIDÉES du cache TRACE, sans
    aucun appel au modèle. Si une réponse manque au cache, rien n'est écrit (ValueError). Les sorties de l'étape 3 sont
    réécrites dans leur format habituel ; l'étape 4 devient « périmée » et sera rejouée par la garde habituelle.
    Renvoie {"metadata", "status", "report"} ; le rapport est aussi écrit dans local_runs/stage3/refilter_report.json."""
    settings = settings or runtime_settings()
    files = _selected_files(metadata, interview_ids)
    ids = [f["ingestion"]["interview_id"] for f in files]
    cache = AnalysisCache()
    plan = analysis.plan_analysis(metadata, ids, settings, cache)
    if plan["calls"]:
        raise ValueError(f"Resélection impossible sans modèle : {plan['calls']} réponse(s) de l'étape 3 absente(s) du "
                         f"cache TRACE pour le modèle {settings.model} (TRACE_LOCAL_MODEL doit être celui de l'exécution "
                         "d'origine). Rien n'a été modifié.")
    previous = {f["ingestion"]["interview_id"]: (_read_analysis(f, analysis.PRACTICE.output_filename),
                                                _read_analysis(f, analysis.INTERACTION.output_filename)) for f in files}
    started = time.monotonic()
    metadata = analysis.analyze_run(metadata, ids, settings=settings, cache=cache, force=False, client=NoModelClient())
    by_id = {f["ingestion"]["interview_id"]: f for f in analysis.eligible_files(metadata)}
    interviews = []
    for interview_id in ids:
        info = by_id[interview_id]
        practices = _read_analysis(info, analysis.PRACTICE.output_filename) or {}
        signals = _read_analysis(info, analysis.INTERACTION.output_filename) or {}
        old_practices, old_signals = previous[interview_id]
        interviews.append({
            "interview_id": interview_id,
            "practices_before": len((old_practices or {}).get("practices", [])) if old_practices else None,
            "practices_kept": len(practices.get("practices", [])),
            "practices_set_aside": [{"reason": p.get("set_aside_reason"), "use_status": p.get("use_status"),
                                     "summary": p.get("summary"), "quotes": [e.get("quote") for e in p["evidence"]]}
                                    for p in practices.get("set_aside_practices", [])],
            "non_use_reason_normalized": (practices.get("practice_selectivity") or {}).get(
                "non_use_reason_normalized", []),
            "signals_before": len((old_signals or {}).get("signals", [])) if old_signals else None,
            "signals_kept": len(signals.get("signals", [])),
            "signals_set_aside": [{"reason": s.get("set_aside_reason"), "signal_type": s.get("signal_type"),
                                   "surface_form": s.get("surface_form"), "turn_ids": s.get("turn_ids"),
                                   "quotes": [e.get("quote") for e in s["evidence"]]}
                                  for s in signals.get("set_aside_signals", [])],
            "previous_practices": _fate(old_practices, practices, "practices", "set_aside_practices", "practice_id"),
            "previous_signals": _fate(old_signals, signals, "signals", "set_aside_signals", "signal_id"),
            "remaining_non_use_reason_mismatch": [p["practice_id"] for p in practices.get("practices", [])
                                                  if "NON_USE_REASON_MISMATCH" in p.get("review_reasons", [])],
        })
    status = _status(metadata, "3", ids, [], {}, settings, round(time.monotonic() - started, 1))
    _record(metadata, status)
    report = {"pipeline_version": PIPELINE_VERSION, "run_id": metadata.get("run_id"), "updated_at": _now(),
              "model_calls": 0, "model": settings.model, "interviews": interviews}
    write_json_atomic(stage_dir(metadata, "3") / REFILTER_REPORT_FILENAME, report)
    return {"metadata": metadata, "status": status, "report": report}


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
               force: bool = False, runner=None, progress=None) -> dict:
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
    _attach_progress(runner, progress)
    if progress is not None:
        progress({"type": "stage_started", "stage": CORPUS_STAGE, "corpus_id": prepared.corpus_id,
                  "prepared": prepared, "settings": settings, "force": force, "at": time.time()})
    cache = ReportingCache(progress) if progress is not None else AnalysisCache()
    manifest = cross_interview.run_stage6(uploads, settings=settings, cache=cache, base_dir=base_dir,
                                          force=force, client=runner)
    status = _corpus_status(prepared, corpus_dir, runner.records, manifest, settings,
                            round(time.monotonic() - started, 1))
    write_json_atomic(corpus_runs_dir(corpus_dir) / STATUS_FILENAME, status)
    return {"status": status, "manifest": manifest, "corpus_dir": str(corpus_dir), "prepared": prepared}


def _corpus_status(prepared, corpus_dir: Path, records: dict, manifest: dict, settings: LLMSettings,
                   duration: float, skipped: bool = False) -> dict:
    if manifest.get("status") == cross_interview.STATUS_BLOCKED:
        # un seul entretien exploitable : comparaison inter-entretiens impossible par définition (non applicable)
        state = STAGE_NOT_APPLICABLE if prepared.checked["n_usable"] == 1 else STAGE_BLOCKED
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
        "phase": ("déjà terminée — non rejouée" if skipped else "non applicable — aucune comparaison"
                  if state == STAGE_NOT_APPLICABLE else "comparaison"),
        "mode": checked["mode"], "n_total": checked["n_total"], "n_usable": checked["n_usable"],
        "rows": [{"interview_id": r["interview_id"], "status": r["status"], "reasons": r["reasons"]}
                 for r in checked["rows"]],
        "unrecognized": checked["unrecognized"], "estimated_input_tokens": estimated, "warnings": warnings,
        "reason": (STAGE6_NOT_APPLICABLE_REASON if state == STAGE_NOT_APPLICABLE else
                   (manifest.get("error") or {}).get("message") if state in (STAGE_BLOCKED, STAGE_FAILED) else None),
        "manifest_status": manifest.get("status"),
        "call_count": len(calls), "corrected_calls": sum(c["attempts"] > 1 for c in calls), "calls": calls,
    }


def read_corpus_status(corpus_dir: Path) -> dict | None:
    path = corpus_runs_dir(corpus_dir) / STATUS_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


__all__ = ["LLMError", "run_stage", "run_stage3", "run_stage4", "run_stage5", "run_until", "run_stage6",
           "runtime_status", "runtime_settings", "make_runner", "stage_state", "stage6_state"]
