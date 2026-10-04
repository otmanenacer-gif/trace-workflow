"""Exécution des étapes 3 à 6 EN ARRIÈRE-PLAN, indépendante de la page Streamlit, avec reprise.

    Streamlit (ou la CLI) → start_run_job / start_stage6_job → processus détaché « python -m core.local_jobs DOSSIER »
      → local_pipeline.run_stage(…, progress=…) / run_stage6 → LocalAgentRunner → Ollama local
      → job.json (état, agent en cours, liste des appels, mesures), relu par l'interface toutes les deux secondes.

- Le travail ne dépend pas de la page : rafraîchir, fermer le navigateur ou redémarrer Streamlit n'interrompt rien ;
  à la réouverture, l'interface retrouve le travail en cours (job.json) et en affiche la progression.
- Chaque réponse validée d'un agent est enregistrée IMMÉDIATEMENT dans le cache TRACE (core/analysis_cache.py), avant
  l'appel suivant. Si le processus s'arrête (ordinateur éteint, Ollama arrêté, fermeture de la fenêtre), relancer
  l'étape reprend au premier appel non terminé : un résultat validé n'est jamais recalculé.
- Vivacité : le processus écrit un battement (heartbeat) dans job.json toutes les deux secondes ; un travail « en
  cours » sans battement depuis STALE_AFTER secondes est « interrompu » (aucun signal envoyé aux processus : sous
  Windows, os.kill(pid, 0) terminerait le processus).

Aucune API externe : le processus détaché exécute le même pipeline local (Ollama sur la boucle locale).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from core import config

JOB_FILENAME = "job.json"
LOG_FILENAME = "job.log"
INPUTS_SUBDIR = "inputs"
HEARTBEAT_SECONDS = 2.0
STALE_AFTER = 20.0          # secondes sans battement : travail interrompu
START_GRACE = 60.0          # délai laissé au processus pour démarrer (import, lecture du run)
ENV_JOB_MODE = "TRACE_JOB_MODE"   # subprocess (défaut) | thread | inline (tests)
JOB_MODES = ("subprocess", "thread", "inline")

# États
STARTING, RUNNING, COMPLETE, FAILED, BLOCKED, ERROR, STOPPED = (
    "starting", "running", "complete", "failed", "blocked", "error", "stopped")
INTERRUPTED = "interrupted"  # état déduit : en cours sans battement récent
ACTIVE = (STARTING, RUNNING)
FINAL = (COMPLETE, FAILED, BLOCKED, ERROR, STOPPED)
JOB_LABELS = {STARTING: "démarrage", RUNNING: "en cours", COMPLETE: "terminée", FAILED: "échec ou analyse incomplète",
              BLOCKED: "bloquée", ERROR: "erreur", STOPPED: "arrêtée", INTERRUPTED: "interrompue"}

# Liste des appels
PENDING, CALL_RUNNING, DONE, ISSUES, CACHED, CALL_FAILED, NOT_NEEDED, NOT_RUN = (
    "pending", "running", "done", "issues", "cached", "failed", "not_needed", "not_run")
ITEM_ICONS = {PENDING: "○", CALL_RUNNING: "⏳", DONE: "✓", ISSUES: "✓⚠", CACHED: "✓", CALL_FAILED: "✗",
              NOT_NEEDED: "–", NOT_RUN: "○"}
ITEM_LABELS = {PENDING: "à faire", CALL_RUNNING: "en cours", DONE: "validé", ISSUES: "validé avec anomalies signalées",
               CACHED: "déjà validé (repris, non recalculé)", CALL_FAILED: "échec", NOT_NEEDED: "non nécessaire",
               NOT_RUN: "non exécuté"}

logger = logging.getLogger("trace.local")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_job_dir(run_dir: Path) -> Path:
    from core.local_pipeline import LOCAL_RUNS_SUBDIR
    return Path(run_dir) / LOCAL_RUNS_SUBDIR / "job"


def stage6_job_dir(base_dir: Path | None = None) -> Path:
    return Path(base_dir or config.CROSS_INTERVIEW_DIR) / "local_jobs" / "stage6"


# --- job.json : écriture atomique, lecture tolérante --------------------------------------------------

def _write(path: Path, data: dict) -> None:
    """Écriture atomique ; sous Windows, un lecteur peut bloquer brièvement le renommage : nouvel essai."""
    from core.analysis_cache import write_json_atomic
    for attempt in range(40):
        try:
            write_json_atomic(path, data)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def _read(path: Path) -> dict | None:
    for _ in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, ValueError):
            time.sleep(0.05)
    return None


def read_job(job_dir: Path, stale_after: float = STALE_AFTER) -> dict | None:
    """État du travail, avec `state` (état déduit : INTERRUPTED si en cours sans battement récent), `alive`,
    `elapsed_seconds` et, pour l'appel en cours, `current.elapsed_seconds`."""
    job = _read(Path(job_dir) / JOB_FILENAME)
    if job is None:
        return None
    now = time.time()
    state = job.get("status")
    if state in ACTIVE:
        limit = START_GRACE if state == STARTING else stale_after
        if now - float(job.get("heartbeat_at") or 0) > limit:
            state = INTERRUPTED
    job["state"] = state
    job["alive"] = state in ACTIVE
    end = job.get("finished_at_epoch") or (now if job["alive"] else job.get("heartbeat_at") or now)
    job["elapsed_seconds"] = round(max(0.0, end - float(job.get("started_at_epoch") or end)), 1)
    current = job.get("current")
    if current and job["alive"]:
        current["elapsed_seconds"] = round(now - float(current.get("started_at") or now), 1)
    return job


def job_log_tail(job_dir: Path, lines: int = 15) -> str:
    path = Path(job_dir) / LOG_FILENAME
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# --- Liste des appels attendus -----------------------------------------------------------------------

def _item(stage: str, label: str, agent: str, agent_label: str, cache_key: str | None = None,
          optional: bool = False) -> dict:
    return {"stage": stage, "label": label, "interview_id": label.split("/", 1)[0], "agent": agent,
            "agent_label": agent_label, "cache_key": cache_key, "optional": optional, "status": PENDING,
            "attempts": None, "duration_seconds": None, "input_tokens": None, "output_tokens": None,
            "tokens_per_second": None, "num_ctx": None, "correction_reasons": [], "call_id": None, "saved": False}


def _stage3_items(metadata: dict, ids: list[str], settings, force: bool) -> list[dict]:
    from core import analysis
    from core.analysis_cache import AnalysisCache, compute_cache_key
    cache = AnalysisCache()
    items = []
    for info in analysis.eligible_files(metadata):
        interview_id = info["ingestion"]["interview_id"]
        if interview_id not in ids:
            continue
        prepared = analysis.prepare_interview(info["ingestion"])
        request = analysis.speaker_audit.prepare_audit(prepared.transcript)
        warnings: dict | None = {}
        if request.needs_llm:
            fields = analysis.audit_cache_key_fields(prepared, request, settings)
            items.append(_item("3", f"{interview_id}/{analysis.AUDITOR.name}", analysis.AUDITOR.name,
                               analysis.AUDITOR.label, compute_cache_key(fields)))
            entry = None if force else cache.load(fields, analysis.AUDITOR.output_model)
            warnings = None
            if entry is not None:
                document = analysis.speaker_audit.build_audit_document(prepared.transcript, request, entry["output"])
                warnings = analysis.speaker_audit.agent_warnings(document, prepared.transcript)
        agent_prepared = analysis.with_speaker_warnings(prepared, warnings) if warnings is not None else None
        for spec in analysis.AGENTS:
            chunks = analysis.agent_chunks(spec, prepared, settings)
            base = f"{interview_id}/{spec.name}"
            if len(chunks) > 1:
                for chunk in chunks:
                    key = None
                    if warnings is not None:
                        message = analysis.chunk_request_for(spec, prepared.transcript, chunk, warnings)["user_message"]
                        key = compute_cache_key(analysis.chunk_cache_key_fields(analysis.chunk_spec(spec), message,
                                                                                settings))
                    items.append(_item("3", f"{base}/bloc{chunk.index}", spec.name,
                                       f"{spec.label} — bloc {chunk.index}/{len(chunks)}", key))
                if spec.name == analysis.INTERACTION.name:
                    items.append(_item("3", f"{base}/longue_distance", analysis.LONG_DISTANCE.name,
                                       analysis.LONG_DISTANCE.label, optional=True))
            else:
                key = (compute_cache_key(analysis.cache_key_fields(spec, agent_prepared, settings))
                       if agent_prepared is not None else None)
                items.append(_item("3", base, spec.name, spec.label, key))
    return items


def _stage4_items(metadata: dict, ids: list[str], settings) -> list[dict]:
    from core import accountability
    from core.analysis_cache import compute_cache_key
    items = []
    for info in accountability.eligible_files(metadata):
        interview_id = info["ingestion"]["interview_id"]
        if interview_id not in ids:
            continue
        prepared = accountability.prepare_stage4(info["ingestion"])
        if prepared.blocked or not prepared.needs_llm:
            continue
        for request in prepared.requests:
            key = compute_cache_key(accountability.cache_key_fields(prepared, settings, request["user_message"]))
            suffix = f" — bloc {request['chunk']}" if len(prepared.requests) > 1 else ""
            items.append(_item("4", f"{interview_id}/{accountability.SPEC.name}/bloc{request['chunk']}",
                               accountability.SPEC.name, accountability.SPEC.label + suffix, key))
    return items


def expected_items(event: dict) -> list[dict]:
    """Appels attendus d'une étape (début d'étape) ; liste vide si l'estimation échoue (les appels s'ajoutent
    alors au fil de l'exécution)."""
    stage = event["stage"]
    try:
        if stage == "3":
            return _stage3_items(event["metadata"], event["interview_ids"], event["settings"], event["force"])
        if stage == "4":
            return _stage4_items(event["metadata"], event["interview_ids"], event["settings"])
        if stage == "5":
            from core import trajectory
            return [_item("5", f"{i}/{trajectory.SPEC.name}", trajectory.SPEC.name, trajectory.SPEC.label)
                    for i in event["interview_ids"]]
        from core import cross_interview
        prepared = event["prepared"]
        return [] if prepared.blocked else [_item("6", f"{prepared.corpus_id}/{cross_interview.SPEC.name}",
                                                  cross_interview.SPEC.name, cross_interview.SPEC.label)]
    except Exception:  # noqa: BLE001 — la liste n'est qu'un affichage
        logger.exception("Liste des appels attendus non calculée (étape %s)", stage)
        return []


# --- Suivi d'un travail (dans le processus qui l'exécute) -----------------------------------------------

class JobReporter:
    """Tient job.json à jour : événements de progression (local_pipeline / LocalAgentRunner) et battement."""

    def __init__(self, job_dir: Path):
        self.job_dir = Path(job_dir)
        self.path = self.job_dir / JOB_FILENAME
        self.job = _read(self.path) or {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # état
    def save(self) -> None:
        with self._lock:
            self.job["heartbeat_at"] = time.time()
            self.job["updated_at"] = _now()
            _write(self.path, self.job)

    def start(self) -> None:
        with self._lock:
            self.job.update(status=RUNNING, pid=os.getpid(), started_at=self.job.get("started_at") or _now(),
                            started_at_epoch=self.job.get("started_at_epoch") or time.time(), current=None)
            self.save()
        self._thread = threading.Thread(target=self._beat, name="trace-job-heartbeat", daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self.save()
            except OSError:
                logger.exception("Battement du travail non écrit")

    def finish(self, status: str, message: str | None = None, result: dict | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            for item in self.job.get("checklist", []):
                if item["status"] == CALL_RUNNING:
                    item["status"] = CALL_FAILED if status != COMPLETE else DONE
            self.job.update(status=status, message=message, result=result, current=None, finished_at=_now(),
                            finished_at_epoch=time.time())
            self.save()

    # événements
    def _find(self, label: str) -> dict | None:
        return next((i for i in self.job.setdefault("checklist", []) if i["label"] == label), None)

    def event(self, event: dict) -> None:
        kind = event.get("type")
        with self._lock:
            if kind == "stage_started":
                stage = event["stage"]
                self.job["stage"] = stage
                kept = [i for i in self.job.get("checklist", []) if i["stage"] != stage]
                self.job["checklist"] = kept + expected_items(event)
                self.save()
            elif kind == "call_started":
                item = self._find(event["label"])
                if item is None:
                    item = _item(self.job.get("stage") or "?", event["label"], event.get("agent") or "?",
                                 event.get("agent_label") or event.get("agent") or "agent")
                    self.job["checklist"].append(item)
                item.update(status=CALL_RUNNING, attempts=event["attempt"], num_ctx=event.get("num_ctx"))
                phase = (f" — réparation ciblée de l'objet n° {event['item_index'] + 1}"
                         if event.get("phase") == "repair" and event.get("item_index") is not None else "")
                self.job["current"] = {"label": event["label"], "agent_label": item["agent_label"] + phase,
                                       "attempt": event["attempt"], "started_at": event["at"],
                                       "call_started_at": (self.job.get("current") or {}).get("call_started_at")
                                       if event["attempt"] > 1 else event["at"],
                                       "output_tokens": 0, "num_ctx": event.get("num_ctx"),
                                       "estimated_input_tokens": event.get("estimated_input_tokens")}
                self.save()
            elif kind == "tokens":
                current = self.job.get("current")
                if current and current["label"] == event["label"]:
                    current["output_tokens"] = event["output_tokens"]  # écrit au prochain battement
            elif kind == "call_finished":
                record = event["record"]
                item = self._find(event["label"]) or _item(self.job.get("stage") or "?", event["label"],
                                                           record.get("agent") or "?",
                                                           record.get("agent_label") or "agent")
                if item not in self.job["checklist"]:
                    self.job["checklist"].append(item)
                item.update(status={"accepted": DONE, "accepted_with_issues": ISSUES}.get(record["outcome"],
                                                                                          CALL_FAILED),
                            call_id=record.get("call_id"),
                            **{k: record.get(k) for k in ("attempts", "duration_seconds", "input_tokens",
                                                          "output_tokens", "tokens_per_second", "num_ctx",
                                                          "correction_reasons")})
                self.job.setdefault("calls", []).append(
                    {k: record.get(k) for k in ("label", "agent", "agent_label", "outcome", "attempts",
                                                "prompt_chars", "estimated_input_tokens", "num_ctx", "num_predict",
                                                "input_tokens", "output_tokens", "duration_seconds",
                                                "generation_seconds", "tokens_per_second", "load_seconds",
                                                "response_chars", "correction_reasons", "started_at",
                                                "finished_at", "error", "full_generations", "repairs",
                                                "initial_output_tokens", "repair_output_tokens",
                                                "repair_input_tokens", "repair_seconds", "objects_total",
                                                "objects_repaired", "objects_rejected", "objects_unresolved",
                                                "citations_invalid_initial", "citations_repaired",
                                                "citations_withdrawn", "citations_invalid_after_repair")})
                self.job["current"] = None
                self.save()
            elif kind == "stored":
                item = next((i for i in self.job.get("checklist", []) if i["call_id"]
                             and i["call_id"] == event.get("request_id")), None)
                if item is not None:
                    item["saved"] = True
                    self.save()
            elif kind == "cached":
                items = self.job.get("checklist", [])
                item = next((i for i in items if i["cache_key"] and i["cache_key"] == event.get("cache_key")), None)
                item = item or next((i for i in items if i["status"] == PENDING and i["agent"] == event["agent"]
                                     and not i["optional"]), None)
                if item is not None:
                    item.update(status=CACHED, saved=True)
                self.job["cached_count"] = self.job.get("cached_count", 0) + 1
                self.save()

    def stage_finished(self, stage: str, status: str) -> None:
        """Fin d'une étape : un appel attendu jamais lancé a été repris du cache (étape terminée), n'était pas
        nécessaire (lecture à longue distance facultative) ou n'a pas été exécuté (étape en échec)."""
        from core.local_pipeline import STAGE_COMPLETE
        with self._lock:
            for item in self.job.get("checklist", []):
                if item["stage"] == stage and item["status"] == PENDING:
                    item["status"] = (NOT_NEEDED if item["optional"] else
                                      CACHED if status == STAGE_COMPLETE else NOT_RUN)
            self.save()


# --- Exécution d'un travail -----------------------------------------------------------------------------

def _public_status(status: dict) -> dict:
    return {k: v for k, v in status.items() if k != "calls"}


def execute(job_dir: Path, log_to_file: bool = True) -> dict:
    """Exécute le travail décrit dans job.json (dans le processus courant). Ne lève pas d'exception.
    `log_to_file` : journal Python « trace.local » recopié dans job.log (inutile pour le processus détaché, dont la
    sortie est déjà job.log)."""
    from core import local_pipeline as lp
    from core.llm_client import LLMError
    from core.run_manager import load_metadata

    job_dir = Path(job_dir)
    reporter = JobReporter(job_dir)
    spec = reporter.job.get("spec") or {}
    handler = logging.FileHandler(job_dir / LOG_FILENAME, encoding="utf-8") if log_to_file else logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s : %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    reporter.start()
    logger.info("Travail %s démarré : %s", reporter.job.get("job_id"), json.dumps(spec, ensure_ascii=False))
    try:
        if spec["kind"] == "run":
            metadata = load_metadata(Path(spec["run_dir"]))
            stages = ([spec["stage"]] if spec.get("single", True)
                      else list(lp.STAGES[:lp.STAGES.index(spec["stage"]) + 1]))
            result = None
            for stage in stages:
                result = lp.run_stage(stage, metadata, spec.get("interview_ids"), force=spec.get("force", False),
                                      progress=reporter.event)
                metadata = result["metadata"]
                reporter.stage_finished(stage, result["status"]["status"])
                if result["status"]["status"] != lp.STAGE_COMPLETE:
                    break
            status = result["status"]
        else:
            inputs = sorted((job_dir / INPUTS_SUBDIR).iterdir())
            uploads = [(p.name.split("__", 1)[1], p.read_bytes()) for p in inputs]
            base_dir = Path(spec["base_dir"]) if spec.get("base_dir") else None
            result = lp.run_stage6(uploads, base_dir=base_dir, force=spec.get("force", False),
                                   progress=reporter.event)
            status = {**result["status"], "corpus_dir_abs": str(result["corpus_dir"])}
            reporter.stage_finished(lp.CORPUS_STAGE, status["status"])
        final = {lp.STAGE_COMPLETE: COMPLETE, lp.STAGE_FAILED: FAILED, lp.STAGE_BLOCKED: BLOCKED,
                 lp.STAGE_NOT_APPLICABLE: COMPLETE}[status["status"]]
        reporter.finish(final, lp.STATUS_LABELS[status["status"]], _public_status(status))
    except LLMError as error:  # Ollama arrêté, modèle absent… : erreur claire, aucun repli
        logger.error("Travail interrompu : %s", error.user_message)
        reporter.finish(ERROR, error.user_message, {"error": error.to_dict()})
    except Exception as exc:  # noqa: BLE001 — l'erreur est rapportée dans job.json, jamais silencieuse
        logger.error("Travail interrompu : %s\n%s", type(exc).__name__, traceback.format_exc())
        reporter.finish(ERROR, f"Erreur inattendue ({type(exc).__name__}) : voir {LOG_FILENAME}.", None)
    finally:
        logger.removeHandler(handler)
        handler.close()
    return reporter.job


# --- Lancement ------------------------------------------------------------------------------------------

class JobAlreadyRunning(RuntimeError):
    pass


def job_mode() -> str:
    mode = (os.environ.get(ENV_JOB_MODE) or "subprocess").strip().lower()
    return mode if mode in JOB_MODES else "subprocess"


def worker_command(job_dir: Path) -> list[str]:
    return [sys.executable, "-m", "core.local_jobs", str(job_dir)]


def _spawn(job_dir: Path) -> int:
    """Processus détaché : il survit à la fermeture de la page, du navigateur et de Streamlit."""
    log = open(Path(job_dir) / LOG_FILENAME, "ab")  # noqa: SIM115 — hérité par le processus détaché
    kwargs: dict = {"cwd": str(config.PROJECT_ROOT), "stdin": subprocess.DEVNULL, "stdout": log,
                    "stderr": subprocess.STDOUT, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0x08)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(worker_command(job_dir), **kwargs)  # noqa: S603 — interpréteur courant
    finally:
        log.close()
    return process.pid


def _launch(job_dir: Path, spec: dict, mode: str | None = None) -> dict:
    job_dir = Path(job_dir)
    existing = read_job(job_dir)
    if existing and existing["alive"]:
        raise JobAlreadyRunning(f"Une exécution est déjà en cours ({existing.get('job_id')}).")
    job_dir.mkdir(parents=True, exist_ok=True)
    previous = existing or {}
    job = {"job_id": uuid.uuid4().hex[:12], "spec": spec, "status": STARTING, "mode": mode or job_mode(),
           "created_at": _now(), "started_at": None, "started_at_epoch": None, "heartbeat_at": time.time(),
           "stage": spec.get("stage"), "current": None, "checklist": [], "calls": [], "cached_count": 0,
           "message": None, "result": None, "acknowledged": False,
           "resumed_from": previous.get("job_id") if previous.get("state") in (INTERRUPTED, FAILED, ERROR, STOPPED)
           else None}
    _write(job_dir / JOB_FILENAME, job)
    mode = job["mode"]
    if mode == "inline":
        execute(job_dir)
    elif mode == "thread":
        threading.Thread(target=execute, args=(job_dir,), name="trace-job", daemon=True).start()
    else:
        job["pid"] = _spawn(job_dir)
        current = _read(job_dir / JOB_FILENAME) or job
        if current.get("status") == STARTING:
            current["pid"] = job["pid"]
            _write(job_dir / JOB_FILENAME, current)
    return read_job(job_dir)


def start_run_job(run_dir: Path, stage: str, interview_ids: list[str] | None = None, *, single: bool = True,
                  force: bool = False, mode: str | None = None) -> dict:
    """Étape `stage` d'un run (ou étapes 3 → `stage` si `single=False`), en arrière-plan."""
    spec = {"kind": "run", "run_dir": str(Path(run_dir)), "stage": stage, "single": single,
            "interview_ids": list(interview_ids) if interview_ids is not None else None, "force": force}
    return _launch(run_job_dir(run_dir), spec, mode)


def start_stage6_job(uploads: list[tuple[str, bytes]], *, base_dir: Path | None = None, force: bool = False,
                     mode: str | None = None) -> dict:
    """Étape 6 en arrière-plan ; les fichiers importés sont copiés dans le dossier du travail."""
    job_dir = stage6_job_dir(base_dir)
    existing = read_job(job_dir)
    if existing and existing["alive"]:
        raise JobAlreadyRunning(f"Une exécution est déjà en cours ({existing.get('job_id')}).")
    inputs = job_dir / INPUTS_SUBDIR
    if inputs.exists():
        for path in inputs.iterdir():
            path.unlink()
    inputs.mkdir(parents=True, exist_ok=True)
    for index, (name, data) in enumerate(uploads):
        (inputs / f"{index:04d}__{Path(name).name}").write_bytes(data)
    spec = {"kind": "stage6", "stage": "6", "base_dir": str(base_dir) if base_dir else None, "force": force,
            "input_count": len(uploads)}
    return _launch(job_dir, spec, mode)


def acknowledge(job_dir: Path) -> None:
    """L'interface a pris en compte la fin du travail (message affiché une seule fois, même après un refresh)."""
    path = Path(job_dir) / JOB_FILENAME
    job = _read(path)
    if job and job.get("status") in FINAL and not job.get("acknowledged"):
        job["acknowledged"] = True
        _write(path, job)


def stop(job_dir: Path) -> bool:
    """Arrête un travail détaché (les résultats déjà validés restent en cache ; « Reprendre » repart de là)."""
    path = Path(job_dir) / JOB_FILENAME
    job = read_job(job_dir)
    if not job or not job["alive"] or not job.get("pid") or job.get("mode") != "subprocess":
        return False
    try:
        os.kill(int(job["pid"]), signal.SIGTERM)  # Windows : TerminateProcess
    except (OSError, ValueError):
        pass
    time.sleep(0.5)  # le processus ne doit plus écrire de battement après l'état « arrêtée »
    raw = _read(path) or job
    raw.update(status=STOPPED, message="Arrêtée à la demande (résultats validés conservés).", current=None,
               finished_at=_now(), finished_at_epoch=time.time())
    _write(path, raw)
    return True


def find_active_run_job(outputs_dir: Path | None = None) -> Path | None:
    """Dossier du run dont un travail est en cours (pour rouvrir la page sur ce run après un refresh)."""
    from core.run_manager import list_runs
    root = Path(outputs_dir or config.OUTPUTS_DIR)
    for run_id in list_runs(root):
        job = read_job(run_job_dir(root / run_id))
        if job and job["alive"]:
            return root / run_id
    return None


def summary_rows(job: dict) -> list[dict]:
    """Tableau des appels : « Practice Extractor — 2 900 tokens entrée — 620 sortie — 41 s »."""
    return [{"Agent": c.get("agent_label") or c.get("agent"), "Appel": c["label"],
             "Tokens entrée": c.get("input_tokens"), "Tokens sortie": c.get("output_tokens"),
             "Durée (s)": c.get("duration_seconds"), "Tokens/s": c.get("tokens_per_second"),
             "Contexte": c.get("num_ctx"), "Tentatives": c.get("attempts"),
             "Corrections": ", ".join(c.get("correction_reasons") or []) or "—",
             "Issue": c.get("outcome")} for c in job.get("calls", [])]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage : python -m core.local_jobs DOSSIER_DU_TRAVAIL", file=sys.stderr)
        return 2
    config.load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    job = execute(Path(argv[0]), log_to_file=False)
    return 0 if job.get("status") == COMPLETE else 1


if __name__ == "__main__":
    sys.exit(main())
