"""Pipeline complet (étapes 1 → 10) lancé depuis Streamlit : corpus géré par TRACE, exécution en arrière-plan, progression.

    Streamlit → create_corpus(fichiers) → data/corpora/<nom>_<empreinte>/ (noms de fichiers conservés : identifiants
    des entretiens inchangés ; mêmes fichiers → même dossier → même run, donc reprise)
      → start(corpus) → processus détaché « python -m core.pipeline_jobs <corpus> » qui exécute EXACTEMENT
        `trace_local.py pipeline <corpus>` (core/pipeline.py), avec un battement dans <corpus>/.trace/job.json
      → progress(corpus) : lecture seule de <run>/pipeline/pipeline_manifest.json et <run>/batch/batch_manifest.json.

Rafraîchir, fermer la page ou redémarrer Streamlit n'interrompt rien ; « Reprendre » relance la même commande sur le
même corpus (reprise par les caches et les gardes existants). Aucune logique analytique ici.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from core import config, local_jobs

CORPORA_SUBDIR = "corpora"
JOB_SUBDIR = ".trace"
CORPUS_FILENAME = "corpus.json"
HEARTBEAT_SECONDS = 2.0
STALE_AFTER = 30.0
MIN_FILES = 2
RUNNING, FINISHED, ERROR = "running", "finished", "error"
FINAL_OK = ("COMPLETE", "SUCCESS_WITH_WARNINGS")
CONTINUABLE = ("COMPLETE", "SUCCESS", "SUCCESS_WITH_WARNINGS")
STAGES = (("1-5", "Étapes 1–5"), ("6", "Étape 6 — comparaison"), ("7", "Étape 7 — théorisation"),
          ("8", "Étape 8 — rédaction"), ("9", "Étape 9 — validation"), ("10", "Étape 10 — livraison"))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def corpora_dir() -> Path:
    return Path(config.DATA_DIR) / CORPORA_SUBDIR


def job_dir(corpus: Path) -> Path:
    return Path(corpus) / JOB_SUBDIR


# --- Corpus géré par TRACE ------------------------------------------------------------------------------------

def check_files(files: list[tuple[str, bytes]]) -> list[str]:
    """Problèmes bloquants : moins de 2 fichiers, format refusé, noms (ou identifiants d'entretien) en double."""
    problems = []
    if len(files) < MIN_FILES:
        problems.append(f"Au moins {MIN_FILES} entretiens sont nécessaires pour une analyse complète du corpus.")
    names, stems = {}, {}
    for name, _ in files:
        base = Path(name).name
        if base.rsplit(".", 1)[-1].lower() not in config.ALLOWED_EXTENSIONS:
            problems.append(f"Format non pris en charge : {base} (attendu : .txt, .docx, .pdf).")
        names.setdefault(base.lower(), []).append(base)
        stems.setdefault(interview_id(base), []).append(base)
    problems += [f"Deux fichiers portent le même nom : {', '.join(v)}." for v in names.values() if len(v) > 1]
    problems += [f"Même identifiant d'entretien pour : {', '.join(v)} (renommez l'un des fichiers)."
                 for k, v in stems.items() if len(v) > 1 and len(names.get(v[0].lower(), [])) < 2]
    return problems


def interview_id(filename: str) -> str:
    """Identifiant d'entretien que l'ingestion dérivera de ce nom de fichier."""
    from core.transcript_structurer import make_interview_id
    return make_interview_id(Path(filename).name)


def _slug(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_")[:40]
    return slug or "corpus"


def create_corpus(files: list[tuple[str, bytes]], label: str = "") -> Path:
    """Dossier STABLE : mêmes fichiers (noms et contenus) → même dossier → même run (reprise, aucun recalcul)."""
    problems = check_files(files)
    if problems:
        raise ValueError(" ".join(problems))
    digest = hashlib.sha256()
    for name, data in sorted(files):
        digest.update(Path(name).name.encode("utf-8") + b"\0" + hashlib.sha256(data).digest())
    corpus = corpora_dir() / f"{_slug(label)}_{digest.hexdigest()[:12]}"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, data in files:
        path = corpus / Path(name).name
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)
    meta = corpus / JOB_SUBDIR / CORPUS_FILENAME
    meta.parent.mkdir(parents=True, exist_ok=True)
    if not meta.is_file():
        local_jobs._write(meta, {"label": label, "created_at": _now(), "files": sorted(Path(n).name for n, _ in files)})
    return corpus


def list_corpora() -> list[Path]:
    root = corpora_dir()
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if (p / JOB_SUBDIR / CORPUS_FILENAME).is_file()]
    return sorted(found, key=lambda p: (p / JOB_SUBDIR / CORPUS_FILENAME).stat().st_mtime, reverse=True)


# --- Travail en arrière-plan ---------------------------------------------------------------------------------

def read_job(corpus: Path) -> dict | None:
    job = local_jobs._read(job_dir(corpus) / local_jobs.JOB_FILENAME)
    if job is None:
        return None
    alive = job.get("status") == RUNNING and time.time() - float(job.get("heartbeat_at") or 0) <= STALE_AFTER
    job["alive"] = alive
    job["state"] = job.get("status") if alive or job.get("status") != RUNNING else "interrupted"
    end = time.time() if alive else float(job.get("finished_at_epoch") or job.get("heartbeat_at") or time.time())
    job["elapsed_seconds"] = round(max(0.0, end - float(job.get("started_at_epoch") or end)), 1)
    return job


def command(corpus: Path, problematique: str = "") -> list[str]:
    args = ["pipeline", str(corpus)]
    return args + (["--problematique", problematique] if problematique else [])


def start(corpus: Path, problematique: str = "", mode: str | None = None) -> dict:
    """Lance (ou relance : reprise) le pipeline complet du corpus, détaché de la page."""
    corpus = Path(corpus)
    existing = read_job(corpus)
    if existing and existing["alive"]:
        raise local_jobs.JobAlreadyRunning("L'analyse complète de ce corpus est déjà en cours.")
    directory = job_dir(corpus)
    directory.mkdir(parents=True, exist_ok=True)
    mode = mode or local_jobs.job_mode()
    job = {"corpus": str(corpus), "args": command(corpus, problematique), "status": RUNNING, "mode": mode,
           "created_at": _now(), "started_at_epoch": time.time(), "heartbeat_at": time.time(), "exit_code": None,
           "launch_count": (existing or {}).get("launch_count", 0) + 1, "message": None}
    local_jobs._write(directory / local_jobs.JOB_FILENAME, job)
    if mode == "inline":
        execute(corpus)
    elif mode == "thread":
        threading.Thread(target=execute, args=(corpus,), name="trace-pipeline", daemon=True).start()
    else:
        pid = _spawn(corpus)
        current = local_jobs._read(directory / local_jobs.JOB_FILENAME) or job
        current["pid"] = pid
        local_jobs._write(directory / local_jobs.JOB_FILENAME, current)
    return read_job(corpus)


def _spawn(corpus: Path) -> int:
    """Processus détaché (mêmes options que core/local_jobs.py) : il survit à la page et à Streamlit."""
    log = open(job_dir(corpus) / local_jobs.LOG_FILENAME, "ab")  # noqa: SIM115 — hérité par le processus détaché
    kwargs: dict = {"cwd": str(config.PROJECT_ROOT), "stdin": subprocess.DEVNULL, "stdout": log,
                    "stderr": subprocess.STDOUT, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0x08)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen([sys.executable, "-u", "-m", "core.pipeline_jobs", str(corpus)], **kwargs)  # noqa: S603
    finally:
        log.close()
    return process.pid


def execute(corpus: Path) -> dict:
    """Exécute `trace_local.py pipeline <corpus>` dans ce processus, avec un battement. Ne lève pas d'exception."""
    path = job_dir(corpus) / local_jobs.JOB_FILENAME
    job = local_jobs._read(path) or {"args": command(corpus)}
    done = threading.Event()

    def beat() -> None:
        while not done.wait(HEARTBEAT_SECONDS):
            current = local_jobs._read(path) or job
            if current.get("status") == RUNNING:
                current["heartbeat_at"] = time.time()
                local_jobs._write(path, current)

    threading.Thread(target=beat, name="trace-pipeline-heartbeat", daemon=True).start()
    code, message = None, None
    try:
        from scripts import trace_local
        if local_jobs.job_mode() == "subprocess":
            code = trace_local.main(job["args"])
        else:  # thread / inline : sortie recopiée dans job.log (la page lit le manifeste, pas la console)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = trace_local.main(job["args"])
            with open(job_dir(corpus) / local_jobs.LOG_FILENAME, "a", encoding="utf-8") as log:
                log.write(buffer.getvalue())
    except KeyboardInterrupt:
        code, message = 130, "interrompu"
    except BaseException as exc:  # noqa: BLE001 — rapporté dans job.json, jamais silencieux
        code, message = None, f"Erreur inattendue ({type(exc).__name__}) : {exc}"
        traceback.print_exc()
    finally:
        done.set()
    final = local_jobs._read(path) or job
    final.update(status=FINISHED if code is not None else ERROR, exit_code=code, message=message,
                 finished_at=_now(), finished_at_epoch=time.time(), heartbeat_at=time.time())
    local_jobs._write(path, final)
    return final


# --- Progression (lecture seule des manifestes) ---------------------------------------------------------------

def run_dir_of(corpus: Path) -> Path | None:
    from core import batch
    pointer = local_jobs._read(batch.pointer_path(Path(corpus)))
    if not pointer:
        return None
    run_dir = Path(config.OUTPUTS_DIR) / pointer["run_id"]
    return run_dir if run_dir.is_dir() else None


def progress(corpus: Path) -> dict:
    corpus = Path(corpus)
    job = read_job(corpus)
    run_dir = run_dir_of(corpus)
    manifest = local_jobs._read(run_dir / "pipeline" / "pipeline_manifest.json") if run_dir else None
    lot = local_jobs._read(run_dir / "batch" / "batch_manifest.json") if run_dir else None
    manifest, lot = manifest or {}, lot or {}
    alive = bool(job and job["alive"])
    interviews = lot.get("interviews") or {}
    total = lot.get("total") or len(interviews) or len([p for p in corpus.iterdir() if p.is_file()])
    finished = [e for e in interviews.values() if e.get("final_status") in ("COMPLETE", "WARNINGS", "FAILED")]
    current = None
    if alive:
        for iid, entry in interviews.items():
            for stage, record in sorted((entry.get("stages") or {}).items()):
                if record.get("status") == "INTERRUPTED" and not record.get("finished_at"):
                    current = {"interview_id": iid, "stage": stage}
    worked = [e["duration_seconds"] for e in finished if e.get("duration_seconds")
              and not all(s.get("skipped") for s in (e.get("stages") or {}).values())]
    remaining = max(0, total - len(finished))
    eta = (sum(worked) / len(worked) * remaining) if worked and remaining and alive \
        and manifest.get("current_stage") in (None, "1-5") else None
    stages = []
    for key, label in STAGES:
        record = (manifest.get("stages") or {}).get(key) or {}
        status = record.get("status")
        if status in CONTINUABLE:
            icon = "✓"
        elif status == "RUNNING" or (alive and key == "1-5" and not manifest.get("stages")):
            icon = "⏳" if alive else "⏸"
        elif status in ("BLOCKED", "FAILED", "ABORTED"):
            icon = "✗"
        elif status == "INTERRUPTED":
            icon = "⏸"
        else:
            icon = "○"
        detail = f"{len(finished)} / {total} entretiens" if key == "1-5" else None
        stages.append({"key": key, "label": label, "icon": icon, "status": status, "detail": detail,
                       "skipped": record.get("skipped", False)})
    files = {}
    if run_dir:
        for name in ("final_report.md", "final_report.json", "delivery_manifest.json"):
            path = run_dir / "stage10" / name
            if path.is_file():
                files[name] = path
    counts = manifest.get("interview_counts") or {}
    overall = manifest.get("overall_status")
    if job and not alive and overall == "RUNNING":
        overall = "INTERRUPTED"
    return {"corpus": corpus, "job": job, "alive": alive, "run_id": manifest.get("run_id") or (run_dir.name if run_dir
                                                                                                    else None),
            "run_dir": run_dir, "overall_status": overall, "reason": manifest.get("reason"),
            "current_stage": manifest.get("current_stage"), "stages": stages, "interviews_done": len(finished),
            "interviews_total": total, "current": current,
            "elapsed_seconds": (job or {}).get("elapsed_seconds"), "active_seconds": manifest.get("active_seconds"),
            "eta_seconds": eta, "warnings": manifest.get("warnings") or [],
            "failed_interviews": manifest.get("failed_interviews") or [],
            "included": counts.get("included_stage6"), "complete": overall in FINAL_OK, "files": files,
            "can_resume": not alive and overall not in FINAL_OK}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage : python -m core.pipeline_jobs DOSSIER_DU_CORPUS", file=sys.stderr)
        return 2
    job = execute(Path(argv[0]))
    return 0 if job.get("exit_code") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
