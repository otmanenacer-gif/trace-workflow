"""Traitement par lots d'un corpus d'entretiens, étapes 1 → 5, sans surveillance (modèle local via Ollama, 0 API).

    python scripts/trace_local.py batch <corpus> --until 5

    corpus (dossier de fichiers .txt / .docx / .pdf, ou run existant)
      → un run TRACE : créé et ingéré au premier lancement (étapes 1-2), puis RETROUVÉ à chaque relance
        (data/outputs/batches/<empreinte du chemin>.json) : reprise après interruption ou redémarrage
      → pour chaque entretien, dans l'ordre : étapes 3, 4, 5 (local_pipeline.run_stage, une étape à la fois) :
           étape déjà COMPLETE → non rejouée (garde habituelle) ; résultats et réponses validées repris du cache ;
           SUCCESS_WITH_WARNINGS accepté ;
           étape en échec → UNE nouvelle tentative (le cache ne refait que les appels manquants), puis erreur
           enregistrée et passage à l'entretien suivant (jamais d'arrêt du lot pour un entretien) ;
           Ollama injoignable → attente puis nouvelle tentative (--runtime-wait, --runtime-retries) ;
           modèle absent ou adresse non locale → arrêt propre (manifeste enregistré, relance possible)
      → manifeste écrit après CHAQUE étape : <run>/batch/batch_manifest.json — COMPLETE / WARNINGS / FAILED / PENDING
        par entretien, étapes, statuts des agents, durées, erreurs ; à la fin, les entretiens dont l'étape 5 est valide
        alimentent l'étape 6 (`trace_local.py stage6 <run>`).

Aucun Grounding Checker, aucun benchmark, aucun audit expérimental ; prompts, schémas et validateurs inchangés.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from core import config
from core import local_pipeline as lp
from core.analysis_cache import write_json_atomic
from core.ingestion import ingest_run
from core.llm_client import RUNTIME_ERROR_CODES, LLMError
from core.run_manager import init_run, load_metadata
from core.schemas import STATUS_FAIL

BATCH_VERSION = "1.0"
BATCHES_SUBDIR = "batches"
MANIFEST_DIRNAME = "batch"
MANIFEST_FILENAME = "batch_manifest.json"
COMPLETE, WARNINGS, FAILED, PENDING, INTERRUPTED = "COMPLETE", "WARNINGS", "FAILED", "PENDING", "INTERRUPTED"
WARNING_STATUSES = {"SUCCESS_WITH_WARNINGS"}
FATAL_RUNTIME_CODES = {"MODEL_NOT_FOUND", "NON_LOCAL_URL"}  # configuration : attendre ne sert à rien
DEFAULT_STAGE_RETRIES = 1
DEFAULT_RUNTIME_WAIT = 60
DEFAULT_RUNTIME_RETRIES = 30


class BatchAborted(Exception):
    """Runtime local inutilisable (configuration) ou toujours injoignable : lot arrêté, manifeste enregistré."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


# --- Corpus → run (créé une fois, retrouvé ensuite) ------------------------------------------------------

def corpus_files(corpus: Path) -> list[Path]:
    return sorted(p for p in Path(corpus).iterdir() if p.is_file() and not p.name.startswith((".", "~"))
                  and p.suffix.lower().lstrip(".") in config.ALLOWED_EXTENSIONS)


def pointer_path(corpus: Path, outputs_dir: Path | None = None) -> Path:
    digest = hashlib.sha256(str(Path(corpus).resolve()).lower().encode("utf-8")).hexdigest()[:16]
    return Path(outputs_dir or config.OUTPUTS_DIR) / BATCHES_SUBDIR / f"{Path(corpus).resolve().name}_{digest}.json"


def open_corpus(corpus: Path, problematique: str = "", log=print) -> dict:
    """Run du corpus : retrouvé s'il existe (reprise), sinon créé et ingéré (étapes 1-2). Renvoie les métadonnées."""
    corpus = Path(corpus)
    files = corpus_files(corpus)
    if not files:
        raise ValueError(f"Aucun entretien (.txt, .docx, .pdf) dans {corpus}.")
    pointer = pointer_path(corpus)
    known = json.loads(pointer.read_text(encoding="utf-8")) if pointer.is_file() else None
    if known and (config.OUTPUTS_DIR / known["run_id"] / config.METADATA_FILENAME).is_file():
        metadata = load_metadata(config.OUTPUTS_DIR / known["run_id"])
        added = sorted({p.name for p in files} - set(known["files"]))
        log(f"Reprise du lot : run {metadata['run_id']} ({len(known['files'])} fichier(s) au départ).")
        if added:
            log(f"ATTENTION : {len(added)} fichier(s) ajouté(s) depuis le premier lancement, non traités dans ce run : "
                + ", ".join(added))
        return metadata
    log(f"Nouveau lot : {len(files)} fichier(s) — création du run et ingestion (étapes 1-2, sans IA)…")
    metadata = init_run([(p.name, p.read_bytes()) for p in files], problematique, config.INPUTS_DIR,
                        config.OUTPUTS_DIR)
    metadata = ingest_run(metadata)
    write_json_atomic(pointer, {"batch_version": BATCH_VERSION, "corpus": str(corpus.resolve()),
                                "run_id": metadata["run_id"], "files": [p.name for p in files], "created_at": _now()})
    return metadata


# --- Statuts ---------------------------------------------------------------------------------------------

def _agent_statuses(status: dict) -> dict:
    entry = next(iter(status.get("interviews") or []), {})
    return dict(entry.get("agent_statuses") or {})


def _stored_statuses(info: dict, stage: str) -> dict:
    """Statuts des agents enregistrés dans les métadonnées (étape déjà terminée, non rejouée)."""
    if stage == "3":
        agents = (info.get("analysis") or {}).get("agents") or {}
        return {name: (a or {}).get("status") for name, a in agents.items()}
    summary = info.get({"4": "accountability", "5": "trajectory"}[stage]) or {}
    return {lp.SPEC_BY_STAGE[stage].name: summary.get("status")} if summary else {}


def final_status(entry: dict, until: str) -> str:
    stages = [entry["stages"].get(s) for s in lp.STAGES[:lp.STAGES.index(until) + 1]]
    if entry.get("ingestion_error") or any(s and s["status"] == FAILED for s in stages):
        return FAILED
    if any(s is None or s["status"] != COMPLETE for s in stages):
        return INTERRUPTED if any(s and s["status"] == INTERRUPTED for s in stages) else PENDING
    warned = any(v in WARNING_STATUSES for s in stages for v in (s.get("agent_statuses") or {}).values())
    return WARNINGS if warned else COMPLETE


# --- Le lot ---------------------------------------------------------------------------------------------

class Batch:
    def __init__(self, metadata: dict, until: str = "5", *, settings=None, stage_retries: int = DEFAULT_STAGE_RETRIES,
                 runtime_wait: float = DEFAULT_RUNTIME_WAIT, runtime_retries: int = DEFAULT_RUNTIME_RETRIES,
                 log=print, sleep=time.sleep):
        self.metadata, self.until = metadata, until
        self.settings = settings or lp.runtime_settings()
        self.stage_retries, self.runtime_wait, self.runtime_retries = stage_retries, runtime_wait, runtime_retries
        self.log, self.sleep = log, sleep
        self.path = Path(metadata["output_dir"]) / MANIFEST_DIRNAME / MANIFEST_FILENAME
        previous = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.manifest = {"batch_version": BATCH_VERSION, "run_id": metadata["run_id"], "until": until,
                         "model": self.settings.model, "execution": "locale", "api_calls": 0,
                         "created_at": previous.get("created_at") or _now(), "started_at": _now(), "updated_at": None,
                         "finished_at": None, "state": "RUNNING", "interviews": previous.get("interviews") or {}}
        self.started = time.monotonic()
        self.worked: list[float] = []  # durées des entretiens réellement traités pendant cette session

    # manifeste ---------------------------------------------------------------------------------------
    def save(self) -> None:
        interviews = self.manifest["interviews"]
        for entry in interviews.values():
            entry["final_status"] = final_status(entry, self.until)
        counts = {k: sum(e["final_status"] == k for e in interviews.values())
                  for k in (COMPLETE, WARNINGS, FAILED, PENDING, INTERRUPTED)}
        self.manifest.update(updated_at=_now(), counts=counts, total=len(interviews),
                             stage5_valid=[iid for iid, e in interviews.items()
                                           if e["final_status"] in (COMPLETE, WARNINGS) and self.until == "5"])
        write_json_atomic(self.path, self.manifest)

    def _entry(self, info: dict) -> dict:
        iid = info["ingestion"]["interview_id"]
        entry = self.manifest["interviews"].setdefault(iid, {"interview_id": iid, "stages": {}})
        entry.update(file=info.get("stored_name") or info.get("original_name"))
        return entry

    # progression -------------------------------------------------------------------------------------
    def _eta(self, remaining: int) -> str:
        if not self.worked:
            return "estimation après le premier entretien traité"
        return f"reste ≈ {fmt_duration(sum(self.worked) / len(self.worked) * remaining)}"

    # une étape d'un entretien ------------------------------------------------------------------------
    def _run_stage(self, stage: str, iid: str) -> dict:
        """Une étape, avec attente si Ollama est injoignable. Lève BatchAborted (configuration ou attente épuisée)."""
        waits = 0
        while True:
            try:
                result = lp.run_stage(stage, self.metadata, [iid], settings=self.settings)
            except LLMError as error:
                if error.code not in RUNTIME_ERROR_CODES or error.code == "PROMPT_TOO_LONG":
                    raise
                if error.code in FATAL_RUNTIME_CODES or waits >= self.runtime_retries:
                    raise BatchAborted(error.user_message) from error
                waits += 1
                self.log(f"    Ollama indisponible ({error.code}) : nouvelle tentative dans {int(self.runtime_wait)} s "
                         f"({waits}/{self.runtime_retries})")
                self.sleep(self.runtime_wait)
                continue
            self.metadata = result["metadata"]
            return result["status"]

    def _interview(self, info: dict, position: int, total: int) -> None:
        iid = info["ingestion"]["interview_id"]
        entry = self._entry(info)
        if info["ingestion"]["status"] == STATUS_FAIL or not info["ingestion"].get("turn_count"):
            entry["ingestion_error"] = "ingestion en échec ou aucun tour de parole : entretien non analysable"
            self.log(f"[{position}/{total}] {iid} — ingestion en échec : ignoré")
            self.save()
            return
        started, worked = time.monotonic(), False
        for stage in lp.STAGES[:lp.STAGES.index(self.until) + 1]:
            info = next(f for f in self.metadata["files"] if f.get("ingestion", {}).get("interview_id") == iid)
            if lp.stage_complete(stage, info):
                entry["stages"][stage] = {"status": COMPLETE, "agent_statuses": _stored_statuses(info, stage),
                                          "skipped": True, "attempts": 0, "error": None,
                                          "duration_seconds": (entry["stages"].get(stage) or {}).get("duration_seconds")}
                continue
            worked = True
            stage_started = time.monotonic()
            self.log(f"[{position}/{total}] {iid} — étape {stage} en cours… (écoulé {fmt_duration(time.monotonic() - self.started)})")
            record = {"status": INTERRUPTED, "agent_statuses": {}, "skipped": False, "attempts": 0, "error": None,
                      "started_at": _now()}
            entry["stages"][stage] = record
            self.save()
            status = None
            for attempt in range(1 + self.stage_retries):
                record["attempts"] = attempt + 1
                try:
                    status = self._run_stage(stage, iid)
                except BatchAborted:
                    raise
                except Exception as exc:  # noqa: BLE001 — un entretien ne bloque jamais le lot
                    status = {"status": lp.STAGE_FAILED, "interviews": [{"errors": [
                        f"erreur inattendue {type(exc).__name__} : {exc}"]}]}
                    record["traceback"] = traceback.format_exc(limit=4)
                if status["status"] == lp.STAGE_COMPLETE:
                    break
                if attempt < self.stage_retries:
                    self.log(f"    étape {stage} : {status['status']} — nouvelle tentative (le cache ne refait que les "
                             "appels manquants)")
            interview = next(iter(status.get("interviews") or []), {})
            record.update(status=COMPLETE if status["status"] == lp.STAGE_COMPLETE else FAILED,
                          stage_status=status["status"], agent_statuses=_agent_statuses(status),
                          duration_seconds=round(time.monotonic() - stage_started, 1), finished_at=_now(),
                          error=None if status["status"] == lp.STAGE_COMPLETE else
                          " ; ".join(interview.get("errors") or []) or status.get("reason") or status["status"],
                          warnings=interview.get("warnings") or [])
            self.save()
            label = " / ".join(sorted({v for v in record["agent_statuses"].values() if v})) or status["status"]
            self.log(f"[{position}/{total}] {iid} — étape {stage} : {record['status']} ({label}) en "
                     f"{fmt_duration(record['duration_seconds'])}")
            if record["status"] == FAILED:
                self.log(f"    ÉCHEC enregistré, entretien suivant : {record['error']}")
                break
        if worked:
            self.worked.append(time.monotonic() - started)
        entry["duration_seconds"] = round(time.monotonic() - started, 1)
        self.save()
        self.log(f"[{position}/{total}] {iid} — {final_status(entry, self.until)} · écoulé "
                 f"{fmt_duration(time.monotonic() - self.started)} · {self._eta(total - position)}")

    def run(self) -> dict:
        files = [f for f in self.metadata["files"] if "ingestion" in f]
        total = len(files)
        self.log(f"Lot {self.metadata['run_id']} : {total} entretien(s), étapes 1 → {self.until}, modèle local "
                 f"{self.settings.model} (Ollama, 0 appel API)")
        try:
            for position, info in enumerate(files, start=1):
                self._interview(info, position, total)
            self.manifest.update(state="FINISHED", finished_at=_now())
        except BatchAborted as error:
            self.manifest.update(state="ABORTED", abort_reason=str(error))
            self.log(f"Lot arrêté : {error} — relancez la même commande pour reprendre.")
        except KeyboardInterrupt:
            self.manifest.update(state="INTERRUPTED")
            self.log("Lot interrompu — relancez la même commande pour reprendre (rien de validé n'est perdu).")
            self.save()
            raise
        self.save()
        return self.manifest


def summary_lines(manifest: dict) -> list[str]:
    counts = manifest.get("counts") or {}
    lines = [f"Lot {manifest['run_id']} — {manifest['state']} : " + ", ".join(
        f"{k} {v}" for k, v in counts.items() if v) + f" (sur {manifest.get('total', 0)})"]
    for iid, entry in manifest["interviews"].items():
        stages = " · ".join(f"é{s} {e['status']}" for s, e in sorted(entry["stages"].items()))
        error = next((e["error"] for e in entry["stages"].values() if e.get("error")), entry.get("ingestion_error"))
        lines.append(f"  {entry['final_status']:<11} {iid} — {stages or '—'}" + (f" — {error}" if error else ""))
    return lines
