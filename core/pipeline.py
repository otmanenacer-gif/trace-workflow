"""Pipeline TRACE complet, sans surveillance : un dossier de corpus → rapport final (étapes 1 → 10). Orchestration seule.

    python scripts/trace_local.py pipeline <corpus>

    1. runtime local vérifié au démarrage (Ollama répond, modèle installé ; jamais démarré par TRACE)
    2. étapes 1 → 5 : le LOT existant (core/batch.py) — run retrouvé par le lien corpus → run, étapes COMPLETE non
       rejouées, une nouvelle tentative par étape, entretien en échec isolé, attente si Ollama est injoignable
       → stage5_valid ≥ 2 : suite ; 1 : arrêt propre (NOT_APPLICABLE_SINGLE_INTERVIEW) ; 0 : BLOCKED
    3. étapes 6, 7, 8, 9, 10 : les fonctions existantes (stage6_blocks, stage7_theory, stage8_report,
       stage9_validation, stage10_delivery), chacune avec son cache et sa reprise ; une étape déjà terminée ET à jour
       (étape 6 : mêmes entretiens que stage5_valid ; étapes 7 à 10 : sortie plus récente que celle de l'étape
       précédente) n'est pas relancée ; FAILED : UNE nouvelle tentative (le cache ne refait que le manquant), puis
       arrêt ; BLOCKED : arrêt ; COMPLETE / SUCCESS / SUCCESS_WITH_WARNINGS : suite
    → <run>/pipeline/pipeline_manifest.json réécrit à chaque transition ; Ctrl+C : manifeste enregistré, relancer
      la même commande reprend le même run.

Aucun agent, prompt, schéma ou validateur nouveau ; aucune logique analytique ; modèle local via Ollama, 0 API.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from core import batch as batch_mod
from core import local_pipeline as lp
from core import stage6_blocks, stage7_theory, stage8_report, stage9_validation, stage10_delivery
from core.analysis_cache import write_json_atomic
from core.llm_client import RUNTIME_ERROR_CODES, LLMError
from core.run_manager import load_metadata

PIPELINE_VERSION = "1.0"
MANIFEST_DIRNAME = "pipeline"
MANIFEST_FILENAME = "pipeline_manifest.json"
COMPLETE, SUCCESS_WITH_WARNINGS = "COMPLETE", "SUCCESS_WITH_WARNINGS"
BLOCKED, FAILED, NOT_APPLICABLE = lp.STAGE_BLOCKED, lp.STAGE_FAILED, lp.STAGE_NOT_APPLICABLE
ABORTED, INTERRUPTED, RUNNING = "ABORTED", "INTERRUPTED", "RUNNING"
CONTINUABLE = {"COMPLETE", "SUCCESS", "SUCCESS_WITH_WARNINGS"}
FINAL_OK = (COMPLETE, SUCCESS_WITH_WARNINGS)
TITLES = {"1-5": "[1–5/10] Batch interviews (ingestion, stages 3 → 5)", "6": "[6/10] Cross-interview comparison",
          "7": "[7/10] Theory", "8": "[8/10] Report draft", "9": "[9/10] Validation", "10": "[10/10] Delivery"}


class PipelineStop(Exception):
    """Arrêt propre : statut et raison enregistrés dans le manifeste."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def manifest_path(metadata: dict) -> Path:
    return Path(metadata["output_dir"]) / MANIFEST_DIRNAME / MANIFEST_FILENAME


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- Étapes 6 à 10 : sortie, statut, fraîcheur (lecture seule des sorties existantes) -------------------------

def _stage_specs(metadata: dict) -> list[dict]:
    s9 = stage9_validation.paths(metadata)
    s10 = stage10_delivery.paths(metadata)
    return [
        {"key": "6", "path": stage6_blocks.out_path(metadata), "upstream": None,
         "run": lambda m, s, log: stage6_blocks.run(m, settings=s, log=log),
         "status": lambda d: d.get("status"),
         "warned": lambda d: bool(d.get("needs_review") or d.get("excluded_interviews"))},
        {"key": "7", "path": stage7_theory.out_path(metadata), "upstream": stage6_blocks.out_path(metadata),
         "run": lambda m, s, log: stage7_theory.run(m, settings=s, log=log),
         "status": lambda d: d.get("status"), "warned": lambda d: bool(d.get("needs_review"))},
        {"key": "8", "path": stage8_report.paths(metadata)[0], "upstream": stage7_theory.out_path(metadata),
         "run": lambda m, s, log: stage8_report.run(m, settings=s, log=log),
         "status": lambda d: d.get("status"), "warned": lambda d: d.get("result_status") == SUCCESS_WITH_WARNINGS},
        {"key": "9", "path": s9["validation"], "upstream": stage8_report.paths(metadata)[0],
         "run": lambda m, s, log: stage9_validation.run(m, settings=s, log=log),
         "status": lambda d: d.get("validation_outcome") or d.get("status"),
         "warned": lambda d: d.get("validation_outcome") == SUCCESS_WITH_WARNINGS},
        {"key": "10", "path": s10["manifest"], "upstream": s9["validation"],
         "run": lambda m, s, log: stage10_delivery.run(m, log=log),
         "status": lambda d: d.get("status"), "warned": lambda d: d.get("status") == SUCCESS_WITH_WARNINGS},
    ]


def _stage6_fresh(metadata: dict, document: dict) -> bool:
    """L'étape 6 porte exactement sur les entretiens retenus aujourd'hui (stage5_valid du lot)."""
    chosen = set(document.get("interview_ids") or []) | {e["interview_id"] for e in document.get("excluded_interviews", [])
                                                         if e.get("source") == "corpus_check"}
    return chosen == set(stage6_blocks.selection(metadata)["interview_ids"])


def _fresh(spec: dict, metadata: dict, document: dict) -> bool:
    if spec["key"] == "6":
        return _stage6_fresh(metadata, document)
    upstream = spec["upstream"]
    return upstream.is_file() and spec["path"].stat().st_mtime >= upstream.stat().st_mtime


# --- Le pipeline ------------------------------------------------------------------------------------------------

class Pipeline:
    def __init__(self, metadata: dict, corpus: Path | str, *, settings=None, stage_retries: int = 1,
                 runtime_wait: float = batch_mod.DEFAULT_RUNTIME_WAIT,
                 runtime_retries: int = batch_mod.DEFAULT_RUNTIME_RETRIES, log=print, sleep=time.sleep):
        self.metadata, self.corpus = metadata, str(corpus)
        self.settings = settings or lp.runtime_settings()
        self.stage_retries, self.runtime_wait, self.runtime_retries = stage_retries, runtime_wait, runtime_retries
        self.log, self.sleep = log, sleep
        self.path = manifest_path(metadata)
        self.session_started = time.monotonic()
        previous = _read(self.path) or {}
        self.manifest = {
            "pipeline_version": PIPELINE_VERSION, "run_id": metadata["run_id"], "corpus_path": self.corpus,
            "started_at": previous.get("started_at") or _now(), "updated_at": None, "finished_at": None,
            "model": self.settings.model, "execution": "locale", "api_calls": 0, "current_stage": None,
            "overall_status": RUNNING, "reason": None, "stages": previous.get("stages") or {},
            "interview_counts": previous.get("interview_counts") or {}, "stage5_valid": previous.get("stage5_valid") or [],
            "failed_interviews": previous.get("failed_interviews") or [], "warnings": [],
            "errors": previous.get("errors") or [],
            "resume_count": previous.get("resume_count", -1) + 1 if previous else 0,
            "active_seconds_before": previous.get("active_seconds") or 0, "active_seconds": previous.get("active_seconds") or 0,
            "output_files": previous.get("output_files") or {}, "final_report": previous.get("final_report")}

    # manifeste ---------------------------------------------------------------------------------------------
    def save(self) -> None:
        self.manifest["updated_at"] = _now()
        self.manifest["active_seconds"] = round(self.manifest["active_seconds_before"]
                                                + time.monotonic() - self.session_started, 1)
        write_json_atomic(self.path, self.manifest)

    def _begin(self, key: str) -> dict:
        self.manifest["current_stage"] = key
        record = {**(self.manifest["stages"].get(key) or {}), "key": key, "status": RUNNING, "started_at": _now(),
                  "completed_at": None, "skipped": False}
        self.manifest["stages"][key] = record
        self.save()
        self.log("")
        self.log(TITLES[key])
        return record

    def _end(self, record: dict, status: str, started: float, result_file: Path | None, warnings: list[str],
             **extra) -> None:
        record.update(status=status, completed_at=_now(), duration=round(time.monotonic() - started, 1),
                      result_file=lp.display_path(result_file) if result_file else None, warnings=warnings, **extra)
        if result_file:
            self.manifest["output_files"][f"stage{record['key']}"] = record["result_file"]
        self.save()

    def _error(self, stage: str, message: str) -> None:
        self.manifest["errors"].append({"stage": stage, "at": _now(), "message": message})

    # attente si Ollama est injoignable (même politique que le lot) ------------------------------------------
    def _with_runtime_wait(self, key: str, call):
        waits = 0
        while True:
            try:
                return call()
            except LLMError as error:
                if error.code not in RUNTIME_ERROR_CODES or error.code == "PROMPT_TOO_LONG":
                    raise
                if error.code in batch_mod.FATAL_RUNTIME_CODES or waits >= self.runtime_retries:
                    raise PipelineStop(ABORTED, f"runtime local inutilisable à l'étape {key} : {error.user_message}")
                waits += 1
                self.log(f"    Ollama indisponible ({error.code}) : nouvelle tentative dans {int(self.runtime_wait)} s "
                         f"({waits}/{self.runtime_retries})")
                self.sleep(self.runtime_wait)

    # étapes 1 → 5 --------------------------------------------------------------------------------------------
    def _batch(self) -> None:
        record = self._begin("1-5")
        started = time.monotonic()
        runner = batch_mod.Batch(self.metadata, "5", settings=self.settings, stage_retries=self.stage_retries,
                                 runtime_wait=self.runtime_wait, runtime_retries=self.runtime_retries, log=self.log,
                                 sleep=self.sleep)
        manifest = runner.run()  # KeyboardInterrupt : manifeste du lot enregistré, propagé
        self.metadata = load_metadata(Path(self.metadata["output_dir"]))
        valid = list(manifest.get("stage5_valid") or [])
        failed = [{"interview_id": iid, "error": next((s.get("error") for s in (e.get("stages") or {}).values()
                                                       if s.get("error")), e.get("ingestion_error"))}
                  for iid, e in manifest["interviews"].items() if e.get("final_status") == batch_mod.FAILED]
        warnings = [f"entretien en échec (exclu) : {f['interview_id']} — {f['error']}" for f in failed]
        self.manifest.update(stage5_valid=valid, failed_interviews=failed,
                             interview_counts={"total": manifest.get("total"), "stage5_valid": len(valid),
                                               "failed": len(failed), **(manifest.get("counts") or {})})
        if manifest["state"] == "ABORTED":
            self._end(record, ABORTED, started, runner.path, warnings)
            raise PipelineStop(ABORTED, f"lot arrêté : {manifest.get('abort_reason')}")
        status = SUCCESS_WITH_WARNINGS if failed or manifest["counts"].get(batch_mod.WARNINGS) else COMPLETE
        if not valid:
            self._end(record, BLOCKED, started, runner.path, warnings)
            raise PipelineStop(BLOCKED, "aucun entretien valide à l'étape 5 : analyse inter-entretiens impossible")
        if len(valid) == 1:
            self._end(record, status, started, runner.path, warnings)
            raise PipelineStop(NOT_APPLICABLE, f"un seul entretien valide à l'étape 5 ({valid[0]}) : "
                                               "analyse inter-entretiens impossible (étapes 6 à 10 non lancées)")
        self._end(record, status, started, runner.path, warnings)
        self.log(f"Étapes 1-5 : {len(valid)} entretien(s) valide(s) sur {manifest.get('total')}"
                 + (f", {len(failed)} en échec (exclus)" if failed else ""))

    # étapes 6 → 10 -------------------------------------------------------------------------------------------
    def _stage(self, spec: dict) -> None:
        key = spec["key"]
        document = _read(spec["path"])
        if document is not None and spec["status"](document) in CONTINUABLE and _fresh(spec, self.metadata, document):
            status = SUCCESS_WITH_WARNINGS if spec["warned"](document) else spec["status"](document)
            previous = self.manifest["stages"].get(key) or {}
            self.manifest["stages"][key] = {**previous, "key": key, "status": status, "skipped": True,
                                            "result_file": lp.display_path(spec["path"]),
                                            "started_at": previous.get("started_at"),
                                            "completed_at": previous.get("completed_at"),
                                            "duration": previous.get("duration"), "warnings": previous.get("warnings", [])}
            self.manifest["output_files"][f"stage{key}"] = lp.display_path(spec["path"])
            self.save()
            self.log(f"{TITLES[key]} — déjà terminée et à jour ({status}) : non relancée")
            return
        record = self._begin(key)
        started = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            document = self._with_runtime_wait(key, lambda: spec["run"](self.metadata, self.settings, self.log))
            raw = spec["status"](document)
            if raw != FAILED or attempts > self.stage_retries:
                break
            self.log(f"    étape {key} : FAILED ({document.get('reason')}) — nouvelle tentative (le cache ne refait que "
                     "le manquant)")
        warnings = [document["reason"]] if document.get("reason") and raw in CONTINUABLE else []
        if raw in CONTINUABLE:
            status = SUCCESS_WITH_WARNINGS if spec["warned"](document) else raw
            self._end(record, status, started, spec["path"], warnings, attempts=attempts)
            self.log(f"{TITLES[key]} — {status}")
            return
        reason = f"étape {key} {raw}" + (f" — {document['reason']}" if document.get("reason") else "")
        self._end(record, raw or FAILED, started, spec["path"], warnings, attempts=attempts)
        self._error(key, reason)
        raise PipelineStop(NOT_APPLICABLE if raw == NOT_APPLICABLE else BLOCKED if raw == BLOCKED else FAILED, reason)

    # exécution -----------------------------------------------------------------------------------------------
    def run(self) -> dict:
        self.log(f"TRACE PIPELINE — {self.metadata['run_id']}"
                 + (f" (reprise n° {self.manifest['resume_count']})" if self.manifest["resume_count"] else ""))
        self.log(f"Corpus : {self.corpus} · modèle local {self.settings.model} (Ollama, 0 appel API)")
        self.manifest.update(overall_status=RUNNING, reason=None, finished_at=None, warnings=[])
        self.save()
        try:
            runtime = lp.runtime_status(self.settings)
            if not runtime["ready"]:
                error = runtime.get("error") or {}
                hint = runtime.get("start_command") if not runtime["available"] else runtime.get("install_command")
                raise PipelineStop(ABORTED, f"runtime local non prêt ({error.get('code')}) : {error.get('message')}"
                                            + (f" — {hint}" if hint else ""))
            self._batch()
            for spec in _stage_specs(self.metadata):
                self._stage(spec)
            self._finish()
        except PipelineStop as stop:
            self.manifest.update(overall_status=stop.status, reason=stop.reason)
            if stop.status != NOT_APPLICABLE:
                self._error(self.manifest.get("current_stage") or "start", stop.reason)
            self._collect_warnings()
            self.save()
        except KeyboardInterrupt:
            current = self.manifest["stages"].get(self.manifest.get("current_stage") or "")
            if current and current.get("status") == RUNNING:
                current["status"] = INTERRUPTED
            self.manifest.update(overall_status=INTERRUPTED, reason="interrompu par l'utilisateur (Ctrl+C)")
            self.save()
            raise
        return self.manifest

    def _finish(self) -> None:
        delivery = _read(stage10_delivery.paths(self.metadata)["manifest"]) or {}
        final = stage10_delivery.paths(self.metadata)["md"]
        counts = delivery.get("interview_counts") or {}
        self.manifest["interview_counts"].update(included_stage6=counts.get("included_stage6"),
                                                 excluded=counts.get("excluded"))
        self.manifest.update(overall_status=delivery.get("status"), current_stage=None, finished_at=_now(),
                             final_report=lp.display_path(final),
                             api_calls=(delivery.get("api_calls") or {}).get("total", 0),
                             api_calls_verified=(delivery.get("api_calls") or {}).get("verified"))
        self.manifest["output_files"].update(final_report_md=lp.display_path(final),
                                             final_report_json=lp.display_path(stage10_delivery.paths(self.metadata)["json"]),
                                             delivery_manifest=lp.display_path(stage10_delivery.paths(self.metadata)["manifest"]))
        self._collect_warnings(delivery)
        self.save()

    def _collect_warnings(self, delivery: dict | None = None) -> None:
        warnings = [w for s in self.manifest["stages"].values() for w in (s.get("warnings") or [])]
        if delivery and delivery.get("unresolved_warnings"):
            warnings.append(f"{len(delivery['unresolved_warnings'])} avertissement(s) non résolu(s) de l'étape 9 : voir "
                            "le rapport final (annexe)")
        self.manifest["warnings"] = list(dict.fromkeys(warnings))


def format_duration(seconds: float | None) -> str:
    seconds = int(round(seconds or 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
