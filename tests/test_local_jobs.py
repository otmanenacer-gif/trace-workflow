"""Exécution en arrière-plan et reprise (core/local_jobs.py) : indépendante de la page, chaque réponse validée
enregistrée immédiatement, reprise au premier appel non terminé, jamais de recalcul. Faux Ollama, aucun réseau."""

import json
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import config, local_jobs
from tests import synthetic_interviews as si
from tests import synthetic_long_interview as L
from tests.fake_llm import INTERACTION, use_fake_runtime
from tests.job_worker_fake import responders

APP = str(config.PROJECT_ROOT / "app.py")
WORKER = Path(__file__).resolve().parent / "job_worker_fake.py"


def long_run(tmp_path):
    return si.make_ingested_run(tmp_path, L.long_files())


def labels(job, *statuses):
    return [i["label"] for i in job["checklist"] if i["status"] in statuses]


def test_inline_job_runs_the_stage_and_reports_every_call(tmp_path, monkeypatch):
    use_fake_runtime(monkeypatch, responders())
    run = long_run(tmp_path)
    job = local_jobs.start_run_job(Path(run["output_dir"]), "3")
    assert job["state"] == local_jobs.COMPLETE and not job["alive"] and job["acknowledged"] is False
    assert job["result"]["status"] == "COMPLETE" and job["result"]["api_calls"] == 0
    statuses = {i["label"].split("/", 1)[1]: i["status"] for i in job["checklist"]}
    assert statuses["practice_extractor/bloc1"] == local_jobs.DONE
    assert all(s in (local_jobs.DONE, local_jobs.NOT_NEEDED) for s in statuses.values())
    assert len(job["calls"]) == len(labels(job, local_jobs.DONE)) and all(i["saved"] for i in job["checklist"]
                                                                         if i["status"] == local_jobs.DONE)
    rows = local_jobs.summary_rows(job)
    assert {"Agent", "Tokens entrée", "Tokens sortie", "Durée (s)", "Tentatives", "Contexte"} <= set(rows[0])
    assert "Practice Extractor" in (Path(run["output_dir"]) / "local_runs/job/job.log").read_text(encoding="utf-8")


def test_resume_redoes_only_the_missing_calls(tmp_path, monkeypatch):
    run_dir = Path(long_run(tmp_path)["output_dir"])
    broken = responders()
    good_interaction = broken[INTERACTION]
    broken[INTERACTION] = lambda p: ('{"signals": [' if any(t["turn_id"] == L.ltid(200) for t in L.sent_turns(p))
                                     else good_interaction(p))
    use_fake_runtime(monkeypatch, broken)
    first = local_jobs.start_run_job(run_dir, "3")
    assert first["state"] == local_jobs.FAILED
    failed = labels(first, local_jobs.CALL_FAILED)
    validated = labels(first, local_jobs.DONE)
    assert failed and len(validated) >= 6

    ollama = use_fake_runtime(monkeypatch, responders())
    second = local_jobs.start_run_job(run_dir, "3")
    assert second["state"] == local_jobs.COMPLETE and second["resumed_from"] == first["job_id"]
    redone = {c["label"] for c in second["calls"]}
    assert set(failed) <= redone and not (redone & set(validated))  # un résultat validé n'est jamais recalculé
    assert set(labels(second, local_jobs.CACHED)) >= set(validated)
    assert len(ollama.calls) == len(redone)


def test_a_job_without_heartbeat_is_interrupted_and_a_live_one_blocks_a_new_start(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    base = {"job_id": "x", "spec": {"kind": "run", "stage": "3"}, "status": local_jobs.RUNNING,
            "started_at_epoch": time.time() - 100, "checklist": []}
    (job_dir / local_jobs.JOB_FILENAME).write_text(json.dumps({**base, "heartbeat_at": time.time() - 60}))
    job = local_jobs.read_job(job_dir)
    assert job["state"] == local_jobs.INTERRUPTED and not job["alive"]
    (job_dir / local_jobs.JOB_FILENAME).write_text(json.dumps({**base, "heartbeat_at": time.time()}))
    assert local_jobs.read_job(job_dir)["alive"]
    with pytest.raises(local_jobs.JobAlreadyRunning):
        local_jobs._launch(job_dir, base["spec"])


def test_detached_process_keeps_running_without_the_page_and_resumes_without_recomputing(tmp_path, monkeypatch):
    """Vrai processus détaché : le lanceur rend la main tout de suite ; le processus avance seul, enregistre chaque
    réponse validée ; arrêté brutalement (comme un ordinateur éteint), il est repris sans rien recalculer."""
    run_dir = Path(long_run(tmp_path)["output_dir"])
    job_dir = local_jobs.run_job_dir(run_dir)
    monkeypatch.setattr(local_jobs, "worker_command", lambda d: [sys.executable, str(WORKER), str(d),
                                                                 str(config.CACHE_DIR), "0.3"])
    started = time.monotonic()
    job = local_jobs.start_run_job(run_dir, "3", mode="subprocess")
    assert time.monotonic() - started < 5 and job["state"] in (local_jobs.STARTING, local_jobs.RUNNING)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job = local_jobs.read_job(job_dir)
        saved = [i["label"] for i in job.get("checklist") or [] if i["saved"]]
        if len(saved) >= 2 and job.get("current"):
            break
        time.sleep(0.1)
    assert job["alive"] and job["state"] == local_jobs.RUNNING and job["pid"]
    assert sum(i["status"] == local_jobs.CALL_RUNNING for i in job["checklist"]) <= 1  # un appel à la fois
    assert job["current"]["agent_label"] and job["current"]["elapsed_seconds"] >= 0
    assert local_jobs.stop(job_dir)
    stopped = local_jobs.read_job(job_dir)
    assert stopped["state"] == local_jobs.STOPPED and not stopped["alive"]
    saved = [i["label"] for i in stopped["checklist"] if i["saved"]]

    ollama = use_fake_runtime(monkeypatch, responders())
    resumed = local_jobs.start_run_job(run_dir, "3")
    assert resumed["state"] == local_jobs.COMPLETE and resumed["resumed_from"] == stopped["job_id"]
    redone = {c["label"] for c in resumed["calls"]}
    assert not (redone & set(saved)) and set(saved) <= set(labels(resumed, local_jobs.CACHED))
    assert len(ollama.calls) == len(redone) > 0


# --- Interface : refresh, réouverture, reprise ------------------------------------------------------------------

def _simulate(job_dir: Path, **changes) -> None:
    path = job_dir / local_jobs.JOB_FILENAME
    job = json.loads(path.read_text(encoding="utf-8"))
    job.update(changes)
    path.write_text(json.dumps(job), encoding="utf-8")


def test_after_a_refresh_the_page_reopens_the_running_job_and_shows_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    use_fake_runtime(monkeypatch, responders())
    run = long_run(tmp_path)
    run_dir = Path(run["output_dir"])
    local_jobs.start_run_job(run_dir, "3")
    job_dir = local_jobs.run_job_dir(run_dir)
    job = json.loads((job_dir / local_jobs.JOB_FILENAME).read_text(encoding="utf-8"))
    job["checklist"][-2]["status"] = local_jobs.CALL_RUNNING  # comme pendant l'exécution du processus détaché
    _simulate(job_dir, status=local_jobs.RUNNING, heartbeat_at=time.time() + 3600, finished_at_epoch=None,
              checklist=job["checklist"], current={"label": job["checklist"][-2]["label"],
                                                   "agent_label": "Interaction Signal Reader", "attempt": 1,
                                                   "started_at": time.time() - 42, "output_tokens": 310,
                                                   "num_ctx": 20480, "estimated_input_tokens": 12000})
    at = AppTest.from_file(APP, default_timeout=30).run()  # nouvelle session : aucun run en mémoire
    assert not at.exception
    page = " ".join(str(e.value) for e in (*at.markdown, *at.info))
    assert f"run `{run['run_id']}` rouvert" in page and "Exécution locale en cours — étape 3" in page
    assert "**Interaction Signal Reader** — " in page and "310 tokens générés" in page
    assert "✓ Practice Extractor — bloc 1/" in page and "⏳ Interaction Signal Reader — bloc" in page
    assert at.button(key="run_stage3").disabled  # pas de second lancement pendant l'exécution
    assert at.query_params["run"] == [run["run_id"]] or at.query_params["run"] == run["run_id"]


def test_an_interrupted_job_is_resumed_from_the_page_without_recomputing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    run = long_run(tmp_path)
    run_dir = Path(run["output_dir"])
    broken = responders()
    good_interaction = broken[INTERACTION]
    broken[INTERACTION] = lambda p: ('{"signals": [' if any(t["turn_id"] == L.ltid(200) for t in L.sent_turns(p))
                                     else good_interaction(p))
    use_fake_runtime(monkeypatch, broken)
    first = local_jobs.start_run_job(run_dir, "3")
    validated = labels(first, local_jobs.DONE)
    job_dir = local_jobs.run_job_dir(run_dir)
    _simulate(job_dir, status=local_jobs.RUNNING, heartbeat_at=time.time() - 3600, finished_at_epoch=None)  # panne

    ollama = use_fake_runtime(monkeypatch, responders())
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["run"] = run["run_id"]  # refresh : l'adresse de la page garde le run
    at.run()
    assert not at.exception
    assert "interrompue" in " ".join(str(w.value) for w in at.warning)
    at.button(key=f"job_resume_{first['job_id']}").click().run()
    assert not at.exception
    assert "Étape 3 terminée (exécution locale, modèle qwen2.5:7b, 0 appel API)." in " ".join(
        str(s.value) for s in at.success)
    resumed = local_jobs.read_job(job_dir)
    assert resumed["state"] == local_jobs.COMPLETE and resumed["acknowledged"]
    assert not ({c["label"] for c in resumed["calls"]} & set(validated)) and ollama.calls

