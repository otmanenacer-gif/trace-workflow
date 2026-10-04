"""Interface Streamlit — « Analyse complète d'un corpus » (AppTest, faux Ollama, exécution en ligne : TRACE_JOB_MODE=inline).
Chemin critique : contrôles des fichiers, lancement du pipeline existant, progression et livrables, réouverture, reprise."""

import time

import pytest
from streamlit.testing.v1 import AppTest

from core import config, local_jobs, pipeline_jobs
from tests import synthetic_stage4 as S4
from tests.fake_llm import use_fake_runtime
from tests.test_pipeline import responders

APP = str(config.PROJECT_ROOT / "app.py")
FILES = [*S4.FILES, *S4.PLAIN_FILES]


def texts(elements):
    return " ".join(str(e.value) for e in elements)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    return use_fake_runtime(monkeypatch, responders(fail_third=False))


def upload(files):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.file_uploader(key="full_files").set_value([(n, d, "text/plain") for n, d in files]).run()
    return at


def launch_button(at):
    return next(b for b in at.button if b.key == "full_launch")


def downloads(at) -> list[str]:
    return [e.proto.label for e in at.get("download_button") if e.proto.label.startswith("Télécharger stage10/")]


def test_file_checks_minimum_two_and_no_duplicate_names(dirs):
    at = upload(FILES[:1])
    assert not at.exception and launch_button(at).disabled
    assert "Au moins 2 entretiens" in texts(at.error)
    at = upload([FILES[0], FILES[0]])
    assert launch_button(at).disabled and "même nom" in texts(at.error)
    at = upload([FILES[0], (FILES[0][0].replace(".txt", ".docx"), b"x")])
    assert launch_button(at).disabled and "Même identifiant d'entretien" in texts(at.error)


def test_launch_runs_the_full_pipeline_in_background_then_reopens_with_deliverables(dirs):
    at = upload(FILES)
    assert not launch_button(at).disabled
    assert {"ENTRETIEN_ETAPE_4", "ENTRETIEN_ORDINAIRE"} <= set(at.table[0].value["Entretien"])
    launch_button(at).click().run()
    assert not at.exception
    assert "TRACE COMPLETE" in texts(at.success) and "2 / 2 entretiens inclus" in texts(at.success)
    assert downloads(at) == [f"Télécharger stage10/{n}" for n in ("final_report.md", "final_report.json",
                                                                 "delivery_manifest.json")]
    assert "✓ Étapes 1–5 — 2 / 2 entretiens" in texts(at.markdown) and "✓ Étape 10 — livraison" in texts(at.markdown)
    corpus = pipeline_jobs.list_corpora()[0]
    assert sorted(p.name for p in corpus.iterdir() if p.is_file()) == sorted(n for n, _ in FILES)  # noms conservés

    again = AppTest.from_file(APP, default_timeout=60)  # plus tard : la page rouvre le corpus et sa progression
    again.query_params["corpus"] = corpus.name
    again.run()
    assert "TRACE COMPLETE" in texts(again.success) and len(downloads(again)) == 3
    assert dirs.calls and not [b for b in again.button if b.key == "full_resume"]


def test_interrupted_run_shows_resume_which_relaunches_the_same_corpus(dirs):
    corpus = pipeline_jobs.create_corpus(FILES, "nuit")
    path = pipeline_jobs.job_dir(corpus) / local_jobs.JOB_FILENAME
    local_jobs._write(path, {"corpus": str(corpus), "args": pipeline_jobs.command(corpus), "status": "running",
                             "started_at_epoch": time.time() - 600, "heartbeat_at": time.time() - 300})
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["corpus"] = corpus.name
    at.run()
    assert not at.exception and "○ Étape 6 — comparaison" in texts(at.markdown)
    resume = next(b for b in at.button if b.key == "full_resume")
    resume.click().run()
    assert "TRACE COMPLETE" in texts(at.success)
    assert pipeline_jobs.read_job(corpus)["launch_count"] == 1 and pipeline_jobs.list_corpora() == [corpus]
