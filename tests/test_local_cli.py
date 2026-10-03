"""CLI locale (scripts/trace_local.py) : ingestion, étapes 3 → 5, étape 6, état du runtime ; faux Ollama."""

import importlib.util

import pytest

from core import config
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests import synthetic_stage5 as S5
from tests import synthetic_stage6 as S6
from tests.fake_llm import ACCOUNTABILITY, COMPARATOR, TRAJECTORY, use_fake_runtime


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    spec = importlib.util.spec_from_file_location("trace_local", config.PROJECT_ROOT / "scripts" / "trace_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def agents() -> dict:
    return {**S4.stage3_responders(), ACCOUNTABILITY: S4.REFERENCE_BUILDER,
            TRAJECTORY: S5.scripted_mapper(S5.REFERENCE_PLAN)}


def test_cli_runs_stages_1_to_5_locally_then_never_replays(tmp_path, cli, capsys, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, agents())
    source = tmp_path / S4.FILENAME
    source.write_text(S4.TEXT, encoding="utf-8")
    assert cli.main(["ingest", str(source)]) == 0
    assert cli.main(["run", "latest", "--until", "5"]) == 0
    out = capsys.readouterr().out
    assert "étape 3 (exécution locale, Ollama, modèle qwen2.5:7b, 0 appel API) : COMPLETE" in out
    assert "étape 5 (exécution locale" in out and "Sorties :" in out
    calls = len(ollama.calls)
    assert cli.main(["run", "latest", "--until", "5"]) == 0
    assert "déjà terminée — non rejouée" in capsys.readouterr().out and len(ollama.calls) == calls
    assert cli.main(["status", "latest"]) == 0 and "étape 5" in capsys.readouterr().out


def test_cli_stage6_on_stage5_outputs(tmp_path, cli, capsys, monkeypatch):
    use_fake_runtime(monkeypatch, {COMPARATOR: S6.scripted_comparator(S6.GOOD_PLAN)})
    paths = S6.write_corpus(tmp_path / "stage5", S6.IDS_2)
    assert cli.main(["stage6", str(tmp_path / "stage5")]) == 0
    out = capsys.readouterr().out
    assert "étape 6 (exécution locale, Ollama" in out and "COMPLETE" in out and "ENT_A : exploitable" in out
    assert cli.main(["stage6", str(paths[0])]) == 6  # un seul entretien : bloqué


def test_cli_reports_ollama_stopped_with_a_clear_message(tmp_path, cli, capsys, monkeypatch):
    use_fake_runtime(monkeypatch, agents(), available=False)
    assert cli.main(["runtime"]) == 7
    out = capsys.readouterr().out
    assert "Exécution : locale · Runtime : Ollama" in out and "Données externes : aucune" in out
    assert "Démarrer Ollama : ollama serve" in out and "ollama pull qwen2.5:7b" in out
    source = tmp_path / si.FILENAME
    source.write_text(si.TEXT, encoding="utf-8")
    assert cli.main(["ingest", str(source)]) == 0  # l'ingestion n'a pas besoin d'Ollama
    capsys.readouterr()
    assert cli.main(["run", "latest", "--until", "3"]) == 7
    assert "Ollama ne répond pas" in capsys.readouterr().err


def test_cli_runtime_ready(cli, capsys, monkeypatch):
    use_fake_runtime(monkeypatch, {})
    assert cli.main(["runtime"]) == 0 and "Runtime local prêt." in capsys.readouterr().out


def test_cli_job_runs_stages_in_the_background_and_reports_progress(tmp_path, cli, capsys, monkeypatch):
    use_fake_runtime(monkeypatch, agents())
    source = tmp_path / S4.FILENAME
    source.write_text(S4.TEXT, encoding="utf-8")
    assert cli.main(["ingest", str(source)]) == 0
    assert cli.main(["job", "start", "latest", "--until", "5", "--wait"]) == 0  # TRACE_JOB_MODE=inline (conftest)
    out = capsys.readouterr().out
    assert "Exécution lancée en arrière-plan" in out and "étape 5 — terminée" in out
    assert "✓ Accountability Episode Builder" in out and "✓ Trajectory Mapper" in out
    assert cli.main(["job", "status", "latest"]) == 0 and "terminée" in capsys.readouterr().out
    assert cli.main(["job", "stop", "latest"]) == 0 and "Aucune exécution détachée" in capsys.readouterr().out
