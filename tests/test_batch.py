"""Lot sans surveillance (étapes 1 → 5) : corpus de deux entretiens synthétiques, faux Ollama derrière le vrai runner.
Chemin critique : lot complet, reprise sans recalcul, échec isolé (avec nouvelle tentative), Ollama injoignable."""

import json
from pathlib import Path

import pytest

from core import batch, config
from core import local_pipeline as lp
from scripts import trace_local
from tests import synthetic_stage4 as S4
from tests import synthetic_stage5 as S5
from tests.fake_llm import (ACCOUNTABILITY, AUDITOR, INTERACTION, PRACTICE, TRAJECTORY, agent_error, text_response,
                            use_fake_runtime)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, data in (*S4.FILES, *S4.PLAIN_FILES):
        (directory / name).write_bytes(data)
    (directory / "notes.xlsx").write_bytes(b"ignore")  # format non pris en charge : ignoré
    return directory


def by_interview(reference, plain):
    return lambda p: reference(p) if S4.INTERVIEW_ID in p["messages"][0]["content"] else plain(p)


def responders(practice=None) -> dict:
    ref, plain = S4.stage3_responders(), S4.plain_responders()
    return {PRACTICE: by_interview(practice or ref[PRACTICE], plain[PRACTICE]),
            INTERACTION: by_interview(ref[INTERACTION], plain[INTERACTION]),
            AUDITOR: by_interview(ref[AUDITOR], lambda p: text_response({"assessments": []})),
            ACCOUNTABILITY: by_interview(S4.REFERENCE_BUILDER, S4.scripted_builder(S4.PLAIN_RULES)),
            TRAJECTORY: by_interview(S5.scripted_mapper(S5.REFERENCE_PLAN), S5.scripted_mapper(S5.generic_plan()))}


def manifest_of(out: str) -> dict:
    line = next(l for l in out.splitlines() if l.startswith("Manifeste : "))
    return json.loads((config.PROJECT_ROOT / line.removeprefix("Manifeste : ")).read_text(encoding="utf-8")
                      if not Path(line.removeprefix("Manifeste : ")).is_absolute()
                      else Path(line.removeprefix("Manifeste : ")).read_text(encoding="utf-8"))


def contents(ollama, interview_id: str) -> int:
    return sum(interview_id in c["params"]["messages"][0]["content"] for c in ollama.calls)


def test_batch_runs_every_interview_to_stage5_then_resumes_without_recomputing(corpus, monkeypatch, capsys):
    ollama = use_fake_runtime(monkeypatch, responders())
    assert trace_local.main(["batch", str(corpus), "--until", "5"]) == 0
    out = capsys.readouterr().out
    assert "[1/2]" in out and "[2/2]" in out and "étape 5" in out and "reste ≈" in out
    manifest = manifest_of(out)
    assert manifest["state"] == "FINISHED" and manifest["total"] == 2 and manifest["api_calls"] == 0
    assert {e["final_status"] for e in manifest["interviews"].values()} <= {batch.COMPLETE, batch.WARNINGS}
    assert sorted(manifest["stage5_valid"]) == sorted([S4.INTERVIEW_ID, S4.PLAIN_ID])
    run_dir = config.OUTPUTS_DIR / manifest["run_id"]
    for info in json.loads((run_dir / config.METADATA_FILENAME).read_text(encoding="utf-8"))["files"]:
        assert lp.stage_complete("5", info)
    assert ollama.calls and batch.pointer_path(corpus).is_file()

    ollama = use_fake_runtime(monkeypatch, responders())  # reprise : tout est COMPLETE, rien n'est recalculé
    assert trace_local.main(["batch", str(corpus), "--until", "5"]) == 0
    out = capsys.readouterr().out
    assert "Reprise du lot" in out and ollama.calls == []
    again = manifest_of(out)
    assert again["run_id"] == manifest["run_id"]
    assert all(s["skipped"] for e in again["interviews"].values() for s in e["stages"].values())


def test_a_failing_interview_is_recorded_retried_once_and_the_batch_continues(corpus, monkeypatch, capsys):
    ollama = use_fake_runtime(monkeypatch, responders(practice=lambda p: agent_error()))
    assert trace_local.main(["batch", str(corpus), "--until", "5"]) == 5  # au moins un échec
    out = capsys.readouterr().out
    manifest = manifest_of(out)
    failed, plain = manifest["interviews"][S4.INTERVIEW_ID], manifest["interviews"][S4.PLAIN_ID]
    assert failed["final_status"] == batch.FAILED and failed["stages"]["3"]["attempts"] == 2
    assert failed["stages"]["3"]["error"] and "4" not in failed["stages"]
    assert plain["final_status"] in (batch.COMPLETE, batch.WARNINGS)  # l'autre entretien va jusqu'à l'étape 5
    assert manifest["stage5_valid"] == [S4.PLAIN_ID]

    ollama = use_fake_runtime(monkeypatch, responders())  # relance : seul l'entretien en échec est retraité
    assert trace_local.main(["batch", str(corpus), "--until", "5"]) == 0
    manifest = manifest_of(capsys.readouterr().out)
    assert manifest["interviews"][S4.INTERVIEW_ID]["final_status"] in (batch.COMPLETE, batch.WARNINGS)
    assert contents(ollama, S4.PLAIN_ID) == 0 and contents(ollama, S4.INTERVIEW_ID) > 0


def test_ollama_unreachable_waits_then_stops_cleanly_with_the_manifest_saved(corpus, monkeypatch, capsys):
    use_fake_runtime(monkeypatch, responders(), available=False)
    code = trace_local.main(["batch", str(corpus), "--runtime-wait", "0", "--runtime-retries", "1"])
    out = capsys.readouterr().out
    assert code == trace_local.EXIT_RUNTIME and "nouvelle tentative dans 0 s (1/1)" in out
    manifest = manifest_of(out)
    assert manifest["state"] == "ABORTED" and manifest["total"] == 1
    assert manifest["interviews"][S4.INTERVIEW_ID]["final_status"] == batch.INTERRUPTED
