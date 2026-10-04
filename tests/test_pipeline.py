"""Pipeline complet 1 → 10 sans surveillance : mini-corpus synthétique, faux Ollama derrière le vrai runner local.
Chemin critique seulement : enchaînement, reprise, entretien en échec, arrêt BLOCKED, interruption, aucune API."""

import json
from pathlib import Path

import pytest

from core import batch, config, pipeline, stage6_blocks, stage8_report
from scripts import trace_local
from tests import synthetic_stage4 as S4
from tests import synthetic_stage6 as S6
from tests.fake_llm import (COMPARATOR, PRACTICE, REPORT_VALIDATOR, REPORT_WRITER, THEORY_BLOCK, THEORY_SYNTHESIS,
                            agent_error, text_response, use_fake_runtime)
from tests.test_batch import responders as batch_responders
from tests.test_stage7_theory import responders as stage7_responders
from tests.test_stage8_report import writer
from tests.test_stage9_validation import validator

FAILING_ID = "ENTRETIEN_TROIS"
ANALYTIC = (COMPARATOR, THEORY_BLOCK, THEORY_SYNTHESIS, REPORT_WRITER, REPORT_VALIDATOR)


def comparator(params):
    """Comparateur générique : la 1re affirmation de chaque entretien qui en a (récurrence si ≥ 2, sinon minorité)."""
    material = S6.sent_material(params)
    support = []
    for entry in material["interviews"]:
        for local, item in list((entry.get("claims") or {}).items())[:1]:
            support.append({"interview_id": entry["id"], "claim_ids": [local], "criterion_ids": [],
                            "evidence_turn_ids": item.get("turns", [])})
    claim = {"claim_type": "recurring_boundary" if len(support) > 1 else "minority_configuration",
             "description": "Une frontière comparable borne l'usage de l'outil.",
             "criterion_label": None, "support": support, "counterexamples": [], "contexts": ["devoirs"],
             "n_supporting_interviews": len(support), "related_claim_numbers": [], "confidence": "medium",
             "needs_review": False}
    return text_response({"cross_case_claims": [claim], "cross_case_summary": "Synthèse simulée.", "confidence": "medium",
                          "needs_review": False, "comparator_notes": None})


def responders(fail_third=True) -> dict:
    agents = batch_responders()
    if fail_third:
        practice = agents[PRACTICE]
        agents[PRACTICE] = lambda p: agent_error() if FAILING_ID in p["messages"][0]["content"] else practice(p)
    return {**agents, COMPARATOR: comparator, **stage7_responders(), REPORT_WRITER: writer, REPORT_VALIDATOR: validator}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, data in (*S4.FILES, *S4.PLAIN_FILES):
        (directory / name).write_bytes(data)
    return directory


def add_failing(corpus):
    (corpus / f"{FAILING_ID}.txt").write_bytes(S4.PLAIN_FILES[0][1])


def manifest_of(corpus) -> dict:
    run_id = json.loads(batch.pointer_path(corpus).read_text(encoding="utf-8"))["run_id"]
    return json.loads((config.OUTPUTS_DIR / run_id / "pipeline" / "pipeline_manifest.json").read_text(encoding="utf-8"))


def analytic_calls(ollama) -> int:
    return sum(len(ollama.calls_for(agent)) for agent in ANALYTIC)


def test_full_pipeline_1_to_10_with_a_failed_interview_then_resume_recomputes_nothing(corpus, monkeypatch, capsys):
    add_failing(corpus)
    ollama = use_fake_runtime(monkeypatch, responders())
    assert trace_local.main(["pipeline", str(corpus)]) == 0
    out = capsys.readouterr().out
    for line in ("TRACE PIPELINE — run_", "[1–5/10]", "[6/10]", "[7/10]", "[8/10]", "[9/10]", "[10/10]",
                 "TRACE PIPELINE COMPLETE", "Interviews: 2 / 3 included", "Result: SUCCESS_WITH_WARNINGS",
                 "Total duration:", "Final report:", "Manifest:"):
        assert line in out
    manifest = manifest_of(corpus)
    assert manifest["overall_status"] == "SUCCESS_WITH_WARNINGS" and manifest["resume_count"] == 0
    assert [f["interview_id"] for f in manifest["failed_interviews"]] == [FAILING_ID]  # échec isolé
    assert sorted(manifest["stage5_valid"]) == sorted([S4.INTERVIEW_ID, S4.PLAIN_ID])
    assert list(manifest["stages"]) == ["1-5", "6", "7", "8", "9", "10"]
    assert manifest["stages"]["9"]["status"] == "SUCCESS_WITH_WARNINGS"  # étape 9 avec avertissements -> étape 10
    assert manifest["stages"]["10"]["status"] in pipeline.FINAL_OK
    for record in manifest["stages"].values():
        for key in ("status", "started_at", "completed_at", "duration", "result_file", "warnings"):
            assert key in record
    for key in ("run_id", "corpus_path", "started_at", "updated_at", "model", "current_stage", "overall_status",
                "interview_counts", "warnings", "errors", "output_files", "final_report"):
        assert key in manifest
    final = Path(manifest["final_report"])
    final = final if final.is_absolute() else config.PROJECT_ROOT / final
    assert manifest["final_report"] in out and final.is_file()
    assert FAILING_ID in final.read_text(encoding="utf-8")  # entretien exclu documenté dans le rapport final
    delivery = json.loads((final.parent / "delivery_manifest.json").read_text(encoding="utf-8"))
    assert delivery["api_calls"] == {"total": 0, "verified": True, "unverifiable": [], "by_source": {
        "stages_3_5": 0, "stage6": 0, "stage7": 0, "stage8": 0, "stage9": 0, "stage10": 0}}  # aucun appel API
    assert manifest["api_calls"] == 0 and analytic_calls(ollama) > 0

    # même commande : même run, étapes 6 → 10 non relancées (seul l'entretien en échec est retenté par le lot)
    ollama = use_fake_runtime(monkeypatch, responders())
    assert trace_local.main(["pipeline", str(corpus)]) == 0
    out = capsys.readouterr().out
    again = manifest_of(corpus)
    assert again["run_id"] == manifest["run_id"] and again["resume_count"] == 1 and "reprise n° 1" in out
    assert analytic_calls(ollama) == 0
    assert all(again["stages"][k]["skipped"] for k in ("6", "7", "8", "9", "10"))
    assert all(FAILING_ID in c["params"]["messages"][0]["content"] for c in ollama.calls)


def test_interruption_then_same_command_resumes_without_replaying_complete_stages(corpus, monkeypatch, capsys):
    use_fake_runtime(monkeypatch, responders(fail_third=False))
    original = stage8_report.run
    monkeypatch.setattr(stage8_report, "run", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert trace_local.main(["pipeline", str(corpus)]) == 130
    out = capsys.readouterr().out
    assert "Pipeline interrupted safely." in out and "Rerun the same command to resume." in out
    manifest = manifest_of(corpus)
    assert manifest["overall_status"] == "INTERRUPTED" and manifest["current_stage"] == "8"
    assert manifest["stages"]["8"]["status"] == "INTERRUPTED" and manifest["stages"]["7"]["status"] in pipeline.CONTINUABLE

    monkeypatch.setattr(stage8_report, "run", original)
    ollama = use_fake_runtime(monkeypatch, responders(fail_third=False))
    assert trace_local.main(["pipeline", str(corpus)]) == 0
    assert "TRACE PIPELINE COMPLETE" in capsys.readouterr().out
    again = manifest_of(corpus)
    assert again["resume_count"] == 1 and again["overall_status"] in pipeline.FINAL_OK
    assert all(again["stages"][k]["skipped"] for k in ("6", "7")) and not again["stages"]["8"]["skipped"]
    assert ollama.calls_for(COMPARATOR) == [] and ollama.calls_for(THEORY_BLOCK) == []  # rien de rejoué
    assert ollama.calls_for(REPORT_WRITER) and ollama.calls_for(PRACTICE) == []  # étapes 1-5 reprises du lot


def test_stage6_blocked_stops_cleanly_and_single_valid_interview_is_not_applicable(corpus, monkeypatch, capsys):
    use_fake_runtime(monkeypatch, responders(fail_third=False))
    monkeypatch.setattr(stage6_blocks, "run", lambda metadata, **k: {"status": "BLOCKED", "reason": "corpus inutilisable"})
    assert trace_local.main(["pipeline", str(corpus)]) == trace_local.EXIT_CODES["BLOCKED"]
    out = capsys.readouterr().out
    assert "TRACE PIPELINE STOPPED — BLOCKED" in out and "corpus inutilisable" in out and "Rerun the same command" in out
    manifest = manifest_of(corpus)
    assert manifest["overall_status"] == "BLOCKED" and manifest["stages"]["6"]["status"] == "BLOCKED"
    assert "7" not in manifest["stages"] and manifest["errors"]

    one = corpus.parent / "solo"
    one.mkdir()
    for name, data in (*S4.FILES, (f"{FAILING_ID}.txt", S4.PLAIN_FILES[0][1])):
        (one / name).write_bytes(data)
    ollama = use_fake_runtime(monkeypatch, responders())
    assert trace_local.main(["pipeline", str(one)]) == 0
    assert "NOT_APPLICABLE_SINGLE_INTERVIEW" in capsys.readouterr().out
    solo = manifest_of(one)
    assert solo["overall_status"] == "NOT_APPLICABLE_SINGLE_INTERVIEW" and "6" not in solo["stages"]
    assert ollama.calls_for(COMPARATOR) == []


def test_runtime_not_ready_at_start_stops_with_a_clear_error(corpus, monkeypatch, capsys):
    ollama = use_fake_runtime(monkeypatch, responders(), available=False)
    assert trace_local.main(["pipeline", str(corpus)]) == trace_local.EXIT_RUNTIME
    out = capsys.readouterr().out
    assert "TRACE PIPELINE STOPPED — ABORTED" in out and "runtime local non prêt" in out
    assert ollama.calls == [] and manifest_of(corpus)["overall_status"] == "ABORTED"
