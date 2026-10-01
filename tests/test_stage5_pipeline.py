"""Étape 5 — orchestration : blocages, étape 4 périmée ou partielle, cache, coût, échecs. LLM simulé."""

import dataclasses
import json

import pytest

from core import accountability, config, trajectory
from core import trajectory_candidates as tc
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from core.llm_client import LLMError, LLMSettings
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests import synthetic_stage5 as S5
from tests.fake_llm import ACCOUNTABILITY, TRAJECTORY, FakeTransport, fake_settings, server_error

CASE = S5.CONTEXT


def files_written(run) -> set[str]:
    return {p.name for p in S5.analysis_dir(run).glob("student_trajectory*.json")}


# --- Blocages --------------------------------------------------------------------------------------------

def test_blocked_without_stage4(tmp_path):
    run = si.make_ingested_run(tmp_path, CASE.files)
    cache = AnalysisCache(tmp_path / "cache")
    run = analyze_run(run, settings=fake_settings(), transport=FakeTransport(CASE.stage3_responders()), cache=cache)
    assert trajectory.plan_stage5(run, [CASE.interview_id], fake_settings(), cache)["blocked"] == 1
    run, transport = S5.run_stage5(run, cache, CASE.mapper())
    assert transport.calls == []
    summary = run["files"][0]["trajectory"]
    assert summary["status"] == "BLOCKED" and summary["stage4_status"] == "NOT_RUN"
    assert summary["error"]["code"] == "STAGE4_UNAVAILABLE"
    assert files_written(run) == {config.STUDENT_TRAJECTORY_MANIFEST_FILENAME, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME}


def test_blocked_when_stage4_failed(tmp_path):
    run, cache = S5.run_to_stage4(tmp_path, CASE.files, CASE.stage3_responders(), server_error(500))
    assert run["files"][0]["accountability"]["status"] == "FAILED"
    run, transport = S5.run_stage5(run, cache, CASE.mapper())
    assert transport.calls == [] and run["files"][0]["trajectory"]["stage4_status"] == "FAILED"
    assert config.STUDENT_TRAJECTORY_FILENAME not in files_written(run)


def test_stale_stage4_blocks_and_removes_the_previous_trajectory(tmp_path):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    run, _ = S5.run_stage5(run, cache, CASE.mapper())
    assert run["files"][0]["trajectory"]["status"] == "SUCCESS"
    summary = run["files"][0]["trajectory"]
    assert trajectory.trajectory_current(summary)
    # l'étape 3 est remplacée après l'étape 4 (autres pratiques) : l'étape 4 ne la décrit plus
    practices = S5.analysis_dir(run) / "practice_extractor.json"
    practices.write_text(practices.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not trajectory.trajectory_current(summary)
    state = trajectory.stage4_state(S5.analysis_dir(run).parent)
    assert state["status"] == "STALE" and "practice_extractor_sha256" in state["reasons"][0]
    run, transport = S5.run_stage5(run, cache, CASE.mapper())
    assert transport.calls == [] and run["files"][0]["trajectory"]["status"] == "BLOCKED"
    assert config.STUDENT_TRAJECTORY_FILENAME not in files_written(run)


def test_partial_stage4_gives_a_partial_stage5(tmp_path):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    path = S5.analysis_dir(run) / config.ACCOUNTABILITY_EPISODES_FILENAME
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(status="PARTIAL", analysis_complete=False)
    path.write_text(json.dumps(document), encoding="utf-8")
    run, transport = S5.run_stage5(run, cache, CASE.mapper())
    output = S5.load(run, config.STUDENT_TRAJECTORY_FILENAME)
    assert len(transport.calls) == 1
    assert output["status"] == "PARTIAL" and output["analysis_complete"] is False and output["needs_review"] is True
    validation = S5.load(run, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME)
    assert "STAGE4_INCOMPLETE" in {i["code"] for i in validation["issues"]}


# --- Cache et coût -------------------------------------------------------------------------------------------

def test_same_inputs_reuse_the_cache_with_zero_call(tmp_path):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    plan = trajectory.plan_stage5(run, [CASE.interview_id], fake_settings(), cache)
    assert (plan["calls"], plan["cached"]) == (1, 0)
    run, first = S5.run_stage5(run, cache, CASE.mapper())
    before = S5.load(run, config.STUDENT_TRAJECTORY_FILENAME)
    assert len(first.calls) == 1 and before["status"] == "SUCCESS"
    plan = trajectory.plan_stage5(run, [CASE.interview_id], fake_settings(), cache)
    assert (plan["calls"], plan["cached"]) == (0, 1)
    run, second = S5.run_stage5(run, cache, CASE.mapper())
    after = S5.load(run, config.STUDENT_TRAJECTORY_FILENAME)
    assert second.calls == [] and after["status"] == "CACHED" and after["cache_hit"] is True
    assert after["trajectory_claims"] == before["trajectory_claims"]
    run, forced = S5.run_stage5(run, cache, CASE.mapper(), force=True)
    assert len(forced.calls) == 1


def test_changing_stage5_invalidates_only_stage5(tmp_path, monkeypatch):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    run, _ = S5.run_stage5(run, cache, CASE.mapper())
    monkeypatch.setattr(trajectory, "SPEC", dataclasses.replace(trajectory.SPEC, version="9.9"))
    assert trajectory.plan_stage5(run, [CASE.interview_id], fake_settings(), cache)["calls"] == 1
    # étapes 3 et 4 : toujours en cache, jamais recalculées
    assert accountability.plan_stage4(run, [CASE.interview_id], fake_settings(), cache)["calls"] == 0
    run, transport = S5.run_stage5(run, cache, CASE.mapper())
    assert [c["agent"] for c in transport.calls] == [TRAJECTORY]


def test_rerunning_stage4_with_identical_episodes_keeps_the_stage5_cache(tmp_path):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    run, _ = S5.run_stage5(run, cache, CASE.mapper())
    run = accountability.analyze_run_stage4(run, settings=fake_settings(), transport=FakeTransport(
        {ACCOUNTABILITY: CASE.builder()}), cache=cache, force=True)
    run, transport = S5.run_stage5(run, cache, CASE.mapper())
    assert transport.calls == [] and run["files"][0]["trajectory"]["status"] == "CACHED"


def test_insufficient_material_costs_nothing(tmp_path):
    files = [si.other_interview("Entretien_court.txt", "Je lui demande des définitions.")]
    responders = {**CASE.stage3_responders()}
    from tests.fake_llm import PRACTICE, text_response
    short_id = "ENTRETIEN_COURT"
    responders[PRACTICE] = lambda p: text_response({"practices": [si.practice(
        turn_start=f"{short_id}_T0001", turn_end=f"{short_id}_T0002", academic_task="définitions",
        evidence=[{"turn_id": f"{short_id}_T0002", "quote": "Je lui demande des définitions."}])],
        "extraction_notes": None})
    responders[S5.INTERACTION] = lambda p: text_response({"signals": [], "reading_notes": None})
    run, cache = S5.run_to_stage4(tmp_path, files, responders, CASE.builder())
    assert trajectory.plan_stage5(run, [short_id], fake_settings(), cache)["no_material"] == 1
    run, transport = S5.run_stage5(run, cache, CASE.mapper())
    output = S5.load(run, config.STUDENT_TRAJECTORY_FILENAME)
    assert transport.calls == [] and output["llm_called"] is False and output["model"] is None
    assert output["configuration_type"] == "no_clear_pattern" and output["status"] == "SUCCESS"
    assert run["last_trajectory"]["usage"]["no_material"] == 1


def test_llm_failure_is_isolated_and_writes_no_trajectory(tmp_path):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    run, transport = S5.run_stage5(run, cache, server_error(500))
    summary = run["files"][0]["trajectory"]
    assert summary["status"] == "FAILED" and summary["api_calls"] == 3  # 1 appel + 2 réessais bornés, jamais plus
    assert config.STUDENT_TRAJECTORY_FILENAME not in files_written(run)
    assert "échec(s)" in run["pipeline"][config.TRAJECTORY_STEP]


def test_one_call_per_interview_and_no_cross_interview_material(tmp_path):
    run = si.make_ingested_run(tmp_path, [*S5.CONTEXT.files, *S5.STABLE.files])
    cache = AnalysisCache(tmp_path / "cache")

    def by_interview(agent_responders):
        def respond(params):
            content = params["messages"][0]["content"]
            case = S5.CONTEXT if S5.CONTEXT.interview_id in content else S5.STABLE
            return agent_responders(case)(params)
        return respond

    from tests.fake_llm import INTERACTION, PRACTICE
    stage3 = {PRACTICE: by_interview(lambda c: c.stage3_responders()[PRACTICE]),
              INTERACTION: by_interview(lambda c: c.stage3_responders()[INTERACTION])}
    run = analyze_run(run, settings=fake_settings(), transport=FakeTransport(stage3), cache=cache)
    run = accountability.analyze_run_stage4(run, settings=fake_settings(), transport=FakeTransport(
        {ACCOUNTABILITY: by_interview(lambda c: c.builder())}), cache=cache)
    run, transport = S5.run_stage5(run, cache, by_interview(lambda c: c.mapper()))
    assert len(transport.calls) == 2
    for call in transport.calls:
        content = call["params"]["messages"][0]["content"]
        assert (S5.CONTEXT.interview_id in content) != (S5.STABLE.interview_id in content)
    assert [f["trajectory"]["status"] for f in run["files"]] == ["SUCCESS", "SUCCESS"]
    assert run["pipeline"][config.TRAJECTORY_STEP] == "terminé (2/2 entretien(s))"
    assert run["last_trajectory"]["usage"]["api_calls"] == 2


def test_over_threshold_is_flagged_but_still_one_call_and_same_max_tokens(tmp_path, monkeypatch):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    monkeypatch.setattr(tc, "SINGLE_CALL_MAX_INPUT_TOKENS", 50)
    settings = fake_settings()
    run, transport = S5.run_stage5(run, cache, CASE.mapper(), settings=settings)
    assert len(transport.calls) == 1 and transport.calls[0]["params"]["max_tokens"] == settings.max_tokens
    summary = run["files"][0]["trajectory"]
    assert summary["over_single_call_threshold"] is True and summary["status"] == "SUCCESS_WITH_WARNINGS"
    validation = S5.load(run, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME)
    assert "PAYLOAD_OVER_THRESHOLD" in {i["code"] for i in validation["issues"]}


def test_not_configured_raises(tmp_path):
    run, cache = S5.case_to_stage4(tmp_path, CASE)
    with pytest.raises(LLMError) as info:
        trajectory.analyze_run_stage5(run, settings=LLMSettings(), cache=cache)
    assert info.value.code == "NOT_CONFIGURED"


# --- Sorties ---------------------------------------------------------------------------------------------------

def test_output_documents(tmp_path):
    run, cache = S5.run_to_stage4(tmp_path, S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    run, transport = S5.run_stage5(run, cache, S5.scripted_mapper(S5.REFERENCE_PLAN))
    document = S5.load(run, config.STUDENT_TRAJECTORY_FILENAME)
    manifest = S5.load(run, config.STUDENT_TRAJECTORY_MANIFEST_FILENAME)
    validation = S5.load(run, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME)
    for key in ("interview_id", "configuration_type", "trajectory_claims", "stable_boundaries", "contextual_variations",
                "explicit_temporal_changes", "exceptions", "unresolved_tensions", "ordinary_zones",
                "student_role_criteria", "trajectory_summary", "confidence", "needs_review"):
        assert key in document
    assert all(isinstance(i, str) for key in trajectory.validator.LIST_KEYS.values() for i in document[key])
    assert document["agent"] == "trajectory_mapper" and document["agent_version"] == "1.0"
    assert document["source_hashes"]["accountability_episodes_sha256"] and document["stage4_status"] == "COMPLETE"
    # épisode incertain de l'étape 4 (tour mal attribué) : utilisable, à revoir, propagé dans la représentation
    sent = S5.sent_material(transport.calls[0]["params"])
    assert any(e.get("speaker_warning") for e in sent["episodes"])
    assert manifest["api_calls"] == 1 and manifest["llm_called"] is True and manifest["cache_key"]
    assert manifest["payload_sha256"] and manifest["estimated_input_tokens"] < tc.SINGLE_CALL_MAX_INPUT_TOKENS
    assert validation["status"] == document["status"] == "SUCCESS" and validation["error_count"] == 0
    assert run["files"][0]["trajectory"]["configuration_type"] == "contextual_configuration"
    assert run["pipeline"][config.TRAJECTORY_STEP] == "terminé (1/1 entretien(s))"
