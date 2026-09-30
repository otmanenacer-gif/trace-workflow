"""Tests du cache déterministe des analyses : aucun appel repayé à l'identique, et seulement à l'identique."""

import dataclasses
import json
from pathlib import Path

import pytest

import core.analysis as analysis
from agents import interaction_signal_reader, practice_extractor
from core.analysis import analyze_run, plan_analysis
from core.analysis_cache import KEY_FIELDS, AnalysisCache, compute_cache_key
from tests import synthetic_interviews as si
from tests.fake_llm import INTERACTION, PRACTICE, FakeTransport, fake_settings, text_response


def responders():
    return {PRACTICE: lambda p: text_response(si.GOOD_PRACTICES), INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)}


@pytest.fixture
def env(tmp_path):
    run = si.make_ingested_run(tmp_path)
    return {"run": run, "cache": AnalysisCache(tmp_path / "cache")}


def analyze(env, settings=None, **kwargs):
    transport = FakeTransport(responders())
    run = analyze_run(env["run"], settings=settings or fake_settings(), transport=transport, cache=env["cache"],
                      **kwargs)
    return run, transport


def manifest(run, spec):
    return json.loads((Path(run["files"][0]["analysis"]["analysis_dir"]) / spec.manifest_filename).read_text())


def test_identical_inputs_hit_the_cache(env):
    run, first = analyze(env)
    assert len(first.calls) == 2
    assert plan_analysis(run, [si.INTERVIEW_ID], fake_settings(), env["cache"]) == {
        "interviews": 1, "calls": 0, "cached": 2}
    run, second = analyze(env)
    assert second.calls == []  # aucun appel API
    for spec in analysis.AGENTS:
        m = manifest(run, spec)
        assert m["cache_hit"] is True and m["status"] == "CACHED" and m["api_calls"] == 0
        assert m["billed_this_run"] is False and m["usage"]["input_tokens"] == 1200  # coût d'origine, pour mémoire
    usage = run["last_analysis"]["usage"]
    assert usage["api_calls"] == 0 and usage["input_tokens"] == 0 and usage["cached_results"] == 2


def test_cached_results_are_revalidated_and_identical(env):
    run, _ = analyze(env)
    path = Path(run["files"][0]["analysis"]["analysis_dir"]) / practice_extractor.SPEC.output_filename
    fresh = json.loads(path.read_text())
    run, _ = analyze(env)
    cached = json.loads(path.read_text())
    assert cached["practices"] == fresh["practices"] and cached["status"] == "CACHED"


def test_manifest_records_all_cache_key_elements(env):
    run, _ = analyze(env)
    m = manifest(run, practice_extractor.SPEC)
    for key in ("source_sha256", "transcript_sha256", "agent", "agent_version", "prompt_sha256", "model",
                "schema_version", "schema_sha256", "created_at", "cache_hit", "usage", "cache_key"):
        assert key in m, key
    assert m["model"] == "fake-model" and m["agent"] == "practice_extractor"
    assert m["cache_key"] == compute_cache_key(m)


def test_prompt_change_triggers_new_call(env, tmp_path, monkeypatch):
    analyze(env)
    prompt = tmp_path / "practice_extractor_v2.md"
    prompt.write_text(practice_extractor.SPEC.system_prompt + "\nConsigne ajoutée.", encoding="utf-8")
    changed = dataclasses.replace(practice_extractor.SPEC, prompt_path=prompt)
    monkeypatch.setattr(analysis, "AGENTS", (changed, interaction_signal_reader.SPEC))
    _, transport = analyze(env)
    assert [c["agent"] for c in transport.calls] == [PRACTICE]  # l'autre agent reste en cache


def test_agent_version_change_triggers_new_call(env, monkeypatch):
    analyze(env)
    changed = dataclasses.replace(interaction_signal_reader.SPEC, version="1.1")
    monkeypatch.setattr(analysis, "AGENTS", (practice_extractor.SPEC, changed))
    _, transport = analyze(env)
    assert [c["agent"] for c in transport.calls] == [INTERACTION]


def test_model_change_triggers_new_calls(env):
    analyze(env)
    run, transport = analyze(env, settings=fake_settings(model="other-fake-model"))
    assert len(transport.calls) == 2
    assert all(c["params"]["model"] == "other-fake-model" for c in transport.calls)
    assert manifest(run, practice_extractor.SPEC)["model"] == "other-fake-model"


def test_generation_parameter_change_triggers_new_calls(env):
    analyze(env)
    _, transport = analyze(env, settings=fake_settings(effort="low"))
    assert len(transport.calls) == 2


def test_transcript_change_triggers_new_calls(tmp_path):
    cache = AnalysisCache(tmp_path / "cache")
    first = si.make_ingested_run(tmp_path / "a")
    analyze_run(first, settings=fake_settings(), transport=FakeTransport(responders()), cache=cache)
    modified = si.TEXT.replace("c'est rapide", "c'est très rapide")
    second = si.make_ingested_run(tmp_path / "b", [(si.FILENAME, modified.encode("utf-8"))])
    transport = FakeTransport(responders())
    analyze_run(second, settings=fake_settings(), transport=transport, cache=cache)
    assert len(transport.calls) == 2


def test_same_file_in_a_new_run_reuses_the_cache(tmp_path):
    """Recharger la page puis réimporter le même fichier ne repaie pas les appels."""
    cache = AnalysisCache(tmp_path / "cache")
    analyze_run(si.make_ingested_run(tmp_path / "a"), settings=fake_settings(),
                transport=FakeTransport(responders()), cache=cache)
    transport = FakeTransport(responders())
    analyze_run(si.make_ingested_run(tmp_path / "b"), settings=fake_settings(), transport=transport, cache=cache)
    assert transport.calls == []


def test_force_bypasses_the_cache(env):
    analyze(env)
    run, transport = analyze(env, force=True)
    assert len(transport.calls) == 2 and not manifest(run, practice_extractor.SPEC)["cache_hit"]
    assert plan_analysis(run, [si.INTERVIEW_ID], fake_settings(), env["cache"], force=True)["calls"] == 2


def test_failed_calls_are_not_cached(env):
    failing = FakeTransport({PRACTICE: text_response("pas du JSON"), INTERACTION: text_response(si.GOOD_SIGNALS)})
    analyze_run(env["run"], settings=fake_settings(), transport=failing, cache=env["cache"])
    _, transport = analyze(env)
    assert [c["agent"] for c in transport.calls] == [PRACTICE]


def test_corrupted_cache_entry_is_ignored(env):
    analyze(env)
    for path in (env["cache"].root / "practice_extractor").glob("*.json"):
        path.write_text("{corrompu", encoding="utf-8")
    _, transport = analyze(env)
    assert [c["agent"] for c in transport.calls] == [PRACTICE]


def test_cache_key_depends_on_every_field():
    base = {k: f"value-{k}" for k in KEY_FIELDS}
    base["request_params"] = {"effort": None, "temperature": None}
    key = compute_cache_key(base)
    assert compute_cache_key(dict(base)) == key
    for field in KEY_FIELDS:
        changed = dict(base)
        changed[field] = "autre" if field != "request_params" else {"effort": "low", "temperature": None}
        assert compute_cache_key(changed) != key, field
    with pytest.raises(ValueError):
        compute_cache_key({"agent": "x"})
