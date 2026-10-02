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
from tests.fake_llm import INTERACTION, PRACTICE, FakeAgents, fake_settings, text_response


def responders():
    return {PRACTICE: lambda p: text_response(si.GOOD_PRACTICES), INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)}


@pytest.fixture
def env(tmp_path):
    run = si.make_ingested_run(tmp_path)
    return {"run": run, "cache": AnalysisCache(tmp_path / "cache")}


def analyze(env, settings=None, **kwargs):
    transport = FakeAgents(responders())
    run = analyze_run(env["run"], settings=settings or fake_settings(), client=transport, cache=env["cache"],
                      **kwargs)
    return run, transport


def manifest(run, spec):
    return json.loads((Path(run["files"][0]["analysis"]["analysis_dir"]) / spec.manifest_filename).read_text())


def test_identical_inputs_hit_the_cache(env):
    run, first = analyze(env)
    assert len(first.calls) == 2
    assert plan_analysis(run, [si.INTERVIEW_ID], fake_settings(), env["cache"]) == {
        "interviews": 1, "calls": 0, "cached": 2, "audit_calls": 0, "candidate_turns": 0, "exact": True}
    run, second = analyze(env)
    assert second.calls == []  # aucun appel API
    for spec in analysis.AGENTS:
        m = manifest(run, spec)
        assert m["cache_hit"] is True and m["status"] == "CACHED" and m["api_calls"] == 0
        assert m["billed_this_run"] is False  # champ legacy : jamais de coût
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
    changed = dataclasses.replace(interaction_signal_reader.SPEC, version=interaction_signal_reader.SPEC.version + ".1")
    monkeypatch.setattr(analysis, "AGENTS", (practice_extractor.SPEC, changed))
    _, transport = analyze(env)
    assert [c["agent"] for c in transport.calls] == [INTERACTION]


def test_model_change_triggers_new_calls(env):
    analyze(env)
    run, transport = analyze(env, settings=fake_settings(model="other-fake-model"))
    assert len(transport.calls) == 2
    assert manifest(run, practice_extractor.SPEC)["model"] == "other-fake-model"


def test_transcript_change_triggers_new_calls(tmp_path):
    cache = AnalysisCache(tmp_path / "cache")
    first = si.make_ingested_run(tmp_path / "a")
    analyze_run(first, settings=fake_settings(), client=FakeAgents(responders()), cache=cache)
    modified = si.TEXT.replace("c'est rapide", "c'est très rapide")
    second = si.make_ingested_run(tmp_path / "b", [(si.FILENAME, modified.encode("utf-8"))])
    transport = FakeAgents(responders())
    analyze_run(second, settings=fake_settings(), client=transport, cache=cache)
    assert len(transport.calls) == 2


def test_same_file_in_a_new_run_reuses_the_cache(tmp_path):
    """Recharger la page puis réimporter le même fichier ne repaie pas les appels."""
    cache = AnalysisCache(tmp_path / "cache")
    analyze_run(si.make_ingested_run(tmp_path / "a"), settings=fake_settings(),
                client=FakeAgents(responders()), cache=cache)
    transport = FakeAgents(responders())
    analyze_run(si.make_ingested_run(tmp_path / "b"), settings=fake_settings(), client=transport, cache=cache)
    assert transport.calls == []


def test_force_bypasses_the_cache(env):
    analyze(env)
    run, transport = analyze(env, force=True)
    assert len(transport.calls) == 2 and not manifest(run, practice_extractor.SPEC)["cache_hit"]
    assert plan_analysis(run, [si.INTERVIEW_ID], fake_settings(), env["cache"], force=True)["calls"] == 2


def test_failed_calls_are_not_cached(env):
    failing = FakeAgents({PRACTICE: text_response("pas du JSON"), INTERACTION: text_response(si.GOOD_SIGNALS)})
    analyze_run(env["run"], settings=fake_settings(), client=failing, cache=env["cache"])
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


def test_interaction_prompt_change_does_not_invalidate_practice(env, tmp_path, monkeypatch):
    analyze(env)
    prompt = tmp_path / "interaction_signal_reader_v2.md"
    prompt.write_text(interaction_signal_reader.SPEC.system_prompt + "\nConsigne ajoutée.", encoding="utf-8")
    changed = dataclasses.replace(interaction_signal_reader.SPEC, prompt_path=prompt)
    monkeypatch.setattr(analysis, "AGENTS", (practice_extractor.SPEC, changed))
    run, transport = analyze(env)
    assert [c["agent"] for c in transport.calls] == [INTERACTION]
    assert manifest(run, practice_extractor.SPEC)["status"] == "CACHED"


def test_guard_and_validator_versions_do_not_invalidate_the_cache(env, monkeypatch):
    """Garde-fou et validation sont recalculés à chaque lecture du cache : les changer ne repaie rien."""
    from core import evidence_validator, interpretation_guard
    analyze(env)
    monkeypatch.setattr(interpretation_guard, "GUARD_VERSION", "9.9")
    monkeypatch.setattr(evidence_validator, "VALIDATOR_VERSION", "9.9")
    run, transport = analyze(env)
    assert transport.calls == [] and manifest(run, practice_extractor.SPEC)["guard_version"] == "9.9"


def test_agent_versions_of_stage_3_5_have_distinct_cache_keys():
    """Les versions 1.2 des deux agents ne peuvent pas réutiliser une entrée de cache 1.1."""
    for spec in (practice_extractor.SPEC, interaction_signal_reader.SPEC):
        # étape 3.7 : Interaction Reader 1.3 (consignes), schémas inchangés (1.2)
        assert spec.schema_version == "1.2" and spec.version == {"practice_extractor": "1.2",
                                                                  "interaction_signal_reader": "1.3"}[spec.name]
        old = dataclasses.replace(spec, version="1.1", schema_version="1.1")
        fields = {"source_sha256": "s", "transcript_sha256": "t", "model": "m",
                  "request_params": {"effort": None, "temperature": None}}
        assert compute_cache_key({**fields, **spec.identity()}) != compute_cache_key({**fields, **old.identity()})
