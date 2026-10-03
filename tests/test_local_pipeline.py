"""Étapes 3 à 6 exécutées LOCALEMENT par TRACE (core/local_pipeline.py) : runner local, validateurs, gardes.

Le vrai LocalAgentRunner tourne sur un faux Ollama (tests/fake_llm.use_fake_runtime) : mêmes requêtes, même
sortie structurée, mêmes corrections locales, même contrôle méthodologique qu'avec un vrai modèle — sans modèle,
sans réseau, sans clé. Les sorties sont comparées à celles du pipeline de référence (mêmes réponses d'agent données
directement aux orchestrateurs), pour vérifier que seuls les moyens d'obtenir la réponse ont changé.
"""

import copy
import json
from pathlib import Path

import pytest

from core import accountability, analysis, config, cross_interview
from core import accountability_candidates as ac
from core import local_pipeline as lp
from core import trajectory_candidates as tc
from core.analysis_cache import AnalysisCache
from core.llm_client import LLMError
from core.stage3_restore import restore_stage3
from core.stage4_restore import restore_stage4
from tests import synthetic_interviews as si
from tests import synthetic_long_interview as L
from tests import synthetic_stage4 as S4
from tests import synthetic_stage4_otmane as O
from tests import synthetic_stage5 as S5
from tests import synthetic_stage6 as S6
from tests.fake_llm import (ACCOUNTABILITY, AUDITOR, COMPARATOR, INTERACTION, LONG_DISTANCE, PRACTICE, TRAJECTORY,
                            REPAIR, FakeAgents, fake_settings, text_response, use_fake_runtime)

GOOD = {PRACTICE: lambda p: text_response(si.GOOD_PRACTICES), INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)}
STAGE5_FILES = (config.STUDENT_TRAJECTORY_FILENAME, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME,
                config.STUDENT_TRAJECTORY_MANIFEST_FILENAME)
STAGE6_FILES = (config.CROSS_INTERVIEW_COMPARISON_FILENAME, config.CROSS_INTERVIEW_VALIDATION_FILENAME,
                config.CROSS_INTERVIEW_MANIFEST_FILENAME)


def analysis_dir(run: dict, interview_id: str) -> Path:
    return Path(run["output_dir"]) / config.INTERVIEWS_SUBDIR / interview_id / config.ANALYSIS_SUBDIR


def load(run: dict, name: str, interview_id: str) -> dict:
    return json.loads((analysis_dir(run, interview_id) / name).read_text(encoding="utf-8"))


def stage3_agents(stage3: dict, builder=None, mapper=None) -> dict:
    agents = dict(stage3)
    if builder:
        agents[ACCOUNTABILITY] = builder
    if mapper:
        agents[TRAJECTORY] = mapper
    return agents


# --- Étape 3 -----------------------------------------------------------------------------------------------

def test_stage3_runs_locally_without_any_key_and_writes_the_usual_outputs(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, GOOD)
    result = lp.run_stage("3", si.make_ingested_run(tmp_path))
    status, run = result["status"], result["metadata"]
    assert status["status"] == lp.STAGE_COMPLETE and status["api_calls"] == 0 and status["call_count"] == 2
    assert (status["execution"], status["runtime"], status["external_data"]) == ("locale", "Ollama", "aucune")
    assert sorted(c["agent"] for c in ollama.calls) == [INTERACTION, PRACTICE]
    for call in ollama.calls:  # prompt du dépôt et JSON Schema strict de l'agent
        spec = analysis.PRACTICE if call["agent"] == PRACTICE else analysis.INTERACTION
        assert call["payload"]["messages"][0]["content"] == spec.system_prompt
        assert call["payload"]["format"] == spec.output_schema
    manifest = load(run, "practice_manifest.json", si.INTERVIEW_ID)
    assert manifest["model"] == "qwen2.5:7b" and manifest["api_calls"] == 0 and manifest["billed_this_run"] is False
    assert manifest["status"] == "SUCCESS" and manifest["request_id"].startswith(f"{si.INTERVIEW_ID}__practice")
    journal = Path(run["output_dir"]) / "local_runs" / "stage3"
    assert len(list(journal.glob(f"{si.INTERVIEW_ID}__*.json"))) == 2 and (journal / "status.json").is_file()
    assert run["pipeline"][config.PRACTICE_STEP] == "terminé (1/1 entretien(s))"


def test_stage3_outputs_equal_the_reference_pipeline_outputs(tmp_path, monkeypatch):
    use_fake_runtime(monkeypatch, GOOD)
    local = lp.run_stage("3", si.make_ingested_run(tmp_path / "local"))["metadata"]
    reference = analysis.analyze_run(si.make_ingested_run(tmp_path / "ref"), settings=fake_settings(),
                                     client=FakeAgents(GOOD), cache=AnalysisCache(tmp_path / "c"))
    for name, key in (("practice_extractor.json", "practices"), ("interaction_signals.json", "signals")):
        assert load(local, name, si.INTERVIEW_ID)[key] == load(reference, name, si.INTERVIEW_ID)[key]
    a, b = (load(r, config.EVIDENCE_VALIDATION_FILENAME, si.INTERVIEW_ID) for r in (local, reference))
    assert a["agents"] == b["agents"]


def test_a_complete_stage_is_never_replayed_unless_forced(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, GOOD)
    run = lp.run_stage("3", si.make_ingested_run(tmp_path))["metadata"]
    out = analysis_dir(run, si.INTERVIEW_ID)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    again = lp.run_stage("3", run)
    assert again["status"]["skipped"] == [si.INTERVIEW_ID] and len(ollama.calls) == 2
    assert again["status"]["interviews"][0]["phase"] == "déjà terminée — non rejouée"
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.name in before} == before
    forced = lp.run_stage("3", run, force=True)
    assert forced["status"]["status"] == lp.STAGE_COMPLETE and len(ollama.calls) == 4


def test_ollama_stopped_is_a_clear_error_and_nothing_is_written(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, GOOD, available=False)
    run = si.make_ingested_run(tmp_path)
    with pytest.raises(LLMError) as info:
        lp.run_stage("3", run)
    assert info.value.code == "OLLAMA_UNAVAILABLE" and "ollama serve" in info.value.user_message
    assert ollama.calls == [] and not analysis_dir(run, si.INTERVIEW_ID).exists()
    assert "stage3_local" not in run


def test_missing_model_is_a_clear_error_and_nothing_is_written(tmp_path, monkeypatch):
    use_fake_runtime(monkeypatch, GOOD, models=["llama3.2:3b"])
    run = si.make_ingested_run(tmp_path)
    with pytest.raises(LLMError) as info:
        lp.run_until(run, "5")
    assert info.value.code == "MODEL_NOT_FOUND" and "ollama pull qwen2.5:7b" in info.value.user_message
    assert not analysis_dir(run, si.INTERVIEW_ID).exists()


def test_a_quote_with_typographic_differences_is_restored_without_the_model(tmp_path, monkeypatch):
    altered = copy.deepcopy(si.GOOD_PRACTICES)
    exact = altered["practices"][0]["evidence"][0]["quote"]
    altered["practices"][0]["evidence"][0]["quote"] = exact.replace("'", "’").upper() + " "
    ollama = use_fake_runtime(monkeypatch, {**GOOD, PRACTICE: text_response(altered)})
    run = lp.run_stage("3", si.make_ingested_run(tmp_path))["metadata"]
    assert len(ollama.calls_for(PRACTICE)) == 1 and not ollama.calls_for(REPAIR + PRACTICE)  # aucun appel en plus
    document = load(run, "practice_extractor.json", si.INTERVIEW_ID)
    assert document["status"] == "SUCCESS" and document["practices"][0]["evidence"][0]["quote"] == exact
    assert load(run, config.EVIDENCE_VALIDATION_FILENAME, si.INTERVIEW_ID)["total_invalid_evidence"] == 0


def test_the_validator_is_never_bypassed_when_a_quote_is_invented(tmp_path, monkeypatch):
    fabricated = si.with_fabricated_quote(si.GOOD_PRACTICES, "practices")
    ollama = use_fake_runtime(monkeypatch, {**GOOD, PRACTICE: lambda p: text_response(fabricated)})
    run = lp.run_stage("3", si.make_ingested_run(tmp_path))["metadata"]
    # citation introuvable : ni régénération, ni réparation par le modèle ; l'objet reste signalé par TRACE
    assert len(ollama.calls_for(PRACTICE)) == 1 and not ollama.calls_for(REPAIR + PRACTICE)
    document = load(run, "practice_extractor.json", si.INTERVIEW_ID)
    assert document["status"] == "SUCCESS_WITH_WARNINGS"
    assert load(run, config.EVIDENCE_VALIDATION_FILENAME, si.INTERVIEW_ID)["total_invalid_evidence"] == 1


def test_speaker_audit_runs_first_and_its_warnings_reach_both_agents(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, {
        PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
        INTERACTION: lambda p: text_response({"signals": [], "reading_notes": None}),
        AUDITOR: lambda p: text_response(si.AUDIT_ASSESSMENTS)})
    run = lp.run_stage("3", si.make_ingested_run(tmp_path, si.AUDIT_FILES))["metadata"]
    assert [c["agent"] for c in ollama.calls][0] == AUDITOR and len(ollama.calls) == 3
    for agent in (PRACTICE, INTERACTION):
        assert "speaker_warning" in ollama.calls_for(agent)[0]["params"]["messages"][0]["content"]
    assert run["stage3_local"]["status"] == lp.STAGE_COMPLETE


def test_a_failed_block_makes_the_stage_incomplete_then_only_it_is_redone(tmp_path, monkeypatch):
    def broken(params):
        if any(t["turn_id"] == L.ltid(200) for t in L.sent_turns(params)):
            return '{"signals": ['
        return L.simulated_chunk_reader(params)

    agents = {PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
              INTERACTION: broken, LONG_DISTANCE: L.simulated_long_distance_reader}
    ollama = use_fake_runtime(monkeypatch, agents)
    first = lp.run_stage("3", si.make_ingested_run(tmp_path, L.long_files()))
    assert first["status"]["status"] == lp.STAGE_FAILED
    [interview] = first["status"]["interviews"]
    assert any("Analyse INCOMPLÈTE" in e for e in interview["errors"])
    calls = len(ollama.calls)
    ollama.responders[INTERACTION] = L.simulated_chunk_reader
    second = lp.run_stage("3", first["metadata"])
    assert second["status"]["status"] == lp.STAGE_COMPLETE
    redone = ollama.calls[calls:]  # les blocs réussis viennent du cache TRACE
    assert sorted(c["agent"] for c in redone) == [INTERACTION, LONG_DISTANCE]


# --- Étape 4 -----------------------------------------------------------------------------------------------

def episodes_comparable(document: dict) -> dict:
    keys = ("status", "analysis_complete", "candidate_count", "component_count", "episode_count",
            "accountability_episode_count", "ordinary_practice_count", "uncertain_count", "rejected_episode_count",
            "usable_episode_count", "unmarked_practice_count", "validation_error_count", "validation_warning_count",
            "candidates", "unmarked_practices", "episodes")
    return {k: document[k] for k in keys}


def test_stage4_runs_locally_and_equals_the_reference_pipeline(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, stage3_agents(S4.stage3_responders(), S4.REFERENCE_BUILDER))
    result = lp.run_until(si.make_ingested_run(tmp_path / "local", S4.FILES), "4")
    assert result["stage"] == "4" and result["status"]["status"] == lp.STAGE_COMPLETE
    assert [c["agent"] for c in ollama.calls].count(ACCOUNTABILITY) == 1
    local = load(result["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME, S4.INTERVIEW_ID)
    reference, _ = S5.run_to_stage4(tmp_path / "ref", S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    assert episodes_comparable(local) == episodes_comparable(
        load(reference, config.ACCOUNTABILITY_EPISODES_FILENAME, S4.INTERVIEW_ID))
    manifest = load(result["metadata"], config.ACCOUNTABILITY_MANIFEST_FILENAME, S4.INTERVIEW_ID)
    assert manifest["api_calls"] == 0 and manifest["model"] == "qwen2.5:7b" and manifest["llm_called"] is True


def test_stage4_without_stage3_is_blocked_and_calls_nothing(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    result = lp.run_stage("4", si.make_ingested_run(tmp_path, S4.FILES))
    assert result["status"]["status"] == lp.STAGE_BLOCKED and ollama.calls == []


def test_stage4_blocks_are_independent_calls_merged_in_block_order(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "SINGLE_CALL_MAX_INPUT_TOKENS", 2500)  # entretien long : plusieurs blocs
    ollama = use_fake_runtime(monkeypatch, stage3_agents(O.stage3_responders(), O.builder()))
    result = lp.run_until(si.make_ingested_run(tmp_path, O.files()), "4")
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    prepared = accountability.prepare_stage4(result["metadata"]["files"][0]["ingestion"])
    calls = ollama.calls_for(ACCOUNTABILITY)
    assert len(calls) == len(prepared.requests) > 1
    assert {c["params"]["messages"][0]["content"] for c in calls} == {r["user_message"] for r in prepared.requests}
    document = load(result["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME, O.INTERVIEW_ID)
    assert document["chunk_count"] == len(calls) and document["status"] == "SUCCESS"
    assert document["usable_episode_count"] == document["candidate_count"] - 1


def test_restored_stage3_feeds_the_local_stage4(tmp_path, monkeypatch):
    before, _ = S5.run_to_stage4(tmp_path / "before", S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    out = analysis_dir(before, S4.INTERVIEW_ID)
    uploads = [(n, (out / n).read_bytes()) for n in ("practice_extractor.json", "interaction_signals.json",
                                                      "evidence_validation.json", "speaker_attribution_audit.json")]
    run = restore_stage3(si.make_ingested_run(tmp_path / "after", S4.FILES), S4.INTERVIEW_ID, uploads)
    ollama = use_fake_runtime(monkeypatch, {ACCOUNTABILITY: S4.REFERENCE_BUILDER})
    result = lp.run_until(run, "4")
    assert result["status"]["status"] == lp.STAGE_COMPLETE
    assert [c["agent"] for c in ollama.calls] == [ACCOUNTABILITY]  # ni étape 3, ni audit
    stage3 = lp.run_stage("3", result["metadata"])["status"]
    assert stage3["skipped"] == [S4.INTERVIEW_ID]  # étape 3 restaurée : complète, non rejouée


# --- Étape 5 -----------------------------------------------------------------------------------------------

def trajectory_comparable(document: dict) -> dict:
    return {k: v for k, v in document.items() if k not in ("generated_at", "model", "source_hashes", "cache_hit")}


def test_stage5_runs_locally_and_equals_the_reference_pipeline(tmp_path, monkeypatch):
    mapper = S5.scripted_mapper(S5.REFERENCE_PLAN)
    ollama = use_fake_runtime(monkeypatch, stage3_agents(S4.stage3_responders(), S4.REFERENCE_BUILDER, mapper))
    result = lp.run_until(si.make_ingested_run(tmp_path / "local", S4.FILES), "5")
    assert result["stage"] == "5" and result["status"]["status"] == lp.STAGE_COMPLETE
    assert len(ollama.calls_for(TRAJECTORY)) == 1
    run = result["metadata"]
    reference, cache = S5.run_to_stage4(tmp_path / "ref", S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    reference, _ = S5.run_stage5(reference, cache, mapper)
    a, b = (load(r, config.STUDENT_TRAJECTORY_FILENAME, S4.INTERVIEW_ID) for r in (run, reference))
    assert trajectory_comparable(a) == trajectory_comparable(b) and a["status"] == "SUCCESS"
    manifest = load(run, config.STUDENT_TRAJECTORY_MANIFEST_FILENAME, S4.INTERVIEW_ID)
    assert manifest["api_calls"] == 0 and manifest["billed_this_run"] is False and manifest["model"] == "qwen2.5:7b"


def test_run_until_5_never_replays_completed_stages_and_redoes_only_stale_ones(tmp_path, monkeypatch):
    mapper = S5.scripted_mapper(S5.REFERENCE_PLAN)
    ollama = use_fake_runtime(monkeypatch, stage3_agents(S4.stage3_responders(), S4.REFERENCE_BUILDER, mapper))
    run = lp.run_until(si.make_ingested_run(tmp_path, S4.FILES), "5")["metadata"]
    calls = len(ollama.calls)
    again = lp.run_until(run, "5")
    assert again["status"]["status"] == lp.STAGE_COMPLETE and len(ollama.calls) == calls
    assert lp.stage_state("5", run["files"][0])["status"] == "COMPLETE"
    # sorties de l'étape 4 réécrites : l'étape 5 devient périmée, et seule elle est rejouée (étapes 3 et 4 gardées)
    episodes = analysis_dir(run, S4.INTERVIEW_ID) / config.ACCOUNTABILITY_EPISODES_FILENAME
    episodes.write_text(json.dumps(json.loads(episodes.read_text(encoding="utf-8")), ensure_ascii=False, indent=1),
                        encoding="utf-8")
    assert lp.stage_state("5", run["files"][0])["status"] == "STALE"
    redone = lp.run_until(again["metadata"], "5")
    assert redone["status"]["status"] == lp.STAGE_COMPLETE and redone["status"]["skipped"] == []
    assert lp.stage_state("5", run["files"][0])["status"] == "COMPLETE"
    assert ACCOUNTABILITY not in [c["agent"] for c in ollama.calls[calls:]]


def test_requalifications_are_applied_as_in_the_reference_pipeline(tmp_path, monkeypatch):
    case = S5.NOPATTERN
    mapper = case.mapper("adversarial")
    use_fake_runtime(monkeypatch, stage3_agents(case.stage3_responders(), case.builder(), mapper))
    run = lp.run_until(si.make_ingested_run(tmp_path / "local", case.files), "5")["metadata"]
    reference, cache = S5.case_to_stage4(tmp_path / "ref", case)
    reference, _ = S5.run_stage5(reference, cache, mapper)
    a, b = (load(r, config.STUDENT_TRAJECTORY_FILENAME, case.interview_id) for r in (run, reference))
    assert trajectory_comparable(a) == trajectory_comparable(b)
    assert any(c.get("model_claim_type") for c in a["trajectory_claims"])  # requalifié par TRACE, comme avant


def test_oversized_stage5_request_stays_single_and_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "SINGLE_CALL_MAX_INPUT_TOKENS", 500)
    ollama = use_fake_runtime(monkeypatch, stage3_agents(S4.stage3_responders(), S4.REFERENCE_BUILDER,
                                                         S5.scripted_mapper(S5.REFERENCE_PLAN)))
    result = lp.run_until(si.make_ingested_run(tmp_path, S4.FILES), "5")
    assert len(ollama.calls_for(TRAJECTORY)) == 1  # jamais découpé
    [interview] = result["status"]["interviews"]
    assert interview["over_single_call_threshold"] and "TRACE_OLLAMA_NUM_CTX" in interview["warnings"][0]
    validation = load(result["metadata"], config.STUDENT_TRAJECTORY_VALIDATION_FILENAME, S4.INTERVIEW_ID)
    assert "PAYLOAD_OVER_THRESHOLD" in {i["code"] for i in validation["issues"]}


def test_restored_stages_3_and_4_feed_the_local_stage5(tmp_path, monkeypatch):
    before, _ = S5.run_to_stage4(tmp_path / "before", S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    out = analysis_dir(before, S4.INTERVIEW_ID)
    stage3 = [(n, (out / n).read_bytes()) for n in ("practice_extractor.json", "interaction_signals.json",
                                                     "evidence_validation.json", "speaker_attribution_audit.json")]
    stage4 = [(n, (out / n).read_bytes()) for n in (config.ACCOUNTABILITY_EPISODES_FILENAME,
                                                     config.ACCOUNTABILITY_VALIDATION_FILENAME)]
    run = restore_stage3(si.make_ingested_run(tmp_path / "after", S4.FILES), S4.INTERVIEW_ID, stage3)
    run = restore_stage4(run, S4.INTERVIEW_ID, stage4)
    ollama = use_fake_runtime(monkeypatch, {TRAJECTORY: S5.scripted_mapper(S5.REFERENCE_PLAN)})
    result = lp.run_until(run, "5")
    assert result["status"]["status"] == lp.STAGE_COMPLETE and [c["agent"] for c in ollama.calls] == [TRAJECTORY]


# --- Étape 6 -----------------------------------------------------------------------------------------------

def comparator(plan=S6.GOOD_PLAN, mutate=None) -> dict:
    return {COMPARATOR: S6.scripted_comparator(plan, mutate=mutate)}


def outputs(corpus_dir) -> dict:
    out = Path(corpus_dir) / config.ANALYSIS_SUBDIR
    return {name: json.loads((out / name).read_text(encoding="utf-8")) for name in STAGE6_FILES}


def comparison_comparable(document: dict) -> dict:
    return {k: v for k, v in document.items() if k not in ("generated_at", "model", "cache_hit", "validated_at")}


def reference_stage6(tmp_path, uploads, agents) -> dict:
    manifest = cross_interview.run_stage6(uploads, settings=fake_settings(), client=FakeAgents(agents),
                                          cache=AnalysisCache(tmp_path / "ref_cache"), base_dir=tmp_path / "ref")
    return outputs(manifest["corpus_dir"])


def test_stage6_runs_locally_on_two_interviews_and_equals_the_reference(tmp_path, monkeypatch):
    uploads = S6.uploads(S6.IDS_2)
    ollama = use_fake_runtime(monkeypatch, comparator())
    result = lp.run_stage6(uploads, base_dir=tmp_path / "local")
    status = result["status"]
    assert status["status"] == lp.STAGE_COMPLETE and status["mode"] == "exploratory" and status["api_calls"] == 0
    assert (status["execution"], status["runtime"], status["external_data"]) == ("locale", "Ollama", "aucune")
    [call] = ollama.calls
    assert call["payload"]["format"] == cross_interview.SPEC.output_schema
    assert call["params"]["messages"][0]["content"] == cross_interview.prepare_stage6(uploads).request["user_message"]
    local = outputs(result["corpus_dir"])
    reference = reference_stage6(tmp_path, uploads, comparator())
    for name in STAGE6_FILES[:2]:
        assert comparison_comparable(local[name]) == comparison_comparable(reference[name]), name
    manifest = local[config.CROSS_INTERVIEW_MANIFEST_FILENAME]
    assert manifest["api_calls"] == 0 and manifest["model"] == "qwen2.5:7b" and manifest["llm_called"] is True
    assert (Path(result["corpus_dir"]) / "local_runs" / "stage6" / "status.json").is_file()


def test_seventeen_interviews_keep_negative_cases_and_review_items(tmp_path, monkeypatch):
    uploads = S6.uploads(S6.IDS_17)
    use_fake_runtime(monkeypatch, comparator())
    local = outputs(lp.run_stage6(uploads, base_dir=tmp_path / "local")["corpus_dir"])
    reference = reference_stage6(tmp_path, uploads, comparator())
    document = local[config.CROSS_INTERVIEW_COMPARISON_FILENAME]
    assert comparison_comparable(document) == comparison_comparable(reference[config.CROSS_INTERVIEW_COMPARISON_FILENAME])
    assert sorted(n["interview_id"] for n in document["negative_cases"]) == ["ENT_D", "ENT_H", "ENT_L"]


def test_an_invented_identifier_is_sent_back_for_correction_then_rejected_as_before(tmp_path, monkeypatch):
    def invent(output, material):
        output["cross_case_claims"][0]["support"][0]["claim_ids"] = ["TC999"]
        return output
    uploads = S6.uploads(S6.IDS_4)
    ollama = use_fake_runtime(monkeypatch, comparator(mutate=invent))
    local = outputs(lp.run_stage6(uploads, base_dir=tmp_path / "local")["corpus_dir"])
    assert len(ollama.calls) == 2 and "UNKNOWN_OBJECT_ID" in ollama.calls[1]["params"]["history"][1]["content"]
    reference = reference_stage6(tmp_path, uploads, comparator(mutate=invent))
    assert comparison_comparable(local[config.CROSS_INTERVIEW_COMPARISON_FILENAME]) == comparison_comparable(
        reference[config.CROSS_INTERVIEW_COMPARISON_FILENAME])  # mêmes rejets que le validateur habituel


def test_stage6_guard_is_based_on_corpus_identity(tmp_path, monkeypatch):
    uploads = S6.uploads(S6.IDS_4)
    ollama = use_fake_runtime(monkeypatch, comparator())
    done = lp.run_stage6(uploads, base_dir=tmp_path)
    out = Path(done["corpus_dir"]) / config.ANALYSIS_SUBDIR
    before = {name: (out / name).read_bytes() for name in STAGE6_FILES}
    again = lp.run_stage6(list(reversed(uploads)), base_dir=tmp_path)  # même corpus, autre ordre d'import
    assert again["status"]["skipped"] is True and len(ollama.calls) == 1
    assert {name: (out / name).read_bytes() for name in STAGE6_FILES} == before
    forced = lp.run_stage6(uploads, base_dir=tmp_path, force=True)
    assert forced["status"]["status"] == lp.STAGE_COMPLETE and len(ollama.calls) == 2
    manifest_path = out / config.CROSS_INTERVIEW_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps({**manifest, "prompt_sha256": "autre-prompt"}), encoding="utf-8")
    assert lp.stage6_state(cross_interview.prepare_stage6(uploads), Path(done["corpus_dir"]))["status"] == "STALE"
    assert lp.run_stage6(uploads, base_dir=tmp_path)["status"]["skipped"] is False


def test_a_modified_corpus_is_a_new_analysis(tmp_path, monkeypatch):
    use_fake_runtime(monkeypatch, comparator())
    uploads = S6.uploads(S6.IDS_4)
    base = lp.run_stage6(uploads, base_dir=tmp_path)["status"]["corpus_id"]
    added = lp.run_stage6(uploads + S6.uploads(["ENT_E"]), base_dir=tmp_path)["status"]
    assert added["corpus_id"] != base and added["skipped"] is False and added["status"] == lp.STAGE_COMPLETE


def test_fewer_than_two_usable_interviews_block_without_needing_ollama(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, comparator(), available=False)
    result = lp.run_stage6(S6.uploads(["ENT_A"]), base_dir=tmp_path)
    assert result["status"]["status"] == lp.STAGE_BLOCKED and ollama.calls == []
    assert "moins de deux entretiens exploitables" in result["status"]["reason"]


def test_stage6_with_ollama_stopped_is_a_clear_error(tmp_path, monkeypatch):
    use_fake_runtime(monkeypatch, comparator(), available=False)
    with pytest.raises(LLMError) as info:
        lp.run_stage6(S6.uploads(S6.IDS_2), base_dir=tmp_path)
    assert info.value.code == "OLLAMA_UNAVAILABLE" and not list(tmp_path.glob("*/analysis/*.json"))


# --- Chaîne complète : un entretien → étape 5, deux entretiens → étape 6 ------------------------------------

def test_two_synthetic_interviews_to_stage5_then_stage6(tmp_path, monkeypatch):
    reference = stage3_agents(S4.stage3_responders(), S4.REFERENCE_BUILDER, S5.scripted_mapper(S5.REFERENCE_PLAN))
    case = S5.TEMPORAL
    temporal = stage3_agents(case.stage3_responders(), case.builder(), case.mapper())

    def by_interview(agent):
        def respond(params):
            content = params["messages"][0]["content"]
            return (temporal if case.interview_id + "_" in content else reference)[agent](params)
        return respond

    agents = {name: by_interview(name) for name in (PRACTICE, INTERACTION, AUDITOR, ACCOUNTABILITY, TRAJECTORY)}
    ollama = use_fake_runtime(monkeypatch, {**agents, COMPARATOR: lambda p: text_response(
        {"cross_case_claims": [], "cross_case_summary": "Deux entretiens disponibles : aucune régularité retenue.",
         "confidence": "low", "needs_review": False, "comparator_notes": None})})
    run_a = lp.run_until(si.make_ingested_run(tmp_path / "a", S4.FILES), "5")
    run_b = lp.run_until(si.make_ingested_run(tmp_path / "b", case.files), "5")
    assert run_a["status"]["status"] == run_b["status"]["status"] == lp.STAGE_COMPLETE
    from core import cross_interview_corpus as corpus
    uploads = corpus.run_stage5_uploads(run_a["metadata"]) + corpus.run_stage5_uploads(run_b["metadata"])
    result = lp.run_stage6(uploads, base_dir=tmp_path / "corpus")
    assert result["status"]["status"] == lp.STAGE_COMPLETE and result["status"]["n_usable"] == 2
    assert [c["agent"] for c in ollama.calls].count(COMPARATOR) == 1
