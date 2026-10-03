"""Étape 3 : réparation CIBLÉE des seuls objets fautifs (core/stage3_repair.py), jamais de régénération complète d'une
réponse lisible. Faux Ollama derrière le vrai LocalAgentRunner et le vrai MethodChecker : aucun modèle, aucun réseau."""

import copy
import json
from pathlib import Path

from agents.interaction_signal_reader import SPEC as INTERACTION_SPEC

from core import config, local_jobs
from core import local_pipeline as lp
from core import stage3_repair
from scripts.trace_benchmark import repair_summary
from tests import synthetic_interviews as si
from tests import synthetic_long_interview as L
from tests import synthetic_stage4 as S4
from tests.fake_llm import (INTERACTION, LONG_DISTANCE, PRACTICE, REPAIR, agent_error, repair_object, repair_turns,
                            repaired, text_response, use_fake_runtime, withdrawn)

INVENTED = "Je recopie tout ce que l'IA me donne sans rien relire."
BAD = (3, 7)  # 2 signaux fautifs sur 10


def corrupted_signals(indices=BAD) -> dict:
    output = copy.deepcopy(si.GOOD_SIGNALS)
    for i in indices:
        output["signals"][i]["evidence"][0]["quote"] = INVENTED
    return output


def good_signal_for(params) -> dict:
    """Le faux modèle répare l'objet : il reprend la citation exacte (signal de référence au même rang)."""
    index = int(params["messages"][0]["content"].split("objet n° ", 1)[1].split(" ", 1)[0]) - 1
    return si.GOOD_SIGNALS["signals"][index]


def analysis_file(run: dict, name: str, interview_id: str = si.INTERVIEW_ID) -> dict:
    info = next(f for f in run["files"] if f["ingestion"]["interview_id"] == interview_id)
    return json.loads((Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR / name).read_text(encoding="utf-8"))


VOLATILE = {"generated_at", "validated_at", "created_at", "updated_at", "duration_seconds", "usage", "request_id",
            "cache_key", "api_calls", "response_model", "stop_reason", "attempts"}


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def run_stage3(tmp_path, monkeypatch, responders, files=None, name="run"):
    ollama = use_fake_runtime(monkeypatch, responders)
    run = lp.run_stage("3", si.make_ingested_run(tmp_path / name, files))
    return ollama, run["metadata"], run["status"]


GOOD_PRACTICE = {PRACTICE: lambda p: text_response(si.GOOD_PRACTICES)}


# 1 et 2 -------------------------------------------------------------------------------------------------------

def test_ten_objects_with_two_invalid_give_exactly_two_repairs_and_valid_ones_are_never_regenerated(tmp_path, monkeypatch):
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(corrupted_signals()),
                                                REPAIR + INTERACTION: lambda p: repaired(good_signal_for(p))})
    assert len(ollama.calls_for(INTERACTION)) == 1  # aucune régénération complète
    repairs = ollama.calls_for(REPAIR + INTERACTION)
    assert len(repairs) == 2
    sent = [repair_object(c["params"]) for c in repairs]
    assert sent == [corrupted_signals()["signals"][i] for i in BAD]  # seuls les objets fautifs sont envoyés
    for call in repairs:  # un seul objet par réparation : aucun des 8 objets valides n'est redemandé
        content = call["params"]["messages"][0]["content"]
        assert content.count("<objet>") == 1
    document = analysis_file(run, "interaction_signals.json")
    assert [s["surface_form"] for s in document["signals"]] == [s["surface_form"] for s in si.GOOD_SIGNALS["signals"]]
    assert analysis_file(run, config.EVIDENCE_VALIDATION_FILENAME)["total_invalid_evidence"] == 0
    [record] = [c for c in status["calls"] if c["agent"] == INTERACTION]
    assert (record["objects_total"], record["objects_valid_initial"], record["objects_repaired"]) == (10, 8, 2)


# 3 ---------------------------------------------------------------------------------------------------------------

def test_an_invalid_quote_gets_a_targeted_repair_with_a_minimal_context(tmp_path, monkeypatch):
    def chunk_reader(params):
        reply = json.loads(L.simulated_chunk_reader(params).text)
        if reply["signals"] and any(t["turn_id"] == L.ltid(200) for t in L.sent_turns(params)):
            reply["signals"][0]["evidence"][0]["quote"] = INVENTED
        return text_response(reply)

    fixed = {}

    def repair(params):
        obj = repair_object(params)
        turns = {t["turn_id"]: t["text"] for t in repair_turns(params)}
        evidence = obj["evidence"][0]
        assert evidence["turn_id"] in turns  # le tour cité est fourni, texte exact
        fixed["quote"] = turns[evidence["turn_id"]][:40]
        return repaired({**obj, "evidence": [{**evidence, "quote": fixed["quote"]}], "surface_form": fixed["quote"]})

    ollama, run, _ = run_stage3(tmp_path, monkeypatch, {PRACTICE: lambda p: text_response({"practices": [],
                                                                               "extraction_notes": None}),
                                           INTERACTION: chunk_reader, LONG_DISTANCE: L.simulated_long_distance_reader,
                                           REPAIR + INTERACTION: repair}, files=L.long_files())
    [call] = ollama.calls_for(REPAIR + INTERACTION)
    content = call["params"]["messages"][0]["content"]
    assert "QUOTE_NOT_FOUND" in content and "citation 1" in content  # erreur exacte du validateur
    chunk = next(c for c in ollama.calls_for(INTERACTION) if L.ltid(200) in c["params"]["messages"][0]["content"])
    assert len(content) < len(chunk["params"]["messages"][0]["content"]) / 3  # pas tout le bloc d'entretien
    assert len(repair_turns(call["params"])) <= stage3_repair.REPAIR_MAX_TURNS
    assert call["payload"]["format"] == stage3_repair.repair_schema(INTERACTION_SPEC)  # schéma de l'objet attendu
    assert call["params"]["system"][0]["text"] == chunk["params"]["system"][0]["text"]  # prompt de l'agent inchangé
    assert analysis_file(run, config.EVIDENCE_VALIDATION_FILENAME, L.LONG_INTERVIEW_ID)["total_invalid_evidence"] == 0


# 4 ---------------------------------------------------------------------------------------------------------------

def test_an_object_without_any_supporting_passage_is_withdrawn_never_invented(tmp_path, monkeypatch):
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(corrupted_signals((3,))),
                                                REPAIR + INTERACTION: withdrawn()})
    assert len(ollama.calls_for(REPAIR + INTERACTION)) == 1 and len(ollama.calls_for(INTERACTION)) == 1
    document = analysis_file(run, "interaction_signals.json")
    assert len(document["signals"]) == 9 and INVENTED not in json.dumps(document, ensure_ascii=False)
    assert analysis_file(run, config.EVIDENCE_VALIDATION_FILENAME)["total_invalid_evidence"] == 0
    [record] = [c for c in status["calls"] if c["agent"] == INTERACTION]
    assert record["objects_rejected"] == 1 and record["citations_withdrawn"] == 1
    assert record["rejected_objects"][0]["status"] == stage3_repair.WITHDRAWN


def test_a_repair_that_invents_a_turn_is_refused_and_the_object_stays_flagged(tmp_path, monkeypatch):
    def invent(params):
        obj = repair_object(params)
        evidence = {**obj["evidence"][0], "turn_id": f"{si.INTERVIEW_ID}_T9999"}
        return repaired({**obj, "evidence": [evidence]})

    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(corrupted_signals((3,))),
                                                REPAIR + INTERACTION: invent})
    assert len(ollama.calls_for(REPAIR + INTERACTION)) == 2  # tentatives bornées
    document = analysis_file(run, "interaction_signals.json")
    assert "T9999" not in json.dumps(document)  # aucun tour inventé dans la sortie
    assert analysis_file(run, config.EVIDENCE_VALIDATION_FILENAME)["total_invalid_evidence"] == 1  # toujours signalée
    [record] = [c for c in status["calls"] if c["agent"] == INTERACTION]
    assert record["objects_unresolved"] == 1 and record["citations_invalid_after_repair"] == 1


# 5 et 6 ---------------------------------------------------------------------------------------------------------

def test_an_unreadable_root_json_is_the_only_case_of_full_regeneration(tmp_path, monkeypatch):
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE,
                                                INTERACTION: ['{"signals": [', text_response(si.GOOD_SIGNALS)]})
    assert len(ollama.calls_for(INTERACTION)) == 2 and not ollama.calls_for(REPAIR + INTERACTION)
    [record] = [c for c in status["calls"] if c["agent"] == INTERACTION]
    assert record["full_generations"] == 2 and record["repairs"] == 0
    assert run["stage3_local"]["status"] == lp.STAGE_COMPLETE


def test_local_methodological_and_schema_errors_never_trigger_a_full_regeneration(tmp_path, monkeypatch):
    practices = copy.deepcopy(si.GOOD_PRACTICES)
    first = practices["practices"][0]
    first["turn_start"], first["turn_end"] = first["turn_end"], first["turn_start"]  # intervalle inversé (méthode)
    if first["turn_start"] == first["turn_end"]:
        first["turn_end"] = f"{si.INTERVIEW_ID}_T0001"
    practices["practices"][1]["use_status"] = "statut_inconnu"  # valeur hors schéma dans UN objet
    good = si.GOOD_PRACTICES["practices"]

    def fix(params):
        obj = repair_object(params)
        index = int(params["messages"][0]["content"].split("objet n° ", 1)[1].split(" ", 1)[0]) - 1
        return repaired(good[index] if obj != good[index] else obj)

    ollama, run, status = run_stage3(tmp_path, monkeypatch, {PRACTICE: text_response(practices), REPAIR + PRACTICE: fix,
                                                INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)})
    assert len(ollama.calls_for(PRACTICE)) == 1 and len(ollama.calls_for(REPAIR + PRACTICE)) == 2
    messages = [c["params"]["messages"][0]["content"] for c in ollama.calls_for(REPAIR + PRACTICE)]
    assert any("INVALID_TURN_RANGE" in m for m in messages) and any("SCHEMA_VALIDATION" in m for m in messages)
    document = analysis_file(run, "practice_extractor.json")
    assert document["status"] == "SUCCESS" and len(document["practices"]) == len(good)


# 7 ---------------------------------------------------------------------------------------------------------------

def test_an_interruption_during_repairs_resumes_with_validated_objects_kept(tmp_path, monkeypatch):
    run = si.make_ingested_run(tmp_path)
    run_dir = Path(run["output_dir"])
    state = {"repairs": 0}

    def flaky_repair(params):
        state["repairs"] += 1
        if state["repairs"] == 2:  # Ollama s'arrête pendant la 2e réparation
            raise agent_error("OLLAMA_UNAVAILABLE", "http://localhost:11434")
        return repaired(good_signal_for(params))

    ollama = use_fake_runtime(monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(corrupted_signals()),
                                            REPAIR + INTERACTION: flaky_repair})
    first = local_jobs.start_run_job(run_dir, "3")
    assert first["state"] != local_jobs.COMPLETE
    partial = list((run_dir / "local_runs" / "stage3" / "partial").glob("*.json"))
    assert len(partial) == 1  # état conservé : 8 objets valides + 1 réparé, 1 à traiter
    saved = json.loads(partial[0].read_text(encoding="utf-8"))
    assert [i["status"] for i in saved["items"]].count(stage3_repair.VALID) == 8
    assert [i["status"] for i in saved["items"]].count(stage3_repair.REPAIRED) == 1
    assert [i["status"] for i in saved["items"]].count(stage3_repair.PENDING) == 1

    resumed_ollama = use_fake_runtime(monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(corrupted_signals()),
                                                    REPAIR + INTERACTION: lambda p: repaired(good_signal_for(p))})
    second = local_jobs.start_run_job(run_dir, "3")
    assert second["state"] == local_jobs.COMPLETE
    assert not resumed_ollama.calls_for(INTERACTION)  # pas de nouvelle génération du bloc
    [only] = resumed_ollama.calls_for(REPAIR + INTERACTION)  # seul l'objet encore fautif
    assert repair_object(only["params"]) == corrupted_signals()["signals"][BAD[1]]
    assert not list((run_dir / "local_runs" / "stage3" / "partial").glob("*.json"))  # appel abouti : état supprimé
    final = json.loads((Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR /
                        config.EVIDENCE_VALIDATION_FILENAME).read_text(encoding="utf-8"))
    assert final["total_invalid_evidence"] == 0 and ollama.calls_for(INTERACTION)


# 8 et 9 ---------------------------------------------------------------------------------------------------------

def test_repaired_outputs_are_identical_to_outputs_that_needed_no_repair(tmp_path, monkeypatch):
    _, repaired_run, _ = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(corrupted_signals()),
                                               REPAIR + INTERACTION: lambda p: repaired(good_signal_for(p))},
                                    name="repaired")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "other_cache")  # aucune réponse reprise du cache
    _, clean_run, _ = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)},
                                 name="clean")
    for name in ("practice_extractor.json", "interaction_signals.json", config.EVIDENCE_VALIDATION_FILENAME):
        assert stable(analysis_file(repaired_run, name)) == stable(analysis_file(clean_run, name)), name


def test_stage4_accepts_a_repaired_stage3_normally(tmp_path, monkeypatch):
    signals = copy.deepcopy(S4.STAGE3_SIGNALS)
    original = signals["signals"][0]
    signals["signals"][0] = {**original, "evidence": [{**original["evidence"][0], "quote": INVENTED}]}

    def run_both(name, responders):
        ollama = use_fake_runtime(monkeypatch, {**responders, "accountability_episode_builder": S4.REFERENCE_BUILDER,
                                                "speaker_attribution_auditor": lambda p: text_response(S4.STAGE3_AUDIT)})
        run = si.make_ingested_run(tmp_path / name, S4.FILES)
        result = lp.run_until(run, "4")
        return ollama, result

    ollama, repaired_result = run_both("repaired", {
        PRACTICE: lambda p: text_response(S4.STAGE3_PRACTICES), INTERACTION: text_response(signals),
        REPAIR + INTERACTION: lambda p: repaired(original)})
    assert ollama.calls_for(REPAIR + INTERACTION) and len(ollama.calls_for(INTERACTION)) == 1
    _, clean_result = run_both("clean", S4.stage3_responders(audit=False))
    assert repaired_result["stage"] == "4" and repaired_result["status"]["status"] == lp.STAGE_COMPLETE
    episodes = [analysis_file(r["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME, S4.INTERVIEW_ID)
                for r in (repaired_result, clean_result)]
    assert stable(episodes[0]["episodes"]) == stable(episodes[1]["episodes"])


# 10 --------------------------------------------------------------------------------------------------------------

def test_metrics_separate_initial_generation_and_targeted_repairs(tmp_path, monkeypatch):
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(corrupted_signals()),
                                                REPAIR + INTERACTION: lambda p: repaired(good_signal_for(p))})
    [record] = [c for c in status["calls"] if c["agent"] == INTERACTION]
    assert record["full_generations"] == 1 and record["repairs"] == 2
    assert record["initial_output_tokens"] > record["repair_output_tokens"] > 0
    assert record["repair_input_tokens"] > 0 and record["repair_seconds"] >= 0
    assert record["citations_invalid_initial"] == 2 and record["citations_repaired"] == 2
    assert record["citations_invalid_after_repair"] == 0
    journal = json.loads((Path(run["output_dir"]) / "local_runs" / "stage3" / f"{record['call_id']}.json")
                         .read_text(encoding="utf-8"))
    assert [a["phase"] for a in journal["attempt_log"]] == ["initial", "repair", "repair"]
    summary = repair_summary(list(status["calls"]), run)
    assert summary["targeted_repairs"] == 2 and summary["full_regenerations"] == 0
    assert summary["citations_invalid_initial"] == 2 and summary["final_evidence"]["total_invalid_evidence"] == 0
