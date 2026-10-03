"""Étape 3 : corrections DÉTERMINISTES des citations (core/citation_resolver.py), au plus UNE réparation par le modèle
et par objet pour les autres anomalies locales (core/stage3_repair.py), jamais de régénération complète d'une réponse
lisible. Faux Ollama derrière le vrai LocalAgentRunner et le vrai MethodChecker : aucun modèle, aucun réseau."""

import copy
import json
from pathlib import Path

from core import citation_resolver as cr
from core import config, local_jobs
from core import local_pipeline as lp
from core import stage3_repair
from scripts.trace_benchmark import repair_summary
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests.fake_llm import (INTERACTION, PRACTICE, REPAIR, agent_error, repair_object, repaired, text_response,
                            use_fake_runtime)

IID = si.INTERVIEW_ID
INVENTED = "Je recopie tout ce que l'IA me donne sans rien relire."
GOOD_PRACTICE = {PRACTICE: lambda p: text_response(si.GOOD_PRACTICES)}
GOOD_SIGNAL = {INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)}


def tid(number: int, interview_id: str = IID) -> str:
    return f"{interview_id}_T{number:04d}"


def signals_with(changes: dict) -> dict:
    """Signaux de référence dont la citation 1 de certains signaux est modifiée : {rang: {"turn_id"?, "quote"?}}."""
    output = copy.deepcopy(si.GOOD_SIGNALS)
    for index, fields in changes.items():
        output["signals"][index]["evidence"][0].update(fields)
    return output


def analysis_file(run: dict, name: str, interview_id: str = IID) -> dict:
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
    result = lp.run_stage("3", si.make_ingested_run(tmp_path / name, files))
    return ollama, result["metadata"], result["status"]


def record_of(status: dict, agent: str) -> dict:
    [record] = [c for c in status["calls"] if c["agent"] == agent]
    return record


# --- CitationResolver seul ----------------------------------------------------------------------------------------

TURNS = [{"turn_id": "A_T0001", "text": "Euh… je sais pas, si la prof voyait ça, elle trouverait peut-être que c'est "
                                        "pas mon travail."},
         {"turn_id": "A_T0002", "text": "Je lis moi-même les textes. Je lis moi-même les textes, vraiment."},
         {"turn_id": "A_T0003", "text": "Pour reformuler mes phrases quand elles sont trop lourdes, en fait."},
         {"turn_id": "A_T0004", "text": "Oui, pour reformuler mes phrases quand elles sont trop lourdes."}]
ALL = {t["turn_id"] for t in TURNS}


def test_resolver_restores_the_literal_passage_and_never_rewrites_words():
    fixed = cr.resolve("si la prof voyait ça elle trouverait peut être que c’est PAS mon travail", "A_T0001", TURNS, ALL)
    assert fixed == {"decision": cr.FIXED_IN_TURN, "turn_id": "A_T0001",
                     "quote": "si la prof voyait ça, elle trouverait peut-être que c'est pas mon travail"}
    assert cr.resolve("si la prof voyait ça, elle jugerait", "A_T0001", TURNS, ALL)["decision"] == cr.NOT_FOUND
    assert cr.resolve("prof voyait", "A_T0001", TURNS, ALL)["quote"] == "prof voyait"
    assert cr.resolve("rof voyai", "A_T0001", TURNS, ALL)["decision"] != cr.FIXED_IN_TURN  # jamais dans un mot


def test_resolver_refuses_ambiguous_short_or_unauthorised_matches():
    same = cr.resolve("je lis moi même les textes", "A_T0002", TURNS, ALL)  # deux fois le MÊME passage littéral
    assert same == {"decision": cr.FIXED_IN_TURN, "turn_id": "A_T0002", "quote": "Je lis moi-même les textes"}
    both = cr.resolve("pour reformuler mes phrases quand elles sont trop lourdes", "A_T0001", TURNS, ALL)
    assert both == {"decision": cr.AMBIGUOUS, "candidates": ["A_T0003", "A_T0004"]}  # aucun choix arbitraire
    assert cr.resolve("trop lourdes", "A_T0001", TURNS, ALL)["decision"] == cr.TOO_SHORT  # ailleurs : ≥ 4 mots
    hidden = cr.resolve("elle trouverait peut-être que", "A_T0003", TURNS, ALL - {"A_T0001"})
    assert hidden["decision"] == cr.NOT_FOUND  # tour hors du matériau envoyé : jamais cité
    assert cr.resolve("", "A_T0001", TURNS, ALL)["decision"] == cr.TOO_SHORT


# --- Étape 3 : citations corrigées sans modèle --------------------------------------------------------------------

def test_exact_quote_with_a_wrong_turn_id_is_repaired_automatically(tmp_path, monkeypatch):
    responses = signals_with({3: {"turn_id": tid(4)}, 5: {"turn_id": tid(99)}})  # tour existant / inexistant
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(responses)})
    assert len(ollama.calls_for(INTERACTION)) == 1 and not ollama.calls_for(REPAIR + INTERACTION)
    signals = analysis_file(run, "interaction_signals.json")["signals"]
    assert signals[3]["evidence"][0]["turn_id"] == tid(6) and signals[5]["evidence"][0]["turn_id"] == tid(8)
    assert analysis_file(run, config.EVIDENCE_VALIDATION_FILENAME)["total_invalid_evidence"] == 0
    record = record_of(status, INTERACTION)
    assert record["citations_fixed_deterministic"] == 2 and record["llm_repairs_attempted"] == 0
    assert {f["decision"] for f in record["citation_fixes"]} == {cr.FIXED_OTHER_TURN}  # corrections journalisées
    assert record["citation_fixes"][0]["before"]["turn_id"] == tid(4)


def test_punctuation_spacing_and_case_differences_restore_the_exact_quote(tmp_path, monkeypatch):
    exact = si.GOOD_SIGNALS["signals"][3]["evidence"][0]["quote"]
    altered = "  " + exact.replace("'", "’").replace(",", " ").upper() + " !"
    ollama, run, status = run_stage3(tmp_path, monkeypatch,
                                     {**GOOD_PRACTICE, INTERACTION: text_response(signals_with({3: {"quote": altered}}))})
    assert not ollama.calls_for(REPAIR + INTERACTION)
    assert analysis_file(run, "interaction_signals.json")["signals"][3]["evidence"][0]["quote"] == exact
    assert record_of(status, INTERACTION)["citation_fixes"][0]["decision"] == cr.FIXED_IN_TURN


def test_a_quote_found_in_a_single_other_authorised_turn_is_moved_there(tmp_path, monkeypatch):
    quote = "je pourrais lui demander un résumé"  # tour 14, annoncé au tour 12
    responses = signals_with({8: {"quote": quote}})
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(responses)})
    signal = analysis_file(run, "interaction_signals.json")["signals"][8]
    assert signal["evidence"][0] == {**signal["evidence"][0], "turn_id": tid(14), "quote": quote}
    assert tid(14) in signal["turn_ids"]  # mise en forme habituelle : tour cité ajouté à turn_ids
    assert not ollama.calls_for(REPAIR + INTERACTION)


AMBIGUOUS_TEXT = ("Enquêteur : Tu l'utilises comment ?\n"
                  "Enquêté : Je lui demande de corriger mes fautes d'orthographe, c'est tout.\n"
                  "Enquêteur : Et pour les mémoires ?\n"
                  "Enquêté : Là aussi, je lui demande de corriger mes fautes d'orthographe.\n"
                  "Enquêteur : D'accord.\n"
                  "Enquêté : Voilà.\n")


def test_two_possible_matches_give_no_arbitrary_correction(tmp_path, monkeypatch):
    signal = {**si.GOOD_SIGNALS["signals"][5], "turn_ids": [tid(6, "ENTRETIEN_AMBIGU")],
              "evidence": [{"turn_id": tid(6, "ENTRETIEN_AMBIGU"), "quote": "je lui demande de corriger mes fautes"}]}
    ollama, run, status = run_stage3(
        tmp_path, monkeypatch, {PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
                                INTERACTION: text_response({"signals": [signal], "reading_notes": None})},
        files=[("Entretien_ambigu.txt", AMBIGUOUS_TEXT.encode("utf-8"))])
    assert not ollama.calls_for(REPAIR + INTERACTION)
    kept = analysis_file(run, "interaction_signals.json", "ENTRETIEN_AMBIGU")["signals"][0]
    assert kept["evidence"][0]["turn_id"] == tid(6, "ENTRETIEN_AMBIGU")  # citation laissée telle quelle
    assert kept["evidence"][0]["validation"]["valid"] is False and kept["needs_review"] is True
    record = record_of(status, INTERACTION)
    assert record["citations_ambiguous"] == 1 and record["citation_fixes"][0]["candidates"] == [
        tid(2, "ENTRETIEN_AMBIGU"), tid(4, "ENTRETIEN_AMBIGU")]


def test_an_invented_quote_is_never_saved_and_the_object_follows_existing_rules(tmp_path, monkeypatch):
    responses = signals_with({3: {"quote": INVENTED}})
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(responses)})
    assert len(ollama.calls_for(INTERACTION)) == 1 and not ollama.calls_for(REPAIR + INTERACTION)  # aucune boucle
    signal = analysis_file(run, "interaction_signals.json")["signals"][3]
    assert signal["evidence"][0]["quote"] == INVENTED and signal["evidence"][0]["validation"]["valid"] is False
    assert "NO_VALID_EVIDENCE" in signal["review_reasons"]  # signalé, puis écarté par l'étape 4 (règle existante)
    record = record_of(status, INTERACTION)
    assert (record["citations_not_found"], record["objects_rejected_no_evidence"], record["citations_invalid_final"]) \
        == (1, 1, 1)


# --- Autres anomalies locales : au plus UNE réparation par le modèle ---------------------------------------------

def inverted_range(practices: dict, index: int) -> dict:
    practices = copy.deepcopy(practices)
    practices["practices"][index]["turn_start"], practices["practices"][index]["turn_end"] = tid(14), tid(2)
    return practices


def test_another_local_error_gets_at_most_one_model_repair(tmp_path, monkeypatch):
    broken = inverted_range(si.GOOD_PRACTICES, 1)
    broken["practices"][2]["use_status"] = "statut_inconnu"  # hors schéma dans UN objet
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {PRACTICE: text_response(broken), **GOOD_SIGNAL})
    # le faux modèle renvoie l'objet sans le corriger : une seule tentative chacun, jamais une 2e
    repairs = ollama.calls_for(REPAIR + PRACTICE)
    assert len(ollama.calls_for(PRACTICE)) == 1 and len(repairs) == 2
    assert sorted(repair_object(c["params"])["summary"] for c in repairs) == sorted(
        broken["practices"][i]["summary"] for i in (1, 2))
    practices = analysis_file(run, "practice_extractor.json")["practices"]
    assert len(practices) == len(si.GOOD_PRACTICES["practices"]) - 1  # objet hors schéma écarté
    assert "INVALID_TURN_RANGE" in practices[1]["review_reasons"]  # objet d'origine conservé, signalé
    record = record_of(status, PRACTICE)
    assert record["llm_repairs_attempted"] == 2 and record["llm_repairs_succeeded"] == 0
    assert record["objects_dropped_schema"] == 1


def test_a_successful_model_repair_is_kept(tmp_path, monkeypatch):
    good = si.GOOD_PRACTICES["practices"][1]
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {PRACTICE: text_response(inverted_range(si.GOOD_PRACTICES, 1)),
                                                             REPAIR + PRACTICE: repaired(good), **GOOD_SIGNAL})
    assert len(ollama.calls_for(REPAIR + PRACTICE)) == 1
    assert "INVALID_TURN_RANGE" in ollama.calls_for(REPAIR + PRACTICE)[0]["params"]["messages"][0]["content"]
    assert analysis_file(run, "practice_extractor.json")["status"] == "SUCCESS"
    assert record_of(status, PRACTICE)["llm_repairs_succeeded"] == 1


def test_an_unreadable_root_json_is_the_only_case_of_full_regeneration(tmp_path, monkeypatch):
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {
        **GOOD_PRACTICE, INTERACTION: ['{"signals": [', text_response(si.GOOD_SIGNALS)]})
    assert len(ollama.calls_for(INTERACTION)) == 2 and not ollama.calls_for(REPAIR + INTERACTION)
    record = record_of(status, INTERACTION)
    assert record["full_generations"] == 2 and record["repairs"] == 0
    assert run["stage3_local"]["status"] == lp.STAGE_COMPLETE


# --- Persistance et reprise ---------------------------------------------------------------------------------------

def test_an_interruption_during_repairs_resumes_without_recomputing(tmp_path, monkeypatch):
    run = si.make_ingested_run(tmp_path)
    run_dir = Path(run["output_dir"])
    broken = inverted_range(inverted_range(si.GOOD_PRACTICES, 1), 3)
    exact = broken["practices"][0]["evidence"][0]["quote"]
    broken["practices"][0]["evidence"][0]["quote"] = exact.replace("'", "’")  # correction déterministe
    good = si.GOOD_PRACTICES["practices"]
    calls = {"n": 0}

    def flaky(params):
        calls["n"] += 1
        if calls["n"] == 2:  # Ollama s'arrête pendant la 2e réparation
            raise agent_error("OLLAMA_UNAVAILABLE", "http://localhost:11434")
        return repaired(good[1])

    use_fake_runtime(monkeypatch, {PRACTICE: text_response(broken), REPAIR + PRACTICE: flaky, **GOOD_SIGNAL})
    first = local_jobs.start_run_job(run_dir, "3")
    assert first["state"] != local_jobs.COMPLETE
    [partial] = list((run_dir / "local_runs" / "stage3" / "partial").glob("*.json"))
    saved = json.loads(partial.read_text(encoding="utf-8"))
    statuses = [i["status"] for i in saved["items"]]
    assert statuses.count(stage3_repair.RESOLVED) == 1 and statuses.count(stage3_repair.REPAIRED) == 1
    assert statuses.count(stage3_repair.PENDING) == 1  # décisions enregistrées, une réparation à faire
    assert saved["items"][0]["citation_log"][0]["after"]["quote"] == exact

    resumed = use_fake_runtime(monkeypatch, {PRACTICE: text_response(broken),
                                             REPAIR + PRACTICE: repaired(good[3]), **GOOD_SIGNAL})
    second = local_jobs.start_run_job(run_dir, "3")
    assert second["state"] == local_jobs.COMPLETE
    assert not resumed.calls_for(PRACTICE)  # pas de nouvelle génération du bloc
    assert len(resumed.calls_for(REPAIR + PRACTICE)) == 1  # seul l'objet encore à réparer
    assert not list((run_dir / "local_runs" / "stage3" / "partial").glob("*.json"))
    document = json.loads((Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR /
                           "practice_extractor.json").read_text(encoding="utf-8"))
    assert document["status"] == "SUCCESS" and document["practices"][0]["evidence"][0]["quote"] == exact


# --- Sorties habituelles, étape 4, mesures ------------------------------------------------------------------------

def test_corrected_outputs_are_identical_to_outputs_that_needed_no_correction(tmp_path, monkeypatch):
    exact = si.GOOD_SIGNALS["signals"][3]["evidence"][0]["quote"]
    responses = signals_with({3: {"quote": exact.replace("'", "’")}, 7: {"turn_id": tid(8)}})
    _, fixed_run, _ = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, INTERACTION: text_response(responses)},
                                 name="fixed")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "other_cache")  # aucune réponse reprise du cache
    _, clean_run, _ = run_stage3(tmp_path, monkeypatch, {**GOOD_PRACTICE, **GOOD_SIGNAL}, name="clean")
    for name in ("practice_extractor.json", "interaction_signals.json", config.EVIDENCE_VALIDATION_FILENAME):
        assert stable(analysis_file(fixed_run, name)) == stable(analysis_file(clean_run, name)), name


def test_stage4_accepts_a_corrected_stage3_normally(tmp_path, monkeypatch):
    signals = copy.deepcopy(S4.STAGE3_SIGNALS)
    evidence = signals["signals"][0]["evidence"][0]
    evidence["quote"] = evidence["quote"].replace("'", "’").upper()

    def run_until_4(name, responders):
        ollama = use_fake_runtime(monkeypatch, {**responders, "accountability_episode_builder": S4.REFERENCE_BUILDER,
                                                "speaker_attribution_auditor": lambda p: text_response(S4.STAGE3_AUDIT)})
        return ollama, lp.run_until(si.make_ingested_run(tmp_path / name, S4.FILES), "4")

    ollama, fixed = run_until_4("fixed", {PRACTICE: lambda p: text_response(S4.STAGE3_PRACTICES),
                                          INTERACTION: text_response(signals)})
    assert not ollama.calls_for(REPAIR + INTERACTION) and len(ollama.calls_for(INTERACTION)) == 1
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "other_cache")
    _, clean = run_until_4("clean", S4.stage3_responders(audit=False))
    assert fixed["stage"] == "4" and fixed["status"]["status"] == lp.STAGE_COMPLETE
    episodes = [analysis_file(r["metadata"], config.ACCOUNTABILITY_EPISODES_FILENAME, S4.INTERVIEW_ID)
                for r in (fixed, clean)]
    assert stable(episodes[0]["episodes"]) == stable(episodes[1]["episodes"])


def test_metrics_separate_generation_deterministic_fixes_and_model_repairs(tmp_path, monkeypatch):
    practices = inverted_range(si.GOOD_PRACTICES, 1)
    practices["practices"][0]["evidence"][0]["turn_id"] = tid(4)  # citation exacte, mauvais tour
    practices["practices"][5]["evidence"][0]["quote"] = INVENTED  # introuvable
    ollama, run, status = run_stage3(tmp_path, monkeypatch, {
        PRACTICE: text_response(practices), REPAIR + PRACTICE: repaired(si.GOOD_PRACTICES["practices"][1]),
        **GOOD_SIGNAL})
    record = record_of(status, PRACTICE)
    assert record["full_generations"] == 1 and record["repairs"] == record["llm_repairs_attempted"] == 1
    assert record["llm_repairs_succeeded"] == 1 and record["initial_output_tokens"] > record["repair_output_tokens"] > 0
    assert (record["citations_invalid_initial"], record["citations_fixed_deterministic"],
            record["citations_not_found"], record["citations_invalid_final"]) == (2, 1, 1, 1)
    journal = json.loads((Path(run["output_dir"]) / "local_runs" / "stage3" / f"{record['call_id']}.json")
                         .read_text(encoding="utf-8"))
    assert [a["phase"] for a in journal["attempt_log"]] == ["initial", "repair"]
    assert journal["citation_fixes"] and journal["llm_repairs_attempted"] == 1
    summary = repair_summary(list(status["calls"]), run)
    assert summary["llm_repairs_attempted"] == 1 and summary["citations_fixed_deterministic"] == 1
    assert summary["full_regenerations"] == 0 and summary["final_evidence"]["total_invalid_evidence"] == 1
