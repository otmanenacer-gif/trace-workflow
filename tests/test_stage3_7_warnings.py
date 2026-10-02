"""Étape 3.7 — les cinq catégories d'avertissements observées sur le vrai entretien long.

Pour chacune : (1) un objet CORRECT ne la déclenche plus (cause évitable corrigée) ;
(2) un objet INCORRECT la déclenche toujours (le contrôle reste actif). agents simulés uniquement.
"""

import json
from pathlib import Path

import pytest

from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from core.evidence_validator import validate_agent_output
from tests import synthetic_interviews as si
from tests import synthetic_stage37 as S
from tests.fake_llm import AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, FakeAgents, fake_settings, text_response

TRANSCRIPT = {"interview_id": "W", "turns": [
    {"turn_id": "W_T0001", "speaker": "enqueteur", "text": "Tu as peur que tes profs le voient ?"},
    {"turn_id": "W_T0002", "speaker": "enquete", "text": "Oui. J'ai peur que le prof pense que j'ai triché."},
    {"turn_id": "W_T0003", "speaker": "enqueteur", "text": "Moi personnellement je lui fais juste reformuler."},
    {"turn_id": "W_T0004", "speaker": "enquete",
     "text": "Je l'utilise pour réviser. Bon, c'est un peu facile ce que je dis, je sais. Là j'abuse."},
    {"turn_id": "W_T0005", "speaker": "enqueteur", "text": "Et la règle ?"},
    {"turn_id": "W_T0006", "speaker": "enquete",
     "text": "ChatGPT est ridicule. Cette règle est absurde. Ce devoir est nul. Je sais que dit comme ça c'est bizarre. "
             "J'étais stressée et ça m'énerve."},
]}


def sig(turns, signal_type="other", quotes=(), **fields):
    return si.signal(turn_ids=[f"W_T{t:04d}" for t in turns], signal_type=signal_type, surface_form=fields.pop("surface", "x"),
                     description=fields.pop("description", "L'enquêté dit cela."),
                     evidence=[{"turn_id": f"W_T{t:04d}", "quote": q} for t, q in quotes], **fields)


def codes(item, warnings=None, agent="interaction_signal_reader"):
    result = validate_agent_output(agent, [item], TRANSCRIPT, "S" if agent != "practice_extractor" else "P", warnings)
    return set(result["items"][0]["review_reasons"])


# --- NO_INTERVIEWEE_EVIDENCE ---------------------------------------------------------------------

def test_no_interviewee_evidence_not_raised_for_audited_interviewer_turn():
    item = sig([3], "restriction", [(3, "je lui fais juste reformuler")])
    warned = {"W_T0003": {"suggested_speaker": "enquete", "confidence": "high"}}
    reasons = codes(item, warned)
    assert "NO_INTERVIEWEE_EVIDENCE" not in reasons and "EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN" in reasons  # à revoir
    assert "NO_INTERVIEWEE_EVIDENCE" in codes(item)  # sans avertissement de l'audit : toujours signalé
    question = si.practice(summary="Il dit cela.", turn_start="W_T0001", turn_end="W_T0001",
                           evidence=[{"turn_id": "W_T0001", "quote": "Tu as peur que tes profs le voient ?"}])
    assert "NO_INTERVIEWEE_EVIDENCE" in codes(question, agent="practice_extractor")


# --- Pipeline : entretien long, lecteur qui surcode (question seule, turn_ids incomplets) ----------

@pytest.fixture
def long_outputs(tmp_path):
    """Fonction (pas module) : chaque test s'exécute sous les garde-fous de tests/conftest.py (aucun appel réel)."""
    run = si.make_ingested_run(tmp_path, S.files())
    transport = FakeAgents({PRACTICE: S.practice_reader, INTERACTION: S.naive_reader,
                               LONG_DISTANCE: S.long_distance_reader,
                               AUDITOR: lambda p: text_response({"assessments": [], "audit_notes": None})})
    run = analyze_run(run, settings=fake_settings(), client=transport, cache=AnalysisCache(tmp_path / "cache"))
    directory = Path(run["files"][0]["analysis"]["analysis_dir"])
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))  # noqa: E731
    return {"signals": read("interaction_signals.json"), "practices": read("practice_extractor.json"),
            "validation": read("evidence_validation.json")}


def all_codes(outputs):
    return {i["code"] for a in outputs["validation"]["agents"].values() for i in a["issues"]}


def test_avoidable_observed_warnings_are_not_produced(long_outputs):
    observed = {"NO_INTERVIEWEE_EVIDENCE", "EVIDENCE_TURN_NOT_LISTED", "METADISCURSIVE_NO_SELF_REFERENCE",
                "AFFECT_NOT_IN_QUOTES", "INTERPRETIVE_VOCABULARY"}
    assert not observed & all_codes(long_outputs)
    for agent in long_outputs["validation"]["agents"].values():
        assert agent["warning_count"] == agent["error_count"] == 0


def test_interviewer_question_is_not_turned_into_a_signal(long_outputs):
    doc = long_outputs["signals"]
    leading = S.tid(S.LEADING_QUESTION_TURN)
    assert not [s for s in doc["signals"] if s["turn_ids"] == [leading]]
    aside, = [s for s in doc["set_aside_signals"] if s["set_aside_reason"] == "INTERVIEWER_ONLY_EVIDENCE"]
    assert aside["turn_ids"] == [leading]
    # le tour T0201 (marqué enquêteur, signalé par l'audit) reste un appui, à revoir
    warned = next(s for s in doc["signals"] if s["turn_ids"] == [S.tid(201)])
    assert warned["needs_review"] and "EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN" in warned["review_reasons"]
    assert "NO_INTERVIEWEE_EVIDENCE" not in warned["review_reasons"]


# --- EVIDENCE_TURN_NOT_LISTED ----------------------------------------------------------------------

def test_cited_turns_are_listed_after_selectivity_and_merge(long_outputs):
    for signal in long_outputs["signals"]["signals"]:
        assert {e["turn_id"] for e in signal["evidence"]} <= set(signal["turn_ids"])
    completed = next(s for s in long_outputs["signals"]["signals"] if s.get("turn_ids_added"))
    assert completed["turn_ids"] == [S.tid(119), S.tid(120)] and completed["turn_ids_added"] == [S.tid(119)]
    assert long_outputs["signals"]["selectivity"]["signals_with_turn_ids_completed"] == 1


def test_short_interview_signal_with_unlisted_cited_turn_is_completed(tmp_path):
    output = {"signals": [si.signal(turn_ids=[si.tid(12)], signal_type="contrast", surface_form="Mais",
                                    description="« Mais » oppose deux passages.",
                                    evidence=[si.ev(11, "Tu disais que tu l'utilisais pour reformuler ?"),
                                              si.ev(12, "Mais le plan de la dissert, c'est moi qui l'ai fait")])],
              "reading_notes": None}
    run = si.make_ingested_run(tmp_path)
    transport = FakeAgents({PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
                               INTERACTION: lambda p: text_response(output)})
    run = analyze_run(run, settings=fake_settings(), client=transport, cache=AnalysisCache(tmp_path / "cache"))
    doc = json.loads((Path(run["files"][0]["analysis"]["analysis_dir"]) / "interaction_signals.json").read_text())
    signal, = doc["signals"]
    assert signal["turn_ids"] == [si.tid(11), si.tid(12)] and signal["review_reasons"] == []
    assert doc["status"] == "SUCCESS"


def test_unlisted_turn_is_still_flagged_by_the_validator_itself():
    item = sig([2], "contrast", [(2, "Oui."), (4, "Là j'abuse.")])
    assert "EVIDENCE_TURN_NOT_LISTED" in codes(item)


# --- METADISCURSIVE_NO_SELF_REFERENCE ---------------------------------------------------------------

@pytest.mark.parametrize("turn, quote", [
    (4, "c'est un peu facile ce que je dis"),
    (4, "c'est un peu facile"),            # citation plus courte que la phrase qui dit « ce que je dis »
    (4, "Là j'abuse."),
    (6, "Je sais que dit comme ça c'est bizarre."),
])
def test_metadiscursive_with_self_reference_is_not_flagged(turn, quote):
    assert "METADISCURSIVE_NO_SELF_REFERENCE" not in codes(sig([turn], "metadiscursive_self_evaluation", [(turn, quote)]))


@pytest.mark.parametrize("quote", ["ChatGPT est ridicule.", "Cette règle est absurde.", "Ce devoir est nul."])
def test_evaluation_of_something_else_is_still_flagged(quote):
    assert "METADISCURSIVE_NO_SELF_REFERENCE" in codes(sig([6], "metadiscursive_self_evaluation", [(6, quote)]))


# --- AFFECT_NOT_IN_QUOTES ----------------------------------------------------------------------------

@pytest.mark.parametrize("affect, turn, quote", [
    ("peur", 2, "J'ai peur que le prof pense que j'ai triché."),
    ("j'avais peur", 2, "J'ai peur"),        # même affect, auxiliaire différent
    ("stressée", 6, "J'étais stressée"),
    ("stressé", 6, "J'étais stressée"),
    ("énervement", 6, "ça m'énerve"),
])
def test_explicit_affect_present_in_quotes_is_not_flagged(affect, turn, quote):
    assert "AFFECT_NOT_IN_QUOTES" not in codes(sig([turn], "explicit_emotion", [(turn, quote)], explicit_affect=affect))


@pytest.mark.parametrize("affect", ["honte", "culpabilité", "gêne"])
def test_affect_absent_from_quotes_is_still_flagged(affect):
    item = sig([2], "explicit_emotion", [(2, "Oui.")], explicit_affect=affect)
    assert "AFFECT_NOT_IN_QUOTES" in codes(item)


# --- INTERPRETIVE_VOCABULARY ----------------------------------------------------------------------------

def test_normal_descriptive_fields_are_not_flagged(long_outputs):
    for key, items in (("signals", long_outputs["signals"]["signals"]), ("practices", long_outputs["practices"]["practices"])):
        assert not [i for i in items if "INTERPRETIVE_VOCABULARY" in i["review_reasons"]], key
    item = sig([2], "explicit_emotion", [(2, "J'ai peur que le prof pense que j'ai triché.")], explicit_affect="peur",
               description="L'enquêté dit avoir peur que le professeur pense qu'il a triché.")
    assert "INTERPRETIVE_VOCABULARY" not in codes(item)  # « peur », « triché » : mots de l'enquêté


@pytest.mark.parametrize("description, term", [
    ("Travail de légitimation de son usage.", "légitimation"),
    ("Normalisation de l'usage de l'outil.", "normalisation"),
    ("Une défense identitaire.", "défense"),
    ("Stratégie de justification.", "stratégie"),
    ("Il rationalise son usage.", "rationalisation"),
    ("Réparation identitaire après l'aveu.", "réparation"),
    ("Son identité étudiante est en jeu.", "identité"),
    ("Moment d'accountability.", "accountability"),
])
def test_interpretive_vocabulary_is_still_flagged(description, term):
    result = validate_agent_output("interaction_signal_reader",
                                   [sig([4], "other", [(4, "Là j'abuse.")], description=description)], TRANSCRIPT, "S")
    terms = {i["term"] for i in result["report"]["issues"] if i["code"] == "INTERPRETIVE_VOCABULARY"}
    assert term in terms, terms


def test_practice_summary_with_analytic_category_is_flagged():
    item = si.practice(summary="Il présente une normalisation de son usage, un travail de légitimation.",
                       turn_start="W_T0004", turn_end="W_T0004",
                       evidence=[{"turn_id": "W_T0004", "quote": "Je l'utilise pour réviser."}])
    result = validate_agent_output("practice_extractor", [item], TRANSCRIPT, "P")
    assert {"normalisation", "légitimation"} <= {i["term"] for i in result["report"]["issues"]
                                                 if i["code"] == "INTERPRETIVE_VOCABULARY"}
