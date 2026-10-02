"""Tests du Speaker Attribution Auditor (étape 3.5) : présélection déterministe, appel conditionnel,
cache, transcript jamais modifié, avertissements transmis aux agents. agents simulés uniquement."""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

import core.analysis as analysis
from agents import interaction_signal_reader, practice_extractor
from core import config, speaker_attribution_auditor as sa
from core.analysis import analyze_run, plan_analysis
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests.fake_llm import AUDITOR, INTERACTION, PRACTICE, FakeAgents, fake_settings, agent_error, text_response

EMPTY_PRACTICES = {"practices": [], "extraction_notes": None}
EMPTY_SIGNALS = {"signals": [], "reading_notes": None}


def transcript_of(turns, interview_id="X"):
    return {"interview_id": interview_id, "turns": [
        {"turn_id": f"{interview_id}_T{i:04d}", "speaker": speaker, "text": text}
        for i, (speaker, text) in enumerate(turns, start=1)]}


def rules(transcript):
    return {c["turn_id"]: c["rules"] for c in sa.find_candidates(transcript)}


def responders(audit=si.AUDIT_ASSESSMENTS, practices=EMPTY_PRACTICES, signals=EMPTY_SIGNALS):
    found = {PRACTICE: lambda p: text_response(practices), INTERACTION: lambda p: text_response(signals)}
    if audit is not None:
        found[AUDITOR] = audit if callable(audit) or isinstance(audit, BaseException) else (
            lambda p: text_response(audit))
    return found


def run(tmp_path, files, cache=None, **kwargs):
    run_ = si.make_ingested_run(tmp_path, files)
    transport = FakeAgents(responders(**kwargs))
    run_ = analyze_run(run_, settings=fake_settings(), client=transport, cache=cache or AnalysisCache(tmp_path / "cache"))
    return run_, transport


def out_dir(run_):
    return Path(run_["files"][0]["analysis"]["analysis_dir"])


def load(run_, name):
    return json.loads((out_dir(run_) / name).read_text(encoding="utf-8"))


# --- A. Présélection déterministe -------------------------------------------------------------

def test_obvious_interviewer_first_person_answer_is_a_candidate():
    transcript = transcript_of([
        ("enquete", "Oui, souvent."),
        ("enqueteur", "Moi personnellement j'utilise ChatGPT tous les jours parce que ça m'aide pour mes cours."),
    ])
    candidate, = sa.find_candidates(transcript)
    assert candidate["turn_id"] == "X_T0002" and candidate["current_speaker"] == "enqueteur"
    assert candidate["rules"] == ["INTERVIEWER_FIRST_PERSON_ANSWER"]
    assert candidate["heuristic_suggestion"] == "enquete"
    assert candidate["cues"][0]["quote"] in transcript["turns"][1]["text"]  # indice = citation exacte


def test_normal_interviewer_question_is_not_a_candidate():
    assert rules(transcript_of([("enqueteur", "Est-ce que tu utilises ChatGPT ?"),
                                ("enquete", "Oui, pour reformuler.")])) == {}


@pytest.mark.parametrize("speaker", ["enqueteur", "enquete"])
def test_ambiguous_tag_question_is_not_a_candidate(speaker):
    other = "enquete" if speaker == "enqueteur" else "enqueteur"
    assert rules(transcript_of([(other, "Je l'utilise pour reformuler."),
                                (speaker, "Tu vois ce que je veux dire ?")])) == {}


def test_interviewee_interview_question_is_a_candidate():
    found = rules(transcript_of([("enqueteur", "Et pour les exposés ?"), ("enquete", "Non, jamais."),
                                 ("enquete", "Est-ce que tu l'utilises pour tes dissertations ?")]))
    assert found == {"X_T0003": ["INTERVIEWEE_INTERVIEW_QUESTION"]}


def test_interviewee_clarification_after_a_question_is_not_a_candidate():
    assert rules(transcript_of([("enqueteur", "Et pour tes études ?"), ("enquete", "Tu veux dire pour les cours ?"),
                                ("enqueteur", "Oui.")])) == {}


def test_consecutive_interviewer_answer_and_conflicting_marker():
    found = rules(transcript_of([("enqueteur", "Tu l'utilises pour les plans ?"),
                                 ("enqueteur", "Oui je l'utilise."),
                                 ("enqueteur", "Et les fiches ? Enquêté : jamais pour les fiches.")]))
    assert found["X_T0002"] == ["CONSECUTIVE_INTERVIEWER_ANSWER"]
    assert found["X_T0003"] == ["CONFLICTING_SPEAKER_MARKER"]


def test_interviewer_framing_and_rephrasing_are_not_candidates():
    assert rules(transcript_of([
        ("enqueteur", "Bonjour, je m'appelle Paul, je suis en master et je fais mon mémoire sur l'IA."),
        ("enquete", "D'accord."),
        ("enqueteur", "Donc si je comprends bien, toi tu l'utilises pour reformuler."),
        ("enquete", "Oui."),
    ])) == {}


def test_unknown_speaker_turns_are_not_audited():
    assert rules(transcript_of([("unknown", "Moi personnellement j'utilise ChatGPT tous les jours pour mes cours."),
                                ("unknown", "Est-ce que tu l'utilises pour tes dissertations ?")])) == {}


def test_excerpt_contains_only_candidates_and_neighbours():
    transcript = transcript_of([("enqueteur", f"Question {i} ?") if i % 2 else ("enquete", f"Réponse {i}.")
                                for i in range(1, 21)])
    transcript["turns"][9] = {**transcript["turns"][9], "speaker": "enqueteur",
                              "text": "Moi personnellement j'utilise ChatGPT tous les jours pour mes cours."}
    request = sa.prepare_audit(transcript)
    assert [c["turn_id"] for c in request.candidates] == ["X_T0010"]
    assert request.excerpt_turn_ids == [f"X_T{i:04d}" for i in range(8, 13)]
    assert "Question 1 ?" not in request.audit_json and "</" not in request.audit_json.replace("<\\/", "")
    candidate_turn = next(t for t in request.audit_input["turns"] if t["candidate"])
    # le tour précédent est une question de l'enquêteur : deux règles concordent
    assert candidate_turn["heuristics"] == ["INTERVIEWER_FIRST_PERSON_ANSWER", "CONSECUTIVE_INTERVIEWER_ANSWER"]


def test_find_candidates_does_not_modify_the_transcript():
    transcript = transcript_of([("enqueteur", "Pour quoi faire ?"),
                                ("enqueteur", "Moi personnellement j'utilise ChatGPT tous les jours pour mes cours.")])
    before = copy.deepcopy(transcript)
    request = sa.prepare_audit(transcript)
    sa.build_audit_document(transcript, request, si.AUDIT_ASSESSMENTS)
    assert transcript == before


# --- B. Appel conditionnel, un seul par entretien ---------------------------------------------

def test_no_candidate_means_no_audit_call(tmp_path):
    run_, transport = run(tmp_path, si.NORMAL_FILES, audit=None)  # aucune réponse d'audit prévue
    assert [c["agent"] for c in transport.calls].count(AUDITOR) == 0 and len(transport.calls) == 2
    audit = load(run_, "speaker_attribution_audit.json")
    assert audit["status"] == "SUCCESS" and audit["audit_mode"] == "heuristics_only"
    assert audit["llm_called"] is False and audit["candidate_count"] == 0 and audit["items"] == []
    manifest = load(run_, "speaker_audit_manifest.json")
    assert manifest["api_calls"] == 0 and manifest["cache_key"] is None and manifest["usage"] is None
    assert run_["files"][0]["analysis"]["speaker_audit"]["candidate_count"] == 0


def test_several_candidates_give_exactly_one_call_on_an_excerpt(tmp_path):
    run_, transport = run(tmp_path, si.AUDIT_FILES)
    audit_calls = transport.calls_for(AUDITOR)
    assert len(audit_calls) == 1 and len(transport.calls) == 3
    params = audit_calls[0]["params"]
    content = params["messages"][0]["content"]
    assert si.au_tid(4) in content and si.au_tid(13) in content
    assert si.AUDIT_FAR_TURN_TEXT not in content  # jamais tout l'entretien
    assert params["system"][0]["text"] == sa.SPEC.system_prompt
    # l'audit précède les deux agents
    assert transport.calls[0]["agent"] == AUDITOR
    audit = load(run_, "speaker_attribution_audit.json")
    assert audit["status"] == "SUCCESS" and audit["candidate_count"] == 2 and audit["review_count"] == 2
    first = audit["items"][0]
    assert (first["turn_id"], first["current_speaker"], first["suggested_speaker"], first["confidence"]) == (
        si.au_tid(4), "enqueteur", "enquete", "high")
    assert first["needs_review"] is True and all(e["validation"]["valid"] for e in first["evidence"])
    usage = run_["last_analysis"]["usage"]
    assert usage["api_calls"] == 0 and len(transport.calls) == 3


def test_audit_result_is_cached(tmp_path):
    cache = AnalysisCache(tmp_path / "cache")
    run(tmp_path / "a", si.AUDIT_FILES, cache=cache)
    run_, transport = run(tmp_path / "b", si.AUDIT_FILES, cache=cache)
    assert transport.calls == []
    assert load(run_, "speaker_audit_manifest.json")["status"] == "CACHED"
    assert load(run_, "speaker_attribution_audit.json")["items"][0]["suggested_speaker"] == "enquete"


def test_plan_counts_the_audit_call_only_when_needed(tmp_path):
    run_ = si.make_ingested_run(tmp_path, si.AUDIT_FILES)
    cache = AnalysisCache(tmp_path / "cache")
    plan = plan_analysis(run_, [si.AUDIT_INTERVIEW_ID], fake_settings(), cache)
    assert plan == {"interviews": 1, "calls": 3, "cached": 0, "audit_calls": 1, "candidate_turns": 2, "exact": False}
    analyze_run(run_, settings=fake_settings(), client=FakeAgents(responders()), cache=cache)
    assert plan_analysis(run_, [si.AUDIT_INTERVIEW_ID], fake_settings(), cache) == {
        "interviews": 1, "calls": 0, "cached": 3, "audit_calls": 0, "candidate_turns": 2, "exact": True}
    normal = si.make_ingested_run(tmp_path / "n", si.NORMAL_FILES)
    assert plan_analysis(normal, [si.NORMAL_INTERVIEW_ID], fake_settings(), cache)["audit_calls"] == 0


# --- Versionnage : n'invalider que ce qui doit l'être -----------------------------------------

def test_auditor_version_change_reruns_the_audit_only(tmp_path, monkeypatch):
    cache = AnalysisCache(tmp_path / "cache")
    run(tmp_path / "a", si.AUDIT_FILES, cache=cache)
    monkeypatch.setattr(analysis, "AUDITOR", dataclasses.replace(sa.SPEC, version="1.0.1"))
    _, transport = run(tmp_path / "b", si.AUDIT_FILES, cache=cache)
    # mêmes avertissements → mêmes entrées pour les agents → agents repris du cache
    assert [c["agent"] for c in transport.calls] == [AUDITOR]


def test_changed_warnings_rerun_both_agents(tmp_path, monkeypatch):
    cache = AnalysisCache(tmp_path / "cache")
    run(tmp_path / "a", si.AUDIT_FILES, cache=cache)
    monkeypatch.setattr(analysis, "AUDITOR", dataclasses.replace(sa.SPEC, version="1.0.1"))
    different = copy.deepcopy(si.AUDIT_ASSESSMENTS)
    different["assessments"][1].update(suggested_speaker=None, confidence="low")
    _, transport = run(tmp_path / "b", si.AUDIT_FILES, cache=cache, audit=different)
    assert sorted(c["agent"] for c in transport.calls) == sorted([AUDITOR, PRACTICE, INTERACTION])


def test_agent_changes_do_not_invalidate_the_audit_or_the_other_agent(tmp_path, monkeypatch):
    cache = AnalysisCache(tmp_path / "cache")
    run(tmp_path / "a", si.AUDIT_FILES, cache=cache)
    monkeypatch.setattr(analysis, "AGENTS", (practice_extractor.SPEC, dataclasses.replace(
        interaction_signal_reader.SPEC, schema_version="9.9")))
    _, transport = run(tmp_path / "b", si.AUDIT_FILES, cache=cache)
    assert [c["agent"] for c in transport.calls] == [INTERACTION]


# --- Le transcript n'est jamais modifié -------------------------------------------------------

def test_transcript_files_are_never_modified(tmp_path):
    run_ = si.make_ingested_run(tmp_path, si.AUDIT_FILES)
    interview_dir = Path(run_["files"][0]["ingestion"]["output_dir"])

    def digest():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in interview_dir.iterdir() if p.is_file()}

    before = digest()
    transcript_before = json.loads((interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text())
    run_ = analyze_run(run_, settings=fake_settings(), client=FakeAgents(responders()),
                       cache=AnalysisCache(tmp_path / "cache"))
    assert digest() == before
    transcript_after = json.loads((interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text())
    assert transcript_after == transcript_before
    assert transcript_after["turns"][3]["speaker"] == "enqueteur"  # suggestion jamais appliquée
    audit = load(run_, "speaker_attribution_audit.json")
    assert audit["transcript_modified"] is False
    assert audit["structured_transcript_sha256"] == before[config.STRUCTURED_TRANSCRIPT_FILENAME]
    assert "ne modifient pas la transcription originale" in audit["notice"]


def test_turn_ids_and_official_speakers_are_kept_in_agent_inputs(tmp_path):
    run_, transport = run(tmp_path, si.AUDIT_FILES)
    transcript = json.loads((out_dir(run_).parent / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text())
    content = transport.calls_for(PRACTICE)[0]["params"]["messages"][0]["content"]
    sent = json.loads(content[content.index("<transcript>\n") + 13:content.rindex("\n</transcript>")])
    assert [t["turn_id"] for t in sent["turns"]] == [t["turn_id"] for t in transcript["turns"]]
    assert [t["speaker"] for t in sent["turns"]] == [t["speaker"] for t in transcript["turns"]]
    assert [t["text"] for t in sent["turns"]] == [t["text"] for t in transcript["turns"]]


# --- Avertissements transmis aux agents --------------------------------------------------------

def test_warnings_are_sent_to_both_agents_as_secondary_metadata(tmp_path):
    run_, transport = run(tmp_path, si.AUDIT_FILES)
    practice_call = transport.calls_for(PRACTICE)[0]["params"]
    signal_call = transport.calls_for(INTERACTION)[0]["params"]
    assert practice_call["messages"] == signal_call["messages"]  # toujours le même entretien pour les deux
    content = practice_call["messages"][0]["content"]
    assert "le speaker officiel du transcript reste inchangé" in content
    sent = json.loads(content[content.index("<transcript>\n") + 13:content.rindex("\n</transcript>")])
    flagged = {t["turn_id"]: t for t in sent["turns"] if "speaker_warning" in t}
    assert flagged[si.au_tid(4)]["speaker"] == "enqueteur"
    assert flagged[si.au_tid(4)]["speaker_warning"] == {"suggested_speaker": "enquete", "confidence": "high"}
    assert flagged[si.au_tid(13)]["speaker_warning"] == {"suggested_speaker": "enqueteur", "confidence": "medium"}
    assert set(flagged) == {si.au_tid(4), si.au_tid(13)}
    # seule l'information minimale passe : ni raison, ni règles de l'auditeur
    assert si.AUDIT_ASSESSMENTS["assessments"][0]["reason"] not in content
    assert "INTERVIEWER_FIRST_PERSON_ANSWER" not in content and "assessments" not in content
    for name in ("practice_manifest.json", "interaction_manifest.json"):
        assert load(run_, name)["speaker_warning_count"] == 2


def test_agent_objects_citing_a_doubtful_turn_are_marked_for_review(tmp_path):
    practices = {"practices": [si.practice(
        summary="Il indique utiliser ChatGPT tous les jours pour ses cours.", turn_start=si.au_tid(3),
        turn_end=si.au_tid(4), ai_tool=["ChatGPT"], explicitness="unclear",
        uncertainty_note="Attribution du locuteur du tour T0004 douteuse.",
        evidence=[si.au_ev(4, "j'utilise ChatGPT tous les jours")])], "extraction_notes": None}
    run_, _ = run(tmp_path, si.AUDIT_FILES, practices=practices)
    practice = load(run_, "practice_extractor.json")["practices"][0]
    assert "EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN" in practice["review_reasons"] and practice["needs_review"]
    assert practice["evidence"][0]["validation"]["valid"] is True


# --- Revalidation de la réponse du modèle ------------------------------------------------------

def test_model_output_is_validated_without_correction(tmp_path):
    output = {"assessments": [
        si.assessment(turn_id=si.au_tid(4), suggested_speaker="enquete", confidence="high", needs_review=False,
                      reason="Réponse à la première personne.",
                      evidence=[si.au_ev(4, "J'utilise ChatGPT chaque jour")]),          # citation inventée
        si.assessment(turn_id=si.au_tid(99), suggested_speaker="enqueteur", confidence="high",  # tour inexistant
                      reason="Question.", evidence=[si.au_ev(13, "Est-ce que tu l'utilises")]),
        si.assessment(turn_id=si.au_tid(5), suggested_speaker="enqueteur", confidence="low",    # non candidat
                      reason="Question.", evidence=[si.au_ev(5, "Et pour les examens ?")]),
    ], "audit_notes": None}  # T0013 (candidat) absent de la réponse
    run_, _ = run(tmp_path, si.AUDIT_FILES, audit=output)
    audit = load(run_, "speaker_attribution_audit.json")
    assert audit["status"] == "SUCCESS_WITH_WARNINGS"
    items = {item["turn_id"]: item for item in audit["items"]}
    first = items[si.au_tid(4)]
    assert first["evidence"][0]["quote"] == "J'utilise ChatGPT chaque jour"  # jamais corrigée
    assert {"QUOTE_NOT_FOUND", "NO_VALID_EVIDENCE", "SUGGESTION_WITHOUT_REVIEW"} <= set(first["review_reasons"])
    assert first["needs_review"] is True and first["model_needs_review"] is False
    assert "UNKNOWN_TURN_ID" in items[si.au_tid(99)]["review_reasons"] and items[si.au_tid(99)]["current_speaker"] is None
    assert "SUGGESTION_EQUALS_CURRENT" in items[si.au_tid(5)]["review_reasons"]
    assert "NOT_A_CANDIDATE" in items[si.au_tid(5)]["review_reasons"]
    missing = items[si.au_tid(13)]
    assert missing["source"] == "heuristics" and missing["suggested_speaker"] is None
    assert missing["confidence"] == "low" and "CANDIDATE_NOT_ASSESSED" in missing["review_reasons"]
    # un turn_id inexistant n'est jamais transmis aux agents
    validation = load(run_, "evidence_validation.json")
    assert validation["speaker_audit"]["invalid_evidence_count"] == 1
    content = json.dumps(json.loads((out_dir(run_) / "practice_manifest.json").read_text()))
    assert load(run_, "practice_manifest.json")["speaker_warning_count"] == 3 and si.au_tid(99) not in content


def test_ambiguous_assessment_keeps_low_confidence_and_no_suggestion(tmp_path):
    output = {"assessments": [
        si.assessment(turn_id=si.au_tid(4), suggested_speaker=None, confidence="low", needs_review=True,
                      reason="Le passage pourrait être prononcé par l'un ou l'autre.",
                      evidence=[si.au_ev(4, "ça m'aide pour mes cours")]),
        si.assessment(turn_id=si.au_tid(13), suggested_speaker=None, confidence="medium", needs_review=False,
                      reason="Attribution actuelle plausible.",
                      evidence=[si.au_ev(13, "Est-ce que tu l'utilises aussi pour tes dissertations ?")]),
    ], "audit_notes": None}
    run_, transport = run(tmp_path, si.AUDIT_FILES, audit=output)
    items = load(run_, "speaker_attribution_audit.json")["items"]
    assert (items[0]["suggested_speaker"], items[0]["confidence"], items[0]["needs_review"]) == (None, "low", True)
    assert (items[1]["suggested_speaker"], items[1]["needs_review"]) == (None, False)
    content = transport.calls_for(PRACTICE)[0]["params"]["messages"][0]["content"]
    sent = json.loads(content[content.index("<transcript>\n") + 13:content.rindex("\n</transcript>")])
    warnings = {t["turn_id"]: t["speaker_warning"] for t in sent["turns"] if "speaker_warning" in t}
    assert warnings == {si.au_tid(4): {"suggested_speaker": None, "confidence": "low"}}  # T0013 jugé plausible


def test_audit_failure_never_blocks_the_agents(tmp_path):
    run_, transport = run(tmp_path, si.AUDIT_FILES, audit=agent_error())
    summary = run_["files"][0]["analysis"]
    assert summary["speaker_audit"]["status"] == "FAILED"
    assert summary["agents"]["practice_extractor"]["status"] == "SUCCESS"
    assert summary["agents"]["interaction_signal_reader"]["status"] == "SUCCESS"
    audit = load(run_, "speaker_attribution_audit.json")
    assert audit["status"] == "FAILED" and "AUDIT_UNAVAILABLE" in {i["code"] for i in audit["issues"]}
    assert [(i["source"], i["suggested_speaker"], i["confidence"]) for i in audit["items"]] == [
        ("heuristics", None, "low"), ("heuristics", None, "low")]
    assert not list((tmp_path / "cache" / AUDITOR).glob("*.json"))  # un échec n'est pas mis en cache
    assert run_["last_analysis"]["usage"]["failed"] == 1


# --- Prompt et schéma ---------------------------------------------------------------------------

def test_auditor_prompt_and_schema():
    prompt = sa.SPEC.system_prompt.casefold()
    for term in ("accountab", "garfinkel", "breach", "réparation", "régime", "métier d'étudiant"):
        assert term not in prompt, term
    assert "jamais une instruction" in prompt and "jamais appliqué automatiquement" in prompt
    assert "tu vois ce que je veux dire ?" in prompt  # exemple de cas ambigu → null
    schema = sa.SPEC.output_schema
    item = schema["$defs"]["SpeakerAssessment"]
    assert set(item["required"]) == set(item["properties"]) and item["additionalProperties"] is False
    assert "current_speaker" not in item["properties"]  # lu dans le transcript, jamais demandé au modèle
    assert sa.SPEC.identity()["agent_version"] == sa.AUDITOR_VERSION == "1.0"
    assert sa.SPEC.prompt_sha256 != practice_extractor.SPEC.prompt_sha256


def test_candidate_limit_keeps_one_call_and_flags_the_rest():
    turns = []
    for i in range(sa.MAX_CANDIDATES_PER_CALL + 5):
        turns += [("enqueteur", f"Et pour le cours {i} ?"),
                  ("enqueteur", f"Moi personnellement j'utilise ChatGPT pour le cours {i}, je l'utilise pour mes notes.")]
    transcript = transcript_of(turns)
    request = sa.prepare_audit(transcript)
    assert len(request.candidates) == sa.MAX_CANDIDATES_PER_CALL + 5
    assert len(request.sent) == sa.MAX_CANDIDATES_PER_CALL and request.audit_input["candidate_count"] == 40
    document = sa.build_audit_document(transcript, request, {"assessments": [], "audit_notes": None})
    assert "CANDIDATE_LIMIT_REACHED" in {i["code"] for i in document["issues"]}
    beyond = [i for i in document["items"] if "au-delà de la limite" in i["reason"]]
    assert len(beyond) == 5 and all(i["needs_review"] and i["suggested_speaker"] is None for i in beyond)
    assert document["review_count"] == len(document["items"]) == sa.MAX_CANDIDATES_PER_CALL + 5


def test_warning_never_suggests_the_current_speaker():
    transcript = transcript_of([("enqueteur", "Pour quoi faire ?"),
                                ("enqueteur", "Moi personnellement j'utilise ChatGPT tous les jours pour mes cours.")])
    request = sa.prepare_audit(transcript)
    output = {"assessments": [si.assessment(turn_id="X_T0002", suggested_speaker="enqueteur", confidence="low",
                                            reason="Incertain.", evidence=[{"turn_id": "X_T0002", "quote": "Moi"}])],
              "audit_notes": None}
    document = sa.build_audit_document(transcript, request, output)
    assert "SUGGESTION_EQUALS_CURRENT" in document["items"][0]["review_reasons"]
    assert sa.agent_warnings(document, transcript) == {"X_T0002": {"suggested_speaker": None, "confidence": "low"}}


def test_unexpected_audit_error_never_blocks_agents_nor_leaves_stale_files(tmp_path, monkeypatch):
    cache = AnalysisCache(tmp_path / "cache")
    run_ = si.make_ingested_run(tmp_path, si.AUDIT_FILES)
    analyze_run(run_, settings=fake_settings(), client=FakeAgents(responders()), cache=cache)
    assert (out_dir(run_) / "speaker_attribution_audit.json").exists()

    def broken(_transcript):
        raise RuntimeError("bug")

    monkeypatch.setattr(sa, "prepare_audit", broken)
    run_ = analyze_run(run_, settings=fake_settings(), client=FakeAgents(responders()), cache=cache, force=True)
    summary = run_["files"][0]["analysis"]
    assert summary["speaker_audit"]["status"] == "FAILED"
    assert summary["agents"]["practice_extractor"]["status"] == "SUCCESS"
    assert not (out_dir(run_) / "speaker_attribution_audit.json").exists()
    assert not (out_dir(run_) / "speaker_audit_manifest.json").exists()
