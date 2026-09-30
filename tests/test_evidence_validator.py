"""Tests du validateur déterministe de preuves (aucun appel IA)."""

import unicodedata

from core.evidence_validator import match_quote, validate_agent_output

TRANSCRIPT = {
    "interview_id": "ALICE",
    "turns": [
        {"turn_id": "ALICE_T0001", "speaker": "enqueteur", "text": "Tu utilises ChatGPT ?"},
        {"turn_id": "ALICE_T0002", "speaker": "enquete", "text": "Euh… oui, j’avoue que je l'utilise juste pour reformuler."},
        {"turn_id": "ALICE_T0003", "speaker": "enqueteur", "text": "Et pour les examens ?"},
        {"turn_id": "ALICE_T0004", "speaker": "enquete", "text": "Jamais pendant les examens.\nC'est interdit."},
    ],
}


def practice(evidence, start="ALICE_T0002", end="ALICE_T0002", **extra):
    return {"summary": "L'étudiante indique reformuler avec ChatGPT.", "turn_start": start, "turn_end": end,
            "ai_tool": ["ChatGPT"], "evidence": evidence, "explicitness": "direct", **extra}


def signal(evidence, turn_ids, **extra):
    return {"turn_ids": turn_ids, "signal_type": extra.pop("signal_type", "minimization"), "surface_form": "juste",
            "description": "« juste » réduit la portée.", "evidence": evidence, "explicit_affect": None,
            "explicitness": "direct", "needs_human_review": False, **extra}


def run_practice(item):
    return validate_agent_output("practice_extractor", [item], TRANSCRIPT, "P")


def run_signal(item):
    return validate_agent_output("interaction_signal_reader", [item], TRANSCRIPT, "S")


def codes(result):
    return {i["code"] for i in result["report"]["issues"]}


def test_exact_quote_is_valid_and_object_not_flagged():
    result = run_practice(practice([{"turn_id": "ALICE_T0002", "quote": "je l'utilise juste pour reformuler"}]))
    item = result["items"][0]
    assert item["practice_id"] == "ALICE_P001"
    assert item["evidence"][0]["validation"] == {"valid": True, "match": "exact", "code": None}
    assert item["needs_review"] is False and item["review_reasons"] == []
    assert result["report"]["invalid_evidence_count"] == 0 and result["report"]["issues"] == []


def test_invented_quote_is_invalid_and_flagged():
    result = run_practice(practice([{"turn_id": "ALICE_T0002", "quote": "je l'utilise pour tout rédiger"}]))
    item = result["items"][0]
    assert item["evidence"][0]["validation"]["valid"] is False
    assert {"QUOTE_NOT_FOUND", "NO_VALID_EVIDENCE"} <= codes(result)
    assert item["needs_review"] is True and result["report"]["invalid_evidence_count"] == 1


def test_invalid_quote_is_never_corrected():
    quote = "je l’utilise juste pour reformuler"  # apostrophe typographique : texte source avec '
    result = run_practice(practice([{"turn_id": "ALICE_T0002", "quote": quote}]))
    evidence = result["items"][0]["evidence"][0]
    assert evidence["quote"] == quote
    assert evidence["validation"] == {"valid": False, "match": "not_exact", "code": "QUOTE_NOT_EXACT"}


def test_quote_from_wrong_turn_is_invalid():
    result = run_practice(practice([{"turn_id": "ALICE_T0004", "quote": "je l'utilise juste pour reformuler"}],
                                   end="ALICE_T0004"))
    assert "QUOTE_NOT_FOUND" in codes(result)


def test_quote_spanning_two_turns_is_invalid():
    result = run_practice(practice([{"turn_id": "ALICE_T0001", "quote": "Tu utilises ChatGPT ? Euh… oui"}],
                                   start="ALICE_T0001"))
    assert "QUOTE_NOT_FOUND" in codes(result)


def test_unknown_turn_id():
    result = run_practice(practice([{"turn_id": "ALICE_T0099", "quote": "Jamais"}]))
    assert "UNKNOWN_TURN_ID" in codes(result)


def test_turn_from_another_interview_is_detected():
    result = run_practice(practice([{"turn_id": "BOB_T0002", "quote": "je l'utilise juste pour reformuler"}]))
    assert "FOREIGN_INTERVIEW_TURN" in codes(result)
    result = run_signal(signal([{"turn_id": "ALICE_T0002", "quote": "juste"}], ["ALICE_T0002", "BOB_T0010"]))
    assert "FOREIGN_INTERVIEW_TURN" in codes(result)


def test_inverted_range_is_invalid():
    result = run_practice(practice([{"turn_id": "ALICE_T0002", "quote": "juste pour reformuler"}],
                                   start="ALICE_T0004", end="ALICE_T0002"))
    assert "INVALID_TURN_RANGE" in codes(result) and result["items"][0]["needs_review"]


def test_unknown_range_turn():
    result = run_practice(practice([{"turn_id": "ALICE_T0002", "quote": "juste pour reformuler"}], end="ALICE_T0999"))
    assert "UNKNOWN_RANGE_TURN" in codes(result)


def test_evidence_outside_range_is_a_warning():
    result = run_practice(practice([{"turn_id": "ALICE_T0002", "quote": "juste pour reformuler"},
                                    {"turn_id": "ALICE_T0004", "quote": "Jamais pendant les examens."}]))
    issues = result["report"]["issues"]
    assert [i["code"] for i in issues] == ["EVIDENCE_OUTSIDE_RANGE"] and issues[0]["severity"] == "warning"
    assert result["report"]["invalid_evidence_count"] == 0


def test_empty_quote_and_multiline_quote():
    result = run_practice(practice([{"turn_id": "ALICE_T0004", "quote": "  "},
                                    {"turn_id": "ALICE_T0004", "quote": "examens.\nC'est interdit."}],
                                   start="ALICE_T0004", end="ALICE_T0004"))
    validations = [e["validation"] for e in result["items"][0]["evidence"]]
    assert validations[0]["code"] == "EMPTY_QUOTE" and validations[1]["valid"] is True


def test_unicode_canonical_equivalence_is_exact():
    decomposed = unicodedata.normalize("NFD", "Jamais pendant les examens.")
    assert match_quote("Jamais pendant les examens.", decomposed) == "exact"
    assert match_quote("JAMAIS pendant les examens.", "Jamais pendant les examens.") == "not_exact"
    assert match_quote("toujours", "Jamais pendant les examens.") == "not_found"


def test_interviewer_only_evidence_is_flagged():
    result = run_practice(practice([{"turn_id": "ALICE_T0001", "quote": "Tu utilises ChatGPT ?"}],
                                   start="ALICE_T0001"))
    assert "NO_INTERVIEWEE_EVIDENCE" in codes(result)


def test_signal_checks_affect_contradiction_and_listed_turns():
    ok = run_signal(signal([{"turn_id": "ALICE_T0002", "quote": "j’avoue"}], ["ALICE_T0002"],
                           signal_type="explicit_emotion", explicit_affect="avoue"))
    assert "AFFECT_NOT_IN_QUOTES" not in codes(ok)
    affect = run_signal(signal([{"turn_id": "ALICE_T0002", "quote": "Euh… oui"}], ["ALICE_T0002"],
                               signal_type="explicit_emotion", explicit_affect="honte"))
    assert "AFFECT_NOT_IN_QUOTES" in codes(affect)
    contradiction = run_signal(signal([{"turn_id": "ALICE_T0002", "quote": "juste pour reformuler"}],
                                      ["ALICE_T0002"], signal_type="cross_turn_contradiction"))
    assert "CONTRADICTION_SINGLE_TURN" in codes(contradiction)
    unlisted = run_signal(signal([{"turn_id": "ALICE_T0004", "quote": "C'est interdit."}], ["ALICE_T0002"]))
    assert "EVIDENCE_TURN_NOT_LISTED" in codes(unlisted)


def test_info_level_review_reasons_do_not_count_as_problems():
    result = run_signal(signal([{"turn_id": "ALICE_T0002", "quote": "juste"}], ["ALICE_T0002"],
                               explicitness="unclear", needs_human_review=True))
    item = result["items"][0]
    assert item["needs_review"] and set(item["review_reasons"]) == {"EXPLICITNESS_UNCLEAR", "AGENT_FLAGGED_REVIEW"}
    assert result["report"]["error_count"] == 0 and result["report"]["warning_count"] == 0
