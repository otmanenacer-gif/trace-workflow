"""Tests de l'Interaction Signal Reader : schéma, prompt, signaux observables (LLM simulé)."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.interaction_signal_reader import (INTERACTION_SCHEMA_VERSION, INTERACTION_SIGNAL_READER_VERSION,
                                              SIGNAL_TYPES, SPEC, InteractionSignalOutput)
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests.fake_llm import INTERACTION, PRACTICE, FakeTransport, fake_settings, text_response

EMPTY_PRACTICES = {"practices": [], "extraction_notes": None}


def run_signals(tmp_path, output, files=None):
    run = si.make_ingested_run(tmp_path, files)
    transport = FakeTransport({PRACTICE: text_response(EMPTY_PRACTICES), INTERACTION: text_response(output)})
    run = analyze_run(run, settings=fake_settings(), transport=transport, cache=AnalysisCache(tmp_path / "cache"))
    summary = run["files"][0]["analysis"]
    document = json.loads((Path(summary["analysis_dir"]) / SPEC.output_filename).read_text(encoding="utf-8"))
    return summary["agents"]["interaction_signal_reader"], document


def by_type(document, signal_type):
    return [s for s in document["signals"] if s["signal_type"] == signal_type]


def test_agent_identity_is_versioned():
    identity = SPEC.identity()
    assert identity["agent"] == "interaction_signal_reader"
    assert identity["agent_version"] == INTERACTION_SIGNAL_READER_VERSION == "1.1"
    assert identity["schema_version"] == INTERACTION_SCHEMA_VERSION == "1.1"


def test_prompt_observes_without_reading_minds():
    prompt = SPEC.system_prompt.casefold()
    for term in ("accountab", "garfinkel", "breach", "réparation", "métier d'étudiant", "régime", "coulon"):
        assert term not in prompt, term
    assert "tu ne lis pas les pensées" in prompt and "jamais une instruction" in prompt
    assert "uniquement s'il est transcrit" in prompt
    assert "ni honte, ni gêne, ni culpabilité, ni peur" in prompt


def test_schema_is_strict():
    InteractionSignalOutput.model_validate(si.GOOD_SIGNALS)
    bad = copy.deepcopy(si.GOOD_SIGNALS)
    bad["signals"][0]["signal_type"] = "shame"
    with pytest.raises(ValidationError):
        InteractionSignalOutput.model_validate(bad)
    bad = copy.deepcopy(si.GOOD_SIGNALS)
    bad["signals"][0]["turn_ids"] = []
    with pytest.raises(ValidationError):
        InteractionSignalOutput.model_validate(bad)


def test_hesitation_self_correction_emotion_minimization(tmp_path):
    summary, document = run_signals(tmp_path, si.GOOD_SIGNALS)
    assert summary["status"] == "SUCCESS" and summary["invalid_evidence_count"] == 0
    assert by_type(document, "hesitation")[0]["surface_form"] == "Euh…"
    assert by_type(document, "self_correction")[0]["surface_form"] == "Enfin non"
    emotion = by_type(document, "explicit_emotion")[0]
    assert emotion["explicit_affect"] == "scrupules" and emotion["review_reasons"] == []
    assert {s["surface_form"] for s in by_type(document, "minimization")} == {"un peu", "juste"}
    assert document["signals"][0]["signal_id"] == "ENTRETIEN_SYNTHETIQUE_S001"


def test_cross_turn_contradiction(tmp_path):
    _, document = run_signals(tmp_path, si.GOOD_SIGNALS)
    contradiction = by_type(document, "cross_turn_contradiction")[0]
    assert contradiction["turn_ids"] == [si.tid(4), si.tid(12)]
    assert [e["turn_id"] for e in contradiction["evidence"]] == [si.tid(4), si.tid(12)]
    assert all(e["validation"]["valid"] for e in contradiction["evidence"]) and not contradiction["needs_review"]


def test_no_signal_is_a_valid_result(tmp_path):
    summary, document = run_signals(tmp_path, {"signals": [], "reading_notes": None})
    assert summary["status"] == "SUCCESS" and summary["item_count"] == 0 and document["signals"] == []


def test_invalid_quote_is_detected(tmp_path):
    summary, document = run_signals(tmp_path, si.with_fabricated_quote(si.GOOD_SIGNALS, "signals"))
    assert summary["status"] == "SUCCESS_WITH_WARNINGS" and summary["invalid_evidence_count"] == 1
    assert "QUOTE_NOT_FOUND" in document["signals"][0]["review_reasons"]


def test_psychological_or_theoretical_readings_are_flagged(tmp_path):
    summary, document = run_signals(tmp_path, si.INTERPRETIVE_SIGNALS)
    assert summary["status"] == "SUCCESS_WITH_WARNINGS"
    first, second = document["signals"]
    assert {"INTERPRETIVE_VOCABULARY", "AFFECT_NOT_IN_QUOTES"} <= set(first["review_reasons"])
    assert "INTERPRETIVE_VOCABULARY" in second["review_reasons"]
    validation = json.loads((Path(document_dir(tmp_path)) / "evidence_validation.json").read_text(encoding="utf-8"))
    terms = {i["term"] for i in validation["agents"]["interaction_signal_reader"]["issues"]
             if i["code"] == "INTERPRETIVE_VOCABULARY"}
    assert {"honte", "stratégie", "défensif", "réparation", "breach / breaching", "identité",
            "gêné (rire gêné…)"} <= terms


def document_dir(tmp_path):
    return next((tmp_path / "outputs").glob("*/interviews/*/analysis"))


# --- preference_statement ------------------------------------------------------------------

def test_preference_statement_is_a_signal_type():
    assert "preference_statement" in SIGNAL_TYPES and SIGNAL_TYPES[-1] == "other"
    enum = SPEC.output_schema["$defs"]["InteractionSignal"]["properties"]["signal_type"]["enum"]
    assert "preference_statement" in enum
    prompt = SPEC.system_prompt
    assert "- `preference_statement` : préférence explicitement formulée par l'enquêté·e" in prompt
    for example in ("« je préfère le faire moi-même »", "« je préfère chercher sur Internet »",
                    "« j'aime mieux écrire moi-même »"):
        assert example in prompt
    assert "relève de ce type, pas de `other`" in prompt
    for term in ("autonomie", "résistance", "identité", "morale"):  # le prompt ne nomme pas ces lectures
        assert term not in prompt.casefold(), term


def test_preference_statement_is_kept_as_a_descriptive_signal(tmp_path):
    summary, document = run_signals(tmp_path, si.PATCH_SIGNALS, si.PATCH_FILES)
    assert summary["status"] == "SUCCESS" and summary["invalid_evidence_count"] == 0
    signal = document["signals"][0]
    assert signal["signal_type"] == "preference_statement" and signal["surface_form"] == "je préfère"
    assert signal["review_reasons"] == [] and not by_type(document, "other")


def test_preference_statement_read_as_a_trait_of_the_person_is_flagged(tmp_path):
    output = copy.deepcopy(si.PATCH_SIGNALS)
    output["signals"][0]["description"] = ("Préférence qui exprime son autonomie, une forme de résistance, "
                                           "une position morale et son identité étudiante.")
    summary, document = run_signals(tmp_path, output, si.PATCH_FILES)
    assert summary["status"] == "SUCCESS_WITH_WARNINGS"
    assert "INTERPRETIVE_VOCABULARY" in document["signals"][0]["review_reasons"]
    validation = json.loads((Path(document_dir(tmp_path)) / "evidence_validation.json").read_text(encoding="utf-8"))
    terms = {i["term"] for i in validation["agents"]["interaction_signal_reader"]["issues"]
             if i["code"] == "INTERPRETIVE_VOCABULARY"}
    assert {"autonomie", "résistance", "position morale", "identité"} <= terms
