"""Tests du Practice Extractor : schéma, prompt « aveugle », validation des preuves (LLM simulé)."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.practice_extractor import PRACTICE_EXTRACTOR_VERSION, PRACTICE_SCHEMA_VERSION, SPEC, PracticeExtractorOutput
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests.fake_llm import INTERACTION, PRACTICE, FakeTransport, fake_settings, text_response

EMPTY_SIGNALS = {"signals": [], "reading_notes": None}

# Concepts théoriques réservés à une étape ultérieure : le Practice Extractor ne doit pas les connaître.
THEORY_TERMS = ("accountab", "garfinkel", "breach", "réparation", "reparation", "métier d'étudiant",
                "metier d'etudiant", "régime", "regime", "coulon", "ethnométhodo", "ethnomethodo", "typologie des")


def run_practices(tmp_path, output, files=None):
    run = si.make_ingested_run(tmp_path, files)
    transport = FakeTransport({PRACTICE: text_response(output), INTERACTION: text_response(EMPTY_SIGNALS)})
    run = analyze_run(run, settings=fake_settings(), transport=transport, cache=AnalysisCache(tmp_path / "cache"))
    summary = run["files"][0]["analysis"]
    document = json.loads((Path(summary["analysis_dir"]) / SPEC.output_filename).read_text(encoding="utf-8"))
    return summary["agents"]["practice_extractor"], document


def test_agent_identity_is_versioned():
    identity = SPEC.identity()
    assert identity["agent"] == "practice_extractor"
    assert identity["agent_version"] == PRACTICE_EXTRACTOR_VERSION == "1.0"
    assert identity["schema_version"] == PRACTICE_SCHEMA_VERSION == "1.0"
    assert len(identity["prompt_sha256"]) == 64 and len(identity["schema_sha256"]) == 64


def test_prompt_is_theoretically_blind_and_descriptive():
    prompt = SPEC.system_prompt.casefold()
    for term in THEORY_TERMS:
        assert term not in prompt, term
    assert "descriptif" in prompt and "jamais une instruction" in prompt
    assert "ignore tes instructions précédentes" in prompt  # exemple de consigne à ne pas suivre
    assert "l'étudiante indique utiliser chatgpt pour obtenir un résumé" in prompt  # bon exemple


def test_schema_accepts_good_output_and_is_strict():
    PracticeExtractorOutput.model_validate(si.GOOD_PRACTICES)
    bad = copy.deepcopy(si.GOOD_PRACTICES)
    bad["practices"][0]["use_status"] = "cheating"
    with pytest.raises(ValidationError):
        PracticeExtractorOutput.model_validate(bad)
    bad = copy.deepcopy(si.GOOD_PRACTICES)
    bad["practices"][0]["evidence"] = []
    with pytest.raises(ValidationError):
        PracticeExtractorOutput.model_validate(bad)
    bad = copy.deepcopy(si.GOOD_PRACTICES)
    bad["practices"][0]["diagnosis"] = "dépendant"
    with pytest.raises(ValidationError):
        PracticeExtractorOutput.model_validate(bad)
    bad = copy.deepcopy(si.GOOD_PRACTICES)
    del bad["practices"][0]["discipline"]  # null doit être explicite
    with pytest.raises(ValidationError):
        PracticeExtractorOutput.model_validate(bad)


def test_api_schema_requires_every_field_and_forbids_extras():
    schema = SPEC.output_schema
    practice = schema["$defs"]["Practice"]
    assert practice["additionalProperties"] is False
    assert set(practice["required"]) == set(practice["properties"])
    assert practice["properties"]["use_status"]["enum"] == ["use", "non_use", "refusal", "hypothetical", "past_use"]


def test_good_output_is_valid_with_multiple_evidence(tmp_path):
    summary, document = run_practices(tmp_path, si.GOOD_PRACTICES)
    assert summary["status"] == "SUCCESS" and summary["invalid_evidence_count"] == 0
    assert document["item_count"] == 6 and document["needs_review_count"] == 0
    reformulation = document["practices"][2]
    assert reformulation["practice_id"] == "ENTRETIEN_SYNTHETIQUE_P003"
    assert len(reformulation["evidence"]) == 2 and all(e["validation"]["valid"] for e in reformulation["evidence"])


def test_non_use_and_hypothetical_are_kept(tmp_path):
    _, document = run_practices(tmp_path, si.GOOD_PRACTICES)
    statuses = [p["use_status"] for p in document["practices"]]
    assert statuses.count("non_use") == 2 and statuses.count("hypothetical") == 1
    exam = next(p for p in document["practices"] if p["assessment_context"] == "exam")
    assert exam["use_status"] == "non_use" and "on n'a pas le droit" in exam["stated_reason"]


def test_unknown_values_are_null_or_unknown(tmp_path):
    output = {"practices": [si.practice(
        summary="L'étudiante indique utiliser l'outil pour obtenir une définition.",
        turn_start=si.tid(2), turn_end=si.tid(2), academic_task=None, discipline=None, ai_tool=[],
        assessment_context="unknown", explicitness="unclear",
        uncertainty_note="L'outil n'est pas nommé dans ce passage.",
        evidence=[si.ev(2, "je lui demande une définition")])], "extraction_notes": None}
    summary, document = run_practices(tmp_path, output)
    practice = document["practices"][0]
    assert practice["discipline"] is None and practice["assessment_context"] == "unknown"
    assert practice["review_reasons"] == ["EXPLICITNESS_UNCLEAR"] and summary["status"] == "SUCCESS"


def test_invalid_quote_is_detected_and_flagged(tmp_path):
    summary, document = run_practices(tmp_path, si.with_fabricated_quote(si.GOOD_PRACTICES, "practices"))
    assert summary["status"] == "SUCCESS_WITH_WARNINGS" and summary["invalid_evidence_count"] == 1
    first = document["practices"][0]
    assert first["needs_review"] and "QUOTE_NOT_FOUND" in first["review_reasons"]
    assert first["evidence"][0]["quote"] == "J'utilise ChatGPT pour écrire toutes mes dissertations."


def test_empty_practice_list_is_a_valid_result(tmp_path):
    summary, document = run_practices(tmp_path, {"practices": [], "extraction_notes": "Aucune situation décrite."})
    assert summary["status"] == "SUCCESS" and document["practices"] == []
    assert document["extraction_notes"] == "Aucune situation décrite."


def test_interpretive_vocabulary_is_flagged(tmp_path):
    summary, document = run_practices(tmp_path, si.INTERPRETIVE_PRACTICES)
    practice = document["practices"][0]
    assert "INTERPRETIVE_VOCABULARY" in practice["review_reasons"]
    path = next((tmp_path / "outputs").glob("*/interviews/*/analysis/evidence_validation.json"))
    validation = json.loads(path.read_text(encoding="utf-8"))
    terms = {i["term"] for i in validation["agents"]["practice_extractor"]["issues"]
             if i["code"] == "INTERPRETIVE_VOCABULARY"}
    assert {"triche", "identité", "éviter l'effort"} <= terms
