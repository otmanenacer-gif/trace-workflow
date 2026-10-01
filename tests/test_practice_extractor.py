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
    assert identity["agent_version"] == PRACTICE_EXTRACTOR_VERSION == "1.2"
    assert identity["schema_version"] == PRACTICE_SCHEMA_VERSION == "1.2"
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
    assert {"stated_frequency", "scope_qualifier"} <= set(practice["required"])


def test_good_output_is_valid_with_multiple_evidence(tmp_path):
    summary, document = run_practices(tmp_path, si.GOOD_PRACTICES)
    assert summary["status"] == "SUCCESS" and summary["invalid_evidence_count"] == 0
    assert document["item_count"] == 6 and document["needs_review_count"] == 0
    reformulation = document["practices"][2]
    assert reformulation["practice_id"] == "ENTRETIEN_SYNTHETIQUE_P003"
    assert len(reformulation["evidence"]) == 2 and all(e["validation"]["valid"] for e in reformulation["evidence"])
    assert reformulation["stated_frequency"] is None and reformulation["scope_qualifier"] == "surtout"


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


# --- stated_frequency / scope_qualifier ------------------------------------------------------

def _with_practice_fields(**fields) -> dict:
    output = copy.deepcopy(si.GOOD_PRACTICES)
    output["practices"][2].update(fields)
    return output


def test_prompt_separates_frequency_from_scope_qualifier():
    prompt = SPEC.system_prompt
    assert "`scope_qualifier`" in prompt
    assert "« je l'utilise surtout pour reformuler » → `stated_frequency` : `null`, `scope_qualifier` : « surtout »" in prompt


@pytest.mark.parametrize("frequency", ["parfois", "souvent", "rarement", "une fois", "jamais", "toujours", None])
def test_real_frequencies_are_accepted(frequency):
    PracticeExtractorOutput.model_validate(_with_practice_fields(stated_frequency=frequency, scope_qualifier=None))


@pytest.mark.parametrize("value", ["surtout", "Surtout", "SURTOUT", "souvent, surtout pour reformuler",
                                   "principalement", "essentiellement", "notamment", "en particulier"])
def test_surtout_can_never_be_a_stated_frequency(value):
    with pytest.raises(ValidationError, match="stated_frequency"):
        PracticeExtractorOutput.model_validate(_with_practice_fields(stated_frequency=value))


def test_scope_qualifier_is_explicit_and_nullable():
    PracticeExtractorOutput.model_validate(_with_practice_fields(stated_frequency=None, scope_qualifier="surtout"))
    PracticeExtractorOutput.model_validate(_with_practice_fields(stated_frequency=None, scope_qualifier=None))
    bad = copy.deepcopy(si.GOOD_PRACTICES)
    del bad["practices"][0]["scope_qualifier"]  # null doit être explicite
    with pytest.raises(ValidationError):
        PracticeExtractorOutput.model_validate(bad)


def test_surtout_as_frequency_is_never_recorded(tmp_path):
    """Une réponse qui range « surtout » dans stated_frequency est rejetée : rien n'est écrit ni mis en cache."""
    run = si.make_ingested_run(tmp_path)
    transport = FakeTransport({PRACTICE: text_response(_with_practice_fields(stated_frequency="surtout")),
                               INTERACTION: text_response(EMPTY_SIGNALS)})
    cache = AnalysisCache(tmp_path / "cache")
    run = analyze_run(run, settings=fake_settings(), transport=transport, cache=cache)
    summary = run["files"][0]["analysis"]
    agent = summary["agents"]["practice_extractor"]
    assert agent["status"] == "FAILED" and agent["error"]["code"] == "SCHEMA_VALIDATION"
    assert "practices.2.stated_frequency" in agent["error"]["message"]
    assert not (Path(summary["analysis_dir"]) / SPEC.output_filename).exists()
    assert not list((tmp_path / "cache" / "practice_extractor").glob("*.json"))
    assert summary["agents"]["interaction_signal_reader"]["status"] == "SUCCESS"  # l'autre agent n'est pas affecté


def test_scope_qualifier_is_scanned_by_the_guard(tmp_path):
    _, document = run_practices(tmp_path, _with_practice_fields(scope_qualifier="par paresse"))
    assert "INTERPRETIVE_VOCABULARY" in document["practices"][2]["review_reasons"]


# --- Étape 3.5 : non-usages et refus, distincts des usages ------------------------------------

def test_prompt_searches_uses_and_non_uses_without_merging():
    prompt = SPEC.system_prompt
    assert "Cherche **systématiquement les usages ET les non-usages ou refus**" in prompt
    for example in ("« pour mes plans je le fais moi-même »", "« je n'utilise jamais ChatGPT en examen »",
                    "« pour cette matière je ne l'utilise pas »", "« je veux pas qu'il fasse mes travaux »",
                    "« je refuse de lui faire écrire mon devoir »", "« je lui demande pas de réfléchir à ma place »"):
        assert example in prompt, example
    assert ("« Il m'arrive de lui demander de rédiger » puis « Mais normalement mes devoirs je les écris "
            "moi-même » donnent DEUX pratiques distinctes") in prompt
    assert "Ne les fusionne pas, ne tranche pas" in prompt
    assert "`past_use` (ce que l'enquêté·e faisait) et `non_use`" in prompt
    for value in ("not_stated", "preference", "personal_rule", "external_rule", "technical_limitation"):
        assert f"`{value}`" in prompt, value
    assert "n'écarte aucune situation au motif qu'elle n'est pas universitaire" in prompt


def test_schema_has_non_use_reason_and_practice_domain():
    practice = SPEC.output_schema["$defs"]["Practice"]
    assert {"non_use_reason", "practice_domain"} <= set(practice["required"])
    assert practice["properties"]["practice_domain"]["enum"] == ["academic", "personal", "professional", "mixed",
                                                                 "unknown"]
    reason = json.dumps(practice["properties"]["non_use_reason"])
    for value in ("not_stated", "preference", "personal_rule", "external_rule", "technical_limitation", "other"):
        assert value in reason
    bad = copy.deepcopy(si.NON_USE_PRACTICES)
    bad["practices"][1]["non_use_reason"] = "laziness"
    with pytest.raises(ValidationError):
        PracticeExtractorOutput.model_validate(bad)


def test_use_and_non_use_on_the_same_theme_stay_two_practices(tmp_path):
    """« Il m'arrive de lui demander de rédiger » / « Mais normalement mes devoirs je les écris moi-même »."""
    summary, document = run_practices(tmp_path, si.NON_USE_PRACTICES, si.NON_USE_FILES)
    assert summary["status"] == "SUCCESS" and summary["invalid_evidence_count"] == 0
    assert document["item_count"] == 6  # rien n'est fusionné
    use, non_use = document["practices"][0], document["practices"][1]
    assert (use["use_status"], non_use["use_status"]) == ("use", "non_use")
    assert use["practice_id"] != non_use["practice_id"]
    assert "rédiger" in use["evidence"][0]["quote"] and "moi-même" in non_use["evidence"][0]["quote"]
    assert use["non_use_reason"] is None and non_use["non_use_reason"] == "preference"
    assert not use["needs_review"] and not non_use["needs_review"]


def test_refusal_is_distinct_from_a_stated_preference(tmp_path):
    _, document = run_practices(tmp_path, si.NON_USE_PRACTICES, si.NON_USE_FILES)
    by_status = {}
    for p in document["practices"]:
        by_status.setdefault(p["use_status"], []).append(p)
    refusal, = by_status["refusal"]
    assert refusal["non_use_reason"] == "personal_rule"
    assert "Je lui demande pas de réfléchir à ma place." in [e["quote"] for e in refusal["evidence"]]
    preference = next(p for p in by_status["non_use"] if p["non_use_reason"] == "preference")
    assert "je préfère" in preference["evidence"][1]["quote"]


def test_past_use_and_current_non_use_are_two_practices(tmp_path):
    _, document = run_practices(tmp_path, si.NON_USE_PRACTICES, si.NON_USE_FILES)
    fiches = [p for p in document["practices"] if p["academic_task"] == "fiches de lecture"]
    assert [(p["use_status"], p["non_use_reason"]) for p in fiches] == [("past_use", None), ("non_use", "not_stated")]
    exam = document["practices"][-1]
    assert (exam["use_status"], exam["non_use_reason"], exam["assessment_context"]) == ("non_use", "external_rule", "exam")


def test_inconsistent_non_use_reason_is_flagged_not_corrected(tmp_path):
    output = copy.deepcopy(si.NON_USE_PRACTICES)
    output["practices"][0]["non_use_reason"] = "preference"   # un usage n'a pas de raison de non-usage
    output["practices"][1]["non_use_reason"] = None           # un non-usage doit en avoir une
    summary, document = run_practices(tmp_path, output, si.NON_USE_FILES)
    assert summary["status"] == "SUCCESS_WITH_WARNINGS"
    for practice in document["practices"][:2]:
        assert "NON_USE_REASON_MISMATCH" in practice["review_reasons"]
    assert document["practices"][0]["non_use_reason"] == "preference"  # rien n'est corrigé


def test_personal_practices_are_kept(tmp_path):
    output = {"practices": [si.practice(
        summary="L'étudiante indique utiliser ChatGPT pour reformuler, y compris hors des cours.",
        turn_start=si.tid(1), turn_end=si.tid(2), practice_domain="personal", assessment_context="personal",
        ai_tool=["ChatGPT"], evidence=[si.ev(2, "Oui, ChatGPT surtout.")])], "extraction_notes": None}
    summary, document = run_practices(tmp_path, output)
    assert summary["status"] == "SUCCESS" and document["practices"][0]["practice_domain"] == "personal"
