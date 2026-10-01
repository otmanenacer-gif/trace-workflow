"""Tests du garde-fou lexical : « réparation » examinée en contexte (étape 3.5), sans IA."""

import pytest

from core.interpretation_guard import GUARD_VERSION, fold, reparation_uses, scan_item

ORDINARY = [
    "réparation du lave-vaisselle",
    "réparation d'un appareil",
    "réparer un ordinateur",
    "réparation domestique",
    "coût de réparation",
    "réparation d'un téléphone",
    "Il dit avoir réparé le lave-vaisselle lui-même.",
    "Il a regardé un tutoriel de réparation.",
    "travail de réparation sur la voiture",
    "réparer soi-même la machine",
]
INTERPRETIVE = [
    "travail de réparation de sa conduite",
    "travail de réparation de la conduite",
    "réparation discursive",
    "opération de réparation",
    "réparation de l'identité",
    "stratégie de réparation",
    "réparation de l'accountability",
    "il accomplit une réparation de son identité étudiante",
    "une forme symbolique de réparation",
    "il effectue un geste réparateur",
]


def test_guard_version_is_bumped():
    assert GUARD_VERSION == "1.3"  # étape 3.7


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_material_uses_are_not_interpretive(text):
    uses = reparation_uses(fold(text))
    assert uses and set(uses) == {"ordinary"}, (text, uses)


@pytest.mark.parametrize("text", INTERPRETIVE)
def test_theoretical_uses_are_interpretive(text):
    assert "interpretive" in reparation_uses(fold(text)), text


def test_unrelated_words_are_ignored():
    assert reparation_uses(fold("Il veut répartir son temps ; la répartition des tâches.")) == []


def test_bare_noun_stays_flagged_unless_the_object_is_material():
    assert reparation_uses(fold("Il évoque une réparation.")) == ["interpretive"]
    # Même nom, mais l'objet décrit un lave-vaisselle ailleurs : sens matériel
    assert reparation_uses(fold("Il évoque une réparation."), fold("contexte : lave-vaisselle")) == ["ordinary"]


def practice_item(**fields):
    item = {"summary": "Il indique avoir demandé à ChatGPT comment procéder.", "evidence": [
        {"turn_id": "X_T0002", "quote": "je lui ai demandé comment réparer le lave-vaisselle"}]}
    item.update(fields)
    return item


def test_practice_context_about_a_dishwasher_raises_no_warning():
    assert scan_item(practice_item(context="réparation du lave-vaisselle"), "practice_extractor") == []
    assert scan_item(practice_item(context="Réparation domestique.", summary="Il indique avoir réparé un appareil."),
                     "practice_extractor") == []


def test_signal_description_with_identity_repair_is_flagged():
    signal = {"description": "il accomplit une réparation de son identité étudiante",
              "evidence": [{"turn_id": "X_T0002", "quote": "c'est assez ridicule ce que je dis"}]}
    terms = {f["term"] for f in scan_item(signal, "interaction_signal_reader")}
    assert {"réparation", "identité"} <= terms


@pytest.mark.parametrize("text", ["travail de réparation de sa conduite", "réparation discursive"])
def test_interpretive_repair_in_any_authored_field_is_flagged(text):
    assert [f["term"] for f in scan_item(practice_item(uncertainty_note=text), "practice_extractor")] == ["réparation"]


def test_ordinary_use_in_quotes_does_not_excuse_an_interpretive_use():
    """L'enquêté parle de réparer son lave-vaisselle : cela n'autorise pas « réparation de son identité »."""
    item = practice_item(summary="Il opère une réparation de sa conduite.")
    assert [f["term"] for f in scan_item(item, "practice_extractor")] == ["réparation"]


def test_interpretive_use_said_by_the_interviewee_is_not_flagged():
    item = {"description": "L'enquêtée parle d'une réparation de son image.",
            "evidence": [{"turn_id": "X_T0002", "quote": "j'ai voulu faire une réparation de mon image"}]}
    assert scan_item(item, "interaction_signal_reader") == []


def test_auditor_reason_is_scanned():
    item = {"reason": "Il triche sur sa réponse.", "evidence": [{"turn_id": "X_T0001", "quote": "Oui"}]}
    assert [f["term"] for f in scan_item(item, "speaker_attribution_auditor")] == ["triche"]
