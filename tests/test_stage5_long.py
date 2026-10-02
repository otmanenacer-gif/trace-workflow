"""Étape 5 — entretien long « OTMANE-like » (section 19) : payload compact, 1 appel, aucune invention temporelle,
aucune comparaison entre entretiens. agents simulés, aucun appel réel."""

import json

import pytest

from core import config
from core import trajectory_candidates as tc
from tests import synthetic_stage5 as S5
from tests import synthetic_stage5_long as L
from tests.fake_llm import TRAJECTORY

ID = L.INTERVIEW_ID


@pytest.fixture
def stage4(tmp_path):
    return S5.run_to_stage4(tmp_path, L.files(), L.stage3_responders(), L.BUILDER)


def stage5(stage4, mode="good"):
    run, cache = stage4
    run, transport = S5.run_stage5(run, cache, L.mapper(mode))
    return run, transport, S5.load(run, config.STUDENT_TRAJECTORY_FILENAME)


def test_fixture_has_the_properties_of_a_long_real_interview(stage4):
    run, _ = stage4
    assert L.PRACTICE_COUNT == 50
    practices = S5.load(run, "practice_extractor.json")
    assert practices["item_count"] == 50 and practices["chunking"]["chunking_used"] is True
    episodes = S5.load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert episodes["status"] == "SUCCESS" and episodes["episode_count"] == 20
    assert (episodes["accountability_episode_count"], episodes["ordinary_practice_count"], episodes["uncertain_count"],
            episodes["unmarked_practice_count"]) == (17, 2, 1, 23)
    assert episodes["needs_review_count"] == 2  # tour mal attribué + épisode signalé par le modèle
    # la contradiction entre deux tours éloignés (lecture à longue distance) a relié T0030 et T0170
    signals = S5.load(run, "interaction_signals.json")["signals"]
    assert any(s["signal_type"] == "cross_turn_contradiction" for s in signals)


def test_one_call_with_a_compact_payload(stage4):
    run, transport, document = stage5(stage4)
    assert [c["agent"] for c in transport.calls] == [TRAJECTORY]
    message = transport.calls[0]["params"]["messages"][0]["content"]
    manifest = S5.load(run, config.STUDENT_TRAJECTORY_MANIFEST_FILENAME)
    # alternative naïve : les sorties brutes des étapes 3 et 4 que l'étape 5 consomme
    episodes = S5.load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    practices = S5.load(run, "practice_extractor.json")["practices"]
    naive = len(json.dumps([episodes["episodes"], episodes["unmarked_practices"], practices], ensure_ascii=False))
    assert manifest["estimated_input_tokens"] < 6_000 < tc.SINGLE_CALL_MAX_INPUT_TOKENS
    assert manifest["payload_chars"] < 0.25 * naive
    assert manifest["over_single_call_threshold"] is False and manifest["api_calls"] == 0
    assert len(transport.calls) == 1
    for filler in L.FILLERS:  # aucun tour sans pratique, jamais l'entretien complet
        assert filler[:60] not in message
    assert L.SPECIAL_QUESTIONS[169] not in message  # questions de l'enquêteur non transmises
    sent = S5.sent_material(transport.calls[0]["params"])
    assert len(sent["episodes"]) == 20 and len(sent["unmarked_practices"]) == 23
    quotes = [(q["turn"], q["quote"]) for q in sent["quotes_by_id"].values()]
    assert len(quotes) == len(set(quotes))  # chaque citation une seule fois
    print(f"payload : {manifest['payload_chars']} caractères ({len(message)} avec les consignes du message), "
          f"{manifest['estimated_input_tokens']} tokens estimés ; sorties brutes des étapes 3-4 : {naive} caractères")


def test_expected_configuration_is_kept(stage4):
    _, _, document = stage5(stage4)
    assert document["status"] == "SUCCESS" and document["configuration_type"] == "mixed"
    counts = {key: len(document[key]) for key in ("stable_boundaries", "contextual_variations",
                                                  "explicit_temporal_changes", "exceptions", "unresolved_tensions",
                                                  "ordinary_zones", "recurring_accounting_moves")}
    assert counts == {"stable_boundaries": 2, "contextual_variations": 2, "explicit_temporal_changes": 1,
                      "exceptions": 1, "unresolved_tensions": 1, "ordinary_zones": 2, "recurring_accounting_moves": 1}
    [change] = [c for c in document["trajectory_claims"] if c["claim_type"] == "explicit_temporal_change"]
    assert [a["text"] for a in change["validated_temporal_anchors"]] == ["Au lycée", "Maintenant"]
    assert document["student_role_criteria_count"] == 5
    anchors = [(a["turn_id"], a["text"]) for a in document["material"]["temporal_anchors"]]
    assert anchors == [(f"{ID}_T0110", "Au lycée"), (f"{ID}_T0112", "Maintenant"), (f"{ID}_T0112", "ne lui fais plus")]


def test_no_temporal_invention_and_no_interviewer_word(stage4):
    _, _, document = stage5(stage4, "adversarial")
    assert document["explicit_temporal_changes"] == [next(c["claim_id"] for c in document["trajectory_claims"]
                                                          if c["claim_type"] == "explicit_temporal_change")]
    invented = document["trajectory_claims"][-1]
    assert invented["model_claim_type"] == "explicit_temporal_change" and invented["claim_type"] == "contextual_variation"
    culpability = document["student_role_criteria"][-1]
    assert culpability["validation_status"] == "rejected"
    assert "INTERVIEWER_TERM_ATTRIBUTED" in culpability["review_reasons"]
    assert document["student_role_criteria_count"] == 5


def test_no_cross_interview_comparison(stage4):
    run, transport, document = stage5(stage4)
    content = transport.calls[0]["params"]["messages"][0]["content"]
    assert set(__import__("re").findall(r'"interview_id":"([^"]+)"', content)) == {ID}
    assert "corpus" not in document["trajectory_summary"].lower()
    for key in ("typology", "student_types", "corpus", "comparison"):
        assert key not in document
