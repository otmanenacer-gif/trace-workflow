"""Étape 5 — tests sociologiques obligatoires (section 18) : pipeline réel, LLM simulés, aucun appel réel.

A. changement temporel réel ; B. variation contextuelle (jamais une évolution) ; C. règle + exception ;
D. stabilité ; E. zone ordinaire ; F. pas de trajectoire ; G. épisode à revoir ; H. mot de l'enquêteur.
"""

import json

from core import config
from tests import synthetic_stage5 as S5
from tests.fake_llm import TRAJECTORY

TRAJECTORY_FILE = config.STUDENT_TRAJECTORY_FILENAME


def stage5(tmp_path, case, mode="good"):
    run, cache = S5.case_to_stage4(tmp_path, case)
    run, transport = S5.run_stage5(run, cache, case.mapper(mode))
    assert [c["agent"] for c in transport.calls] == [TRAJECTORY]  # 1 appel, étapes 3 et 4 jamais relancées
    return S5.load(run, TRAJECTORY_FILE), S5.load(run, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME), transport


def kept(document, claim_type=None):
    return [c for c in document["trajectory_claims"] if c["usable_for_next_stages"]
            and (claim_type is None or c["claim_type"] == claim_type)]


def codes(claim):
    return set(claim["review_reasons"])


# A. ----------------------------------------------------------------------------------------------------

def test_a_explicit_temporal_change_is_accepted_with_explicit_anchors(tmp_path):
    document, validation, transport = stage5(tmp_path, S5.TEMPORAL)
    sent = S5.sent_material(transport.calls[0]["params"])
    # ancrages repérés dans les tours de l'enquêté seulement (pas « aujourd'hui » de la question de l'enquêteur)
    assert [(a["text"], a["kind"]) for a in sent["anchors_by_id"].values()] == [("Au lycée", "period"),
                                                                                 ("Maintenant", "present")]
    assert document["status"] == "SUCCESS" and document["configuration_type"] == "temporal_trajectory"
    [change] = kept(document, "explicit_temporal_change")
    assert change["validation_status"] == "valid" and change["needs_review"] is False
    assert change["temporal_ordering_basis"] == "past_present"
    assert [a["text"] for a in change["validated_temporal_anchors"]] == ["Au lycée", "Maintenant"]
    assert document["explicit_temporal_changes"] == [change["claim_id"]]
    assert validation["error_count"] == validation["warning_count"] == 0


def test_a_without_anchors_the_same_material_is_only_a_contextual_variation(tmp_path):
    case = S5.TEMPORAL
    scripted = json.loads(json.dumps(case.good))
    scripted["claims"][0]["anchors"] = []
    run, cache = S5.case_to_stage4(tmp_path, case)
    run, _ = S5.run_stage5(run, cache, S5.scripted_mapper(scripted))
    document = S5.load(run, TRAJECTORY_FILE)
    [claim] = document["trajectory_claims"]
    assert claim["claim_type"] == "contextual_variation" and claim["model_claim_type"] == "explicit_temporal_change"
    assert "TEMPORAL_CHANGE_REQUALIFIED" in codes(claim) and claim["needs_review"] is True
    assert document["explicit_temporal_changes"] == []
    assert document["configuration_type"] == "contextual_configuration"
    assert document["model_configuration_type"] == "temporal_trajectory"


# B. ----------------------------------------------------------------------------------------------------

def test_b_contextual_variation_without_temporality(tmp_path):
    document, _, transport = stage5(tmp_path, S5.CONTEXT)
    assert S5.sent_material(transport.calls[0]["params"])["anchors_by_id"] == {}
    assert document["configuration_type"] == "contextual_configuration"
    [variation] = kept(document, "contextual_variation")
    assert variation["validation_status"] == "valid" and len(variation["support_ids"]) == 2
    assert document["explicit_temporal_changes"] == []


def test_b_an_evolution_invented_from_the_interview_order_is_requalified(tmp_path):
    document, validation, _ = stage5(tmp_path, S5.CONTEXT, "adversarial")
    invented, exception = document["trajectory_claims"]
    assert invented["model_claim_type"] == "explicit_temporal_change"
    assert invented["claim_type"] == "contextual_variation"
    assert {"TEMPORAL_CHANGE_REQUALIFIED", "TEMPORAL_WORDING_WITHOUT_ANCHOR", "INTERVIEW_ORDER_AS_TIME"} <= codes(invented)
    temporal_issue = next(i for i in validation["issues"] if i["code"] == "TEMPORAL_CHANGE_REQUALIFIED")
    assert "ordering_anchors" in temporal_issue["missing"]
    # « exception » sans règle ni cas : une simple variation (et sur un seul contexte)
    assert exception["claim_type"] == "contextual_variation" and "EXCEPTION_REQUALIFIED" in codes(exception)
    assert document["explicit_temporal_changes"] == [] and document["exceptions"] == []
    assert document["configuration_type"] == "contextual_configuration"
    assert document["model_configuration_type"] == "temporal_trajectory"
    assert {"CONFIGURATION_REQUALIFIED", "SUMMARY_TEMPORAL_WORDING"} <= {i["code"] for i in validation["issues"]}
    assert document["needs_review"] is True and document["status"] == "SUCCESS_WITH_WARNINGS"


# C. ----------------------------------------------------------------------------------------------------

def test_c_rule_and_exception_are_described_without_resolving_the_tension(tmp_path):
    document, _, transport = stage5(tmp_path, S5.EXCEPTION)
    sent = S5.sent_material(transport.calls[0]["params"])
    assert sent["regularities"]["rule_and_case"] == [{"basis": "same_item", "context": "plans", "items": ["E001"]}]
    assert sent["regularities"]["signaled_tensions"] == ["E001"]
    [exception] = kept(document, "exception")
    [tension] = kept(document, "unresolved_tension")
    assert exception["validation_status"] == tension["validation_status"] == "valid"
    assert document["exceptions"] == [exception["claim_id"]] and document["unresolved_tensions"] == [tension["claim_id"]]
    assert {c["criterion"] for c in document["student_role_criteria"]} == {"être l'auteur de son plan", "temps disponible"}


def test_c_resolving_the_tension_or_judging_the_student_is_flagged(tmp_path):
    document, _, _ = stage5(tmp_path, S5.EXCEPTION, "adversarial")
    judged, fiches = document["trajectory_claims"]
    assert {"RESOLVING_VOCABULARY", "INTERPRETIVE_VOCABULARY"} <= codes(judged) and judged["needs_review"] is True
    # une variation (les fiches de lecture) n'est pas une exception : ni règle ni cas
    assert fiches["claim_type"] == "contextual_variation" and "EXCEPTION_REQUALIFIED" in codes(fiches)


# D. ----------------------------------------------------------------------------------------------------

def test_d_stable_boundary_and_recurring_move(tmp_path):
    document, _, transport = stage5(tmp_path, S5.STABLE)
    sent = S5.sent_material(transport.calls[0]["params"])
    assert sent["regularities"]["repeated_moves"] == {"restriction": ["E001", "E002", "E003"]}
    [boundary] = kept(document, "stable_boundary")
    [recurring] = kept(document, "recurring_accounting_move")
    assert boundary["validation_status"] == recurring["validation_status"] == "valid"
    assert len(boundary["support_ids"]) == 3 and recurring["accounting_move_types"] == ["restriction"]
    [criterion] = document["student_role_criteria"]
    assert criterion["criterion"] == "temps disponible" and criterion["validation_status"] == "valid"
    assert document["configuration_type"] == "contextual_configuration"


# E. ----------------------------------------------------------------------------------------------------

def test_e_ordinary_zone_of_unjustified_information_searches(tmp_path):
    document, _, transport = stage5(tmp_path, S5.ORDINARY)
    sent = S5.sent_material(transport.calls[0]["params"])
    assert len(sent["unmarked_practices"]) == 3 and sent["regularities"]["ordinary_items"] == ["P001", "P002", "P003"]
    [zone] = kept(document, "ordinary_zone")
    assert zone["validation_status"] == "valid" and len(zone["support_ids"]) == 3
    assert document["ordinary_zones"] == [zone["claim_id"]]


def test_e_an_ordinary_zone_never_contains_an_accountability_episode(tmp_path):
    document, _, _ = stage5(tmp_path, S5.ORDINARY, "adversarial")
    mixed, single = document["trajectory_claims"]
    assert "ORDINARY_ZONE_WITH_ACCOUNTABILITY" in codes(mixed) and mixed["needs_review"] is True
    assert "SINGLE_SUPPORT_PATTERN" in codes(single)


# F. ----------------------------------------------------------------------------------------------------

def test_f_no_pattern_no_invented_evolution(tmp_path):
    run, cache = S5.case_to_stage4(tmp_path, S5.NOPATTERN)
    episodes = S5.load(run, config.ACCOUNTABILITY_EPISODES_FILENAME)
    assert episodes["candidate_count"] == 0 and episodes["unmarked_practice_count"] == 4  # étape 4 : 0 appel
    run, transport = S5.run_stage5(run, cache, S5.NOPATTERN.mapper())
    document = S5.load(run, TRAJECTORY_FILE)
    assert len(transport.calls) == 1 and document["configuration_type"] == "no_clear_pattern"
    assert document["trajectory_claims"] == [] and document["status"] == "SUCCESS"


def test_f_an_invented_trajectory_is_never_kept(tmp_path):
    document, validation, _ = stage5(tmp_path, S5.NOPATTERN, "adversarial")
    assert document["configuration_type"] not in ("temporal_trajectory", "mixed")
    assert document["explicit_temporal_changes"] == []
    [claim] = document["trajectory_claims"]
    assert claim["claim_type"] == "contextual_variation" and "TEMPORAL_CHANGE_REQUALIFIED" in codes(claim)
    summary_issues = {i["code"]: i for i in validation["issues"] if i["object_id"] == S5.NOPATTERN.interview_id}
    assert "SUMMARY_TEMPORAL_WORDING" in summary_issues
    assert {"prise de conscience", "maturation / maturité"} <= set(summary_issues["SUMMARY_VOCABULARY"]["terms"])


# G. ----------------------------------------------------------------------------------------------------

def test_g_a_claim_resting_only_on_a_review_episode_is_needs_review(tmp_path):
    document, _, transport = stage5(tmp_path, S5.REVIEW)
    sent = S5.sent_material(transport.calls[0]["params"])
    warned = [e for e in sent["episodes"] if e.get("review")]
    assert len(warned) == 1 and warned[0]["speaker_warning"] == ["T0004"]
    only_review, mixed = document["trajectory_claims"]
    assert only_review["model_needs_review"] is False and only_review["needs_review"] is True
    assert {"SUPPORTED_ONLY_BY_REVIEW_EPISODES", "STRONG_CLAIM_ON_REVIEW_EPISODES"} <= codes(only_review)
    assert only_review["speaker_warnings"][0]["turn_id"].endswith("_T0004")
    # appuyée aussi sur un épisode propre : l'épisode à revoir est propagé, la revue n'est pas forcée
    assert mixed["needs_review"] is False and mixed["review_episode_ids"] == only_review["support_ids"]
    assert mixed["speaker_warnings"] and mixed["validation_status"] == "valid"


# H. ----------------------------------------------------------------------------------------------------

def test_h_interviewer_affect_is_never_attributed_to_the_student(tmp_path):
    document, _, _ = stage5(tmp_path, S5.INTERVIEWER)
    text = json.dumps([document["trajectory_claims"], document["student_role_criteria"],
                       document["trajectory_summary"]], ensure_ascii=False).lower()
    assert "honte" not in text and "peur" not in text
    [criterion] = document["student_role_criteria"]
    assert criterion["validation_status"] == "valid"


def test_h_interviewer_affect_as_criterion_or_claim_is_rejected(tmp_path):
    document, validation, _ = stage5(tmp_path, S5.INTERVIEWER, "adversarial")
    [claim] = document["trajectory_claims"]
    [criterion] = document["student_role_criteria"]
    assert claim["validation_status"] == criterion["validation_status"] == "rejected"
    assert "INTERVIEWER_TERM_ATTRIBUTED" in codes(claim)
    assert {"INTERVIEWER_TERM_ATTRIBUTED", "INTERPRETIVE_CRITERION"} <= codes(criterion)
    assert document["contextual_variations"] == [] and document["student_role_criteria_count"] == 0
    summary = next(i for i in validation["issues"] if i["code"] == "SUMMARY_VOCABULARY")
    assert summary["from_interviewer"] == ["honte"]


def test_student_own_word_is_not_flagged(tmp_path):
    """Entretien de référence de l'étape 4 : « J'aurais peur que le professeur pense… » est dit par l'étudiant."""
    from tests import synthetic_stage4 as S4
    run, cache = S5.run_to_stage4(tmp_path, S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    run, _ = S5.run_stage5(run, cache, S5.scripted_mapper(S5.REFERENCE_PLAN))
    document = S5.load(run, TRAJECTORY_FILE)
    judgment = next(c for c in document["student_role_criteria"] if c["criterion"] == "jugement du professeur")
    assert "peur" in judgment["description"] and judgment["validation_status"] == "valid"
    assert document["status"] == "SUCCESS" and document["exceptions"] and document["ordinary_zones"]
