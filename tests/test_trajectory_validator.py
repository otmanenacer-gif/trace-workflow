"""Étape 5 — validateur déterministe (core/trajectory_validator.py), sur de vraies sorties d'étape 4. Aucun LLM."""

import pytest

from core import trajectory
from core import trajectory_validator as v
from tests import synthetic_stage5 as S5

T = S5.TEMPORAL  # T0002 « Au lycée … » (enquêté) ; T0003 « Et aujourd'hui… » (enquêteur) ; T0004 « Maintenant … »
ID = T.interview_id
E1, P_UNMARKED = f"{ID}_E001", f"{ID}_P003"


@pytest.fixture
def prepared(tmp_path):
    run, _ = S5.case_to_stage4(tmp_path, T)
    return trajectory.prepare_stage5(run["files"][0]["ingestion"])


def claim(claim_type="contextual_variation", description="L'étudiant décrit deux usages.", episodes=(E1,),
          practices=(), evidence=(2, 4), anchors=(), confidence="medium", needs_review=False, **extra):
    return {"claim_type": claim_type, "description": description, "episode_ids": list(episodes),
            "practice_ids": list(practices), "contexts": extra.pop("contexts", []),
            "accounting_move_types": extra.pop("moves", []),
            "temporal_anchors": [{"text": text, "turn_id": T.tid(n)} for n, text in anchors],
            "evidence_turn_ids": [n if isinstance(n, str) else T.tid(n) for n in evidence],
            "confidence": confidence, "needs_review": needs_review}


def criterion(text="faire soi-même les exercices", description="Dans cet entretien, l'étudiant associe l'usage au fait "
              "de faire les exercices lui-même.", episodes=(E1,), evidence=(4,)):
    return {"criterion": text, "description": description, "episode_ids": list(episodes),
            "evidence_turn_ids": [T.tid(n) for n in evidence], "confidence": "medium", "needs_review": False}


def validate(prepared, claims=(), criteria=(), configuration="contextual_configuration", summary="Synthèse."):
    output = {"configuration_type": configuration, "claims": list(claims), "student_role_criteria": list(criteria),
              "trajectory_summary": summary, "confidence": "medium", "needs_review": False, "mapper_notes": None}
    return v.validate_trajectory(output, prepared.material, prepared.transcript, prepared.speaker_warnings)


def only(result):
    [c] = result["document"]["trajectory_claims"]
    return c, set(c["review_reasons"])


# --- Changement temporel ---------------------------------------------------------------------------------

def test_temporal_change_with_ordering_anchors_is_kept(prepared):
    c, reasons = only(validate(prepared, [claim("explicit_temporal_change", anchors=[(2, "Au lycée"), (4, "Maintenant")])],
                               configuration="temporal_trajectory"))
    assert c["claim_type"] == "explicit_temporal_change" and c["validation_status"] == "valid" and not reasons


@pytest.mark.parametrize("anchors, code, missing", [
    ([], None, ["ordering_anchors"]),                                     # ordre des tours seulement
    ([(2, "Au lycée")], None, ["ordering_anchors"]),                     # un seul état daté
    ([(2, "Au lycée"), (3, "aujourd'hui")], "ANCHOR_FROM_INTERVIEWER", ["ordering_anchors"]),
    ([(2, "Au lycée"), (4, "les exercices")], "ANCHOR_NOT_TEMPORAL", ["ordering_anchors"]),
])
def test_temporal_change_without_sufficient_anchors_is_requalified(prepared, anchors, code, missing):
    result = validate(prepared, [claim("explicit_temporal_change", anchors=anchors)], configuration="temporal_trajectory")
    c, reasons = only(result)
    assert c["claim_type"] == "contextual_variation" and c["model_claim_type"] == "explicit_temporal_change"
    assert "TEMPORAL_CHANGE_REQUALIFIED" in reasons and c["needs_review"] is True and c["usable_for_next_stages"]
    assert code is None or code in reasons
    issue = next(i for i in result["report"]["issues"] if i["code"] == "TEMPORAL_CHANGE_REQUALIFIED")
    assert issue["missing"] == missing
    assert result["document"]["configuration_type"] == "contextual_configuration"
    assert result["document"]["explicit_temporal_changes"] == []


def test_fabricated_anchor_rejects_the_claim(prepared):
    result = validate(prepared, [claim("explicit_temporal_change", anchors=[(2, "L'année dernière"), (4, "Maintenant")])],
                      configuration="temporal_trajectory")
    c, reasons = only(result)
    assert c["validation_status"] == "rejected" and "ANCHOR_NOT_FOUND" in reasons
    assert result["document"]["configuration_type"] == "no_clear_pattern"  # plus aucune affirmation retenue


def test_two_states_are_required(prepared):
    """Un seul état (une pratique sans marqueur), même daté par deux expressions : pas de changement."""
    material = prepared.material
    one_state = [p for p, item in material["items"].items() if item["kind"] == "unmarked"]
    c, reasons = only(validate(prepared, [claim("explicit_temporal_change", episodes=(), practices=one_state,
                                                evidence=(6,), anchors=[(2, "Au lycée"), (4, "Maintenant")])]))
    assert c["claim_type"] == "contextual_variation" and "ANCHOR_OUTSIDE_MATERIAL" in reasons


# --- Appuis et tours ----------------------------------------------------------------------------------------

def test_unknown_or_unusable_episode_rejects_the_claim(prepared):
    prepared.material["excluded_episodes"].append({"episode_id": f"{ID}_E009", "validation_status": "rejected",
                                                   "review_reasons": ["DISCONNECTED_MERGE"]})
    result = validate(prepared, [claim(episodes=(f"{ID}_E009",)), claim(episodes=(f"{ID}_E077",))])
    unusable, unknown = result["document"]["trajectory_claims"]
    assert {"UNUSABLE_EPISODE", "NO_SUPPORT"} <= set(unusable["review_reasons"])
    assert "UNKNOWN_EPISODE_ID" in unknown["review_reasons"]
    assert unusable["validation_status"] == unknown["validation_status"] == "rejected"
    assert result["report"]["rejected_claim_ids"] == [f"{ID}_TC001", f"{ID}_TC002"]


def test_practice_of_an_episode_is_resolved_to_that_episode(prepared):
    episode_practice = prepared.material["items"][E1]["practice_ids"][0]
    c, reasons = only(validate(prepared, [claim(episodes=(), practices=(episode_practice, P_UNMARKED), evidence=(2, 6))]))
    assert c["support_ids"] == [E1, P_UNMARKED] and c["validation_status"] == "valid"


def test_interviewer_turns_are_not_a_position(prepared):
    c, reasons = only(validate(prepared, [claim(evidence=(3,))]))
    assert {"NO_INTERVIEWEE_EVIDENCE", "EVIDENCE_TURN_INTERVIEWER"} <= reasons and c["validation_status"] == "rejected"


def test_unknown_or_foreign_turns_are_errors(prepared):
    result = validate(prepared, [claim(evidence=("AUTRE_ENTRETIEN_T0002",)), claim(evidence=(99,))])
    foreign, unknown = result["document"]["trajectory_claims"]
    assert "FOREIGN_INTERVIEW_TURN" in foreign["review_reasons"] and "UNKNOWN_EVIDENCE_TURN" in unknown["review_reasons"]


def test_turn_outside_the_supports_is_flagged(prepared):
    c, reasons = only(validate(prepared, [claim(evidence=(2, 6))]))
    assert "EVIDENCE_TURN_OUTSIDE_MATERIAL" in reasons and c["validation_status"] == "needs_review"


# --- Exception, régularités ------------------------------------------------------------------------------

def _item(id_, task, use, rule=False, case=False):
    return {"id": id_, "rule": rule, "case": case, "facts": [{"task_key": task, "use_status": use}]}


def test_exception_requires_a_rule_and_a_case():
    assert v.exception_requirements([_item("A", "plan", "refusal", rule=True, case=True)])["missing"] == []
    # conduite contraire documentée : refus d'un appui, usage d'un autre, même tâche
    assert v.exception_requirements([_item("A", "plan", "refusal", rule=True), _item("B", "plan", "past_use")])[
        "missing"] == []
    assert v.exception_requirements([_item("A", "plan", "refusal", rule=True), _item("B", "fiche", "use")])[
        "missing"] == ["case"]
    assert v.exception_requirements([_item("B", "plan", "use", case=True)])["missing"] == ["rule"]


def test_pattern_claims_need_several_supports(prepared):
    result = validate(prepared, [claim("stable_boundary"), claim("recurring_accounting_move", moves=["restriction"]),
                                 claim("ordinary_zone", episodes=(), practices=(P_UNMARKED,), evidence=(6,)),
                                 claim("unresolved_tension", evidence=(2,))])
    reasons = [set(c["review_reasons"]) for c in result["document"]["trajectory_claims"]]
    assert "SINGLE_SUPPORT_PATTERN" in reasons[0] and "SINGLE_SUPPORT_PATTERN" in reasons[1]
    assert "SINGLE_SUPPORT_PATTERN" in reasons[2] and "TENSION_SINGLE_FORMULATION" in reasons[3]


# --- Vocabulaire ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("description, code", [
    ("L'étudiant fait preuve de maturation dans son usage.", "INTERPRETIVE_VOCABULARY"),
    ("Une prise de conscience progressive apparaît.", "INTERPRETIVE_VOCABULARY"),
    ("L'étudiant apprend à mieux utiliser l'IA.", "INTERPRETIVE_VOCABULARY"),
    ("Une stratégie de légitimation de son usage.", "INTERPRETIVE_VOCABULARY"),
    ("Sa dépendance à l'outil diminue.", "INTERPRETIVE_VOCABULARY"),
    ("Sa véritable règle est de faire faire.", "RESOLVING_VOCABULARY"),
    ("Comme beaucoup d'étudiants du corpus, il varie selon les tâches.", "CROSS_INTERVIEW_COMPARISON"),
    ("L'usage évolue selon la tâche.", "TEMPORAL_WORDING_WITHOUT_ANCHOR"),
])
def test_forbidden_vocabulary_is_flagged(prepared, description, code):
    c, reasons = only(validate(prepared, [claim(description=description)]))
    assert code in reasons and c["needs_review"] is True


def test_interview_order_presented_as_time_is_flagged(prepared):
    c, reasons = only(validate(prepared, [claim(description="Au fil de l'entretien, l'étudiant devient plus prudent.")]))
    assert {"INTERVIEW_ORDER_AS_TIME", "TEMPORAL_WORDING_WITHOUT_ANCHOR"} <= reasons


# --- Critères du métier d'étudiant --------------------------------------------------------------------------

def test_situated_grounded_criterion_is_valid(prepared):
    [c] = validate(prepared, criteria=[criterion()])["document"]["student_role_criteria"]
    assert c["validation_status"] == "valid" and c["criterion_id"] == f"{ID}_RC001"


@pytest.mark.parametrize("text, description, code", [
    ("faire soi-même", "Le métier d'étudiant consiste à faire soi-même ses exercices.", "CRITERION_GENERALIZED"),
    ("faire soi-même", "L'étudiant associe l'usage au fait de faire lui-même.", "CRITERION_NOT_SITUATED"),
    ("éviter la culpabilité", "Dans cet entretien, l'étudiant associe l'usage à sa culpabilité.", "INTERPRETIVE_CRITERION"),
])
def test_criterion_wording(prepared, text, description, code):
    [c] = validate(prepared, criteria=[criterion(text, description)])["document"]["student_role_criteria"]
    assert code in c["review_reasons"]
    assert c["validation_status"] == ("rejected" if code == "INTERPRETIVE_CRITERION" else "needs_review")


def test_criterion_needs_an_episode_that_formulates_it(prepared):
    prepared.material["items"][E1].update(moves=[], boundaries=[], role=None)
    [c] = validate(prepared, criteria=[criterion()])["document"]["student_role_criteria"]
    assert "CRITERION_NOT_GROUNDED" in c["review_reasons"]


# --- Épisodes à revoir, configuration -------------------------------------------------------------------------

def test_review_propagation(prepared):
    prepared.material["items"][E1]["needs_review"] = True
    result = validate(prepared, [claim(confidence="high"), claim(episodes=(E1,), practices=(P_UNMARKED,), evidence=(2, 6))])
    only_review, mixed = result["document"]["trajectory_claims"]
    assert only_review["needs_review"] is True and only_review["model_needs_review"] is False
    assert {"SUPPORTED_ONLY_BY_REVIEW_EPISODES", "STRONG_CLAIM_ON_REVIEW_EPISODES"} <= set(only_review["review_reasons"])
    assert mixed["review_episode_ids"] == [E1] and "SUPPORTED_ONLY_BY_REVIEW_EPISODES" not in mixed["review_reasons"]


def test_configuration_consistency(prepared):
    empty = validate(prepared, configuration="contextual_configuration")["document"]
    assert empty["configuration_type"] == "no_clear_pattern"
    temporal = claim("explicit_temporal_change", anchors=[(2, "Au lycée"), (4, "Maintenant")])
    result = validate(prepared, [temporal], configuration="no_clear_pattern")
    assert result["document"]["configuration_type"] == "no_clear_pattern"
    assert "CONFIGURATION_INCONSISTENT" in {i["code"] for i in result["report"]["issues"]}


def test_lists_reference_claim_ids_without_duplicating_objects(prepared):
    document = validate(prepared, [claim("explicit_temporal_change", anchors=[(2, "Au lycée"), (4, "Maintenant")]),
                                   claim(episodes=(E1,), practices=(P_UNMARKED,), evidence=(2, 6))],
                        configuration="mixed")["document"]
    assert document["explicit_temporal_changes"] == [f"{ID}_TC001"]
    assert document["contextual_variations"] == [f"{ID}_TC002"]
    assert document["configuration_type"] == "mixed"
