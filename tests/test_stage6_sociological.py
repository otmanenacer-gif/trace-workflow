"""Étape 6 — cas sociologiques A à J (section 21 de la consigne). Pipeline réel (import, préparation, validateur),
Comparator simulé ; versions « adverses » pour vérifier ce que TRACE refuse. Aucun appel réel."""

import copy
import json

import pytest

from core import cross_interview as X
from core.analysis_cache import AnalysisCache
from tests import synthetic_stage6 as S6
from tests.fake_llm import COMPARATOR, FakeTransport, fake_settings


def run(tmp_path, ids, plan=S6.GOOD_PLAN, mutate=None, extra=()):
    transport = FakeTransport({COMPARATOR: S6.scripted_comparator(plan, mutate=mutate)})
    manifest = X.run_stage6(S6.uploads([*ids, *extra]), settings=fake_settings(), transport=transport,
                            cache=AnalysisCache(tmp_path / "cache"), base_dir=tmp_path / "out")
    outputs = X.read_outputs(manifest["corpus_dir"])
    document = json.loads(outputs["comparison"]) if outputs["comparison"] else None
    return manifest, document, json.loads(outputs["validation"]), transport


def find(document, claim_type, model_type=None):
    return [c for c in document["cross_case_claims"] if c["claim_type"] == claim_type
            and (model_type is None or c.get("model_claim_type") == model_type)]


def codes(validation, object_id):
    return {i["code"] for i in validation["issues"] if i["object_id"] == object_id}


# A. trois entretiens partagent « IA pour rédiger seulement quand manque de temps » -----------------------------

def test_A_recurring_boundary_in_three_interviews(tmp_path):
    manifest, document, _, transport = run(tmp_path, S6.IDS_4)
    assert len(transport.calls) == 1 and manifest["status"] in X.DONE_STATUSES
    boundary = document["cross_case_claims"][0]
    assert boundary["claim_type"] == "recurring_boundary" and boundary["cross_claim_id"] in document[
        "recurring_boundaries"]
    assert boundary["interview_ids"] == ["ENT_A", "ENT_B", "ENT_C"] and boundary["n_supporting_interviews"] == 3
    assert (boundary["corpus_n_total"], boundary["corpus_n_usable"]) == (4, 4)
    assert all(s["claim_ids"] and s["evidence_turn_ids"] for s in boundary["support"])
    assert boundary["validation_status"] == "valid"


# B. un quatrième : « je l'utilise même quand j'ai le temps » → cas négatif conservé ------------------------------

def test_B_negative_case_is_kept(tmp_path):
    _, document, _, _ = run(tmp_path, S6.IDS_4)
    boundary = document["cross_case_claims"][0]
    assert boundary["interview_positions"]["ENT_D"] == "contrary_case"
    assert [c["interview_id"] for c in boundary["counterexamples"]] == ["ENT_D"]
    negative = [n for n in document["negative_cases"] if n["interview_id"] == "ENT_D"]
    assert len(negative) == 1 and negative[0]["related_cross_claim_ids"] == [boundary["cross_claim_id"]]
    assert set(negative[0]["relations"]) == {"contrary_case", "negative_case"}


def test_B_negative_case_survives_a_comparator_that_drops_the_negative_case_claim(tmp_path):
    def drop(output, material):
        output["cross_case_claims"] = [c for c in output["cross_case_claims"] if c["claim_type"] != "negative_case"]
        return output
    _, document, _, _ = run(tmp_path, S6.IDS_4, mutate=drop)
    assert document["negative_case_claims"] == []
    assert [n["interview_id"] for n in document["negative_cases"]] == ["ENT_D"]  # conservé via le contre-exemple


def test_B_negative_case_survives_when_its_regularity_is_rejected(tmp_path):
    def typology(output, material):
        output["cross_case_claims"][0]["description"] = "Les étudiants de type pressé délèguent la rédaction."
        return output
    _, document, validation, _ = run(tmp_path, S6.IDS_4, mutate=typology)
    assert document["cross_case_claims"][0]["validation_status"] == "rejected"
    assert "TYPOLOGY_OF_PERSONS" in codes(validation, "CC001")
    assert "CC001" not in document["recurring_boundaries"]
    assert "ENT_D" in {n["interview_id"] for n in document["negative_cases"]}


# C. critère d'effort dans plusieurs entretiens --------------------------------------------------------------------

def test_C_recurring_student_role_criterion(tmp_path):
    _, document, _, _ = run(tmp_path, S6.IDS_8)
    pattern = next(p for p in document["student_role_criterion_patterns"] if p["criterion_label"] == "effort")
    assert pattern["claim_type"] == "recurring_student_role_criterion"
    assert pattern["interview_ids"] == ["ENT_A", "ENT_C", "ENT_E", "ENT_F", "ENT_G"]
    assert pattern["n_supporting_interviews"] == 5 and pattern["corpus_n_usable"] == 8
    assert all(ids and ids[0].startswith(iid) for iid, ids in pattern["criterion_ids"].items())
    assert all(turns for turns in pattern["evidence"].values())
    assert pattern["cross_claim_id"] in document["recurring_student_role_criteria"]


# D. un entretien où l'effort n'est jamais mentionné → non observé, PAS absence ---------------------------------------

def test_D_effort_not_mentioned_is_not_observed(tmp_path):
    _, document, _, _ = run(tmp_path, S6.IDS_8)
    pattern = next(p for p in document["student_role_criterion_patterns"] if p["criterion_label"] == "effort")
    assert "ENT_B" in pattern["positions"]["not_observed"] and "ENT_D" in pattern["positions"]["not_observed"]
    assert "ENT_B" not in pattern["positions"]["explicit_refusal"]
    assert all(c["interview_id"] != "ENT_B" for c in pattern["contrasts"])


def test_D_comparator_turning_silence_into_absence_is_corrected(tmp_path):
    def absence(output, material):
        alias = {e["interview_id"]: e["id"] for e in material["interviews"]}
        claim = next(c for c in output["cross_case_claims"] if c["criterion_label"] == "effort")
        claim["counterexamples"].append({"interview_id": alias["ENT_B"], "relation": "explicit_refusal",
                                         "description": "ENT_B ne possède pas ce critère.", "claim_ids": [],
                                         "criterion_ids": [], "evidence_turn_ids": []})
        return output
    _, document, validation, _ = run(tmp_path, S6.IDS_8, mutate=absence)
    claim = next(c for c in document["cross_case_claims"] if c["criterion_label"] == "effort")
    assert claim["interview_positions"]["ENT_B"] == "not_observed"
    assert {"COUNTEREXAMPLE_WITHOUT_SUPPORT", "NOT_OBSERVED_AS_ABSENCE"} <= codes(validation, claim["cross_claim_id"])
    assert claim["needs_review"] is True


# E. « l'effort n'est pas important pour moi » → refus explicite / contraste ------------------------------------------

def test_E_explicit_refusal_of_effort(tmp_path):
    _, document, _, _ = run(tmp_path, S6.IDS_8)
    pattern = next(p for p in document["student_role_criterion_patterns"] if p["criterion_label"] == "effort")
    assert pattern["positions"]["explicit_refusal"] == ["ENT_H"]
    assert pattern["contrasts"][0]["relation"] == "explicit_refusal"
    assert pattern["contrasts"][0]["support_ids"] == [S6.module_index("ENT_H")["NO_EFFORT"]["criteria"][0]]
    negative = next(n for n in document["negative_cases"] if n["interview_id"] == "ENT_H")
    assert negative["relations"] == ["explicit_refusal"] and negative["validation_status"] == "supported"


# F. variation maths / écriture : association descriptive, N et cas, aucune causalité ---------------------------------

def test_F_maths_writing_association_is_descriptive(tmp_path):
    _, document, validation, _ = run(tmp_path, S6.IDS_17)
    association = find(document, "contextual_association")[0]
    assert association["interview_ids"] == ["ENT_E", "ENT_G", "ENT_I", "ENT_J", "ENT_K"]
    assert association["n_supporting_interviews"] == 5 and association["corpus_n_usable"] == 17
    assert association["interview_positions"]["ENT_L"] == "contrary_case"
    assert codes(validation, association["cross_claim_id"]) == set()
    assert association["cross_claim_id"] in document["contextual_associations"]


def test_F_causal_or_disciplinary_explanation_is_flagged(tmp_path):
    def causal(output, material):
        claim = next(c for c in output["cross_case_claims"] if c["claim_type"] == "contextual_association")
        claim["description"] = ("La discipline explique la différence : les étudiants en maths utilisent l'outil pour "
                                "les réponses, ce qui s'explique par leur personnalité.")
        return output
    _, document, validation, _ = run(tmp_path, S6.IDS_17, mutate=causal)
    association = find(document, "contextual_association")[0]
    found = codes(validation, association["cross_claim_id"])
    assert {"CAUSAL_CLAIM", "GENERALIZATION_BEYOND_N", "PERSON_OR_SOCIAL_ATTRIBUTION"} <= found
    assert association["validation_status"] == "rejected"


# G. tous les supports d'un pattern sont à revoir -------------------------------------------------------------------

def test_G_review_only_pattern_needs_review_with_limited_confidence(tmp_path):
    _, document, _, _ = run(tmp_path, S6.IDS_17)
    verification = next(c for c in document["cross_case_claims"] if c["interview_ids"] == ["ENT_O", "ENT_P"])
    assert verification["needs_review"] is True and verification["confidence"] == "low"
    assert verification["model_confidence"] == "high" and verification["n_review_only_interviews"] == 2
    assert "SUPPORTED_ONLY_BY_REVIEW_ITEMS" in verification["review_reasons"]
    assert verification["stage5_review_reasons"] == ["SINGLE_SUPPORT_PATTERN"]  # propagées, jamais effacées
    assert verification["cross_claim_id"] in document["recurring_accounting_moves"]


# H. configuration temporelle dans 2 cas seulement --------------------------------------------------------------------

def test_H_temporal_pattern_in_two_of_seventeen(tmp_path):
    _, document, _, _ = run(tmp_path, S6.IDS_17)
    temporal = find(document, "temporal_pattern")[0]
    assert temporal["interview_ids"] == ["ENT_M", "ENT_N"] and temporal["n_supporting_interviews"] == 2
    assert temporal["n_not_observed_interviews"] == 15 and temporal["corpus_n_usable"] == 17
    assert document["configuration_distribution"]["mixed"]["share"] == "2 entretien(s) sur 17 exploitable(s)"


def test_H_generalized_temporal_claim_is_flagged(tmp_path):
    def generalize(output, material):
        claim = next(c for c in output["cross_case_claims"] if c["claim_type"] == "temporal_pattern")
        claim["description"] = "Les étudiants abandonnent généralement la rédaction par l'outil après le lycée."
        return output
    _, document, validation, _ = run(tmp_path, S6.IDS_17, mutate=generalize)
    temporal = find(document, "temporal_pattern")[0]
    assert "GENERALIZATION_BEYOND_N" in codes(validation, temporal["cross_claim_id"])
    assert temporal["needs_review"] is True


# I. deux entretiens seulement → exploratoire -------------------------------------------------------------------------

def test_I_two_interviews_is_exploratory(tmp_path):
    manifest, document, validation, transport = run(tmp_path, S6.IDS_2)
    assert len(transport.calls) == 1
    assert manifest["mode"] == document["mode"] == "exploratory" and document["exploratory"] is True
    assert (document["corpus_n_total"], document["corpus_n_usable"]) == (2, 2)
    assert "EXPLORATORY_CORPUS" in codes(validation, "corpus")
    assert "mode EXPLORATOIRE (deux entretiens seulement)" in transport.calls[0]["params"]["messages"][0]["content"]
    assert all(c["confidence"] != "high" for c in document["cross_case_claims"])
    boundary = document["cross_case_claims"][0]
    assert boundary["interview_ids"] == ["ENT_A", "ENT_B"] and boundary["model_confidence"] == "high"


# J. un entretien invalide dans un corpus de 5 ------------------------------------------------------------------------

def test_J_invalid_interview_is_excluded_with_its_reason(tmp_path):
    manifest, document, validation, transport = run(tmp_path, S6.IDS_4, extra=["ENT_X_INVALIDE"])
    assert (document["corpus_n_total"], document["corpus_n_usable"]) == (5, 4)
    assert manifest["excluded_interviews"] == [{"interview_id": "ENT_X_INVALIDE", "reasons": [
        "validation_error_count = 2 (manifest 2, validation 2) : une étape 5 avec des erreurs de validation n'est pas "
        "importée."]}]
    assert validation["excluded_interviews"] == manifest["excluded_interviews"]
    assert "ENT_X_INVALIDE" not in transport.calls[0]["params"]["messages"][0]["content"]
    assert "ENT_X_INVALIDE" not in document["interview_ids"]


def test_J_excluded_interview_cannot_support_a_claim(tmp_path):
    def cite_excluded(output, material):
        output["cross_case_claims"][0]["support"].append({"interview_id": "ENT_X_INVALIDE",
                                                          "claim_ids": ["TC001"], "criterion_ids": [],
                                                          "evidence_turn_ids": []})
        return output
    _, document, validation, _ = run(tmp_path, S6.IDS_4, mutate=cite_excluded, extra=["ENT_X_INVALIDE"])
    assert "EXCLUDED_INTERVIEW" in codes(validation, "CC001")
    assert document["cross_case_claims"][0]["validation_status"] == "rejected"


@pytest.mark.parametrize("ids, extra", [(["ENT_A"], ()), (["ENT_A"], ["ENT_X_INVALIDE"]), ([], ())])
def test_fewer_than_two_usable_interviews_blocks_without_any_call(tmp_path, ids, extra):
    transport = FakeTransport({})  # tout appel échouerait
    manifest = X.run_stage6(S6.uploads([*ids, *extra]), settings=fake_settings(), transport=transport,
                            cache=AnalysisCache(tmp_path / "cache"), base_dir=tmp_path / "out")
    assert manifest["status"] == "BLOCKED" and transport.calls == [] and manifest["api_calls"] == 0
    assert manifest["error"]["code"] == "CORPUS_TOO_SMALL"
    outputs = X.read_outputs(manifest["corpus_dir"])
    assert outputs["comparison"] is None and json.loads(outputs["validation"])["available"] is False


def test_no_typology_and_traceable_interview_ids_in_the_good_plan(tmp_path):
    _, document, validation, _ = run(tmp_path, S6.IDS_17)
    errors = [i for i in validation["issues"] if i["severity"] == "error"]
    assert errors == [] and validation["error_count"] == 0
    for claim in document["cross_case_claims"]:
        assert claim["interview_ids"] and all(s["interview_id"] in S6.IDS_17 for s in claim["support"])
        assert claim["n_supporting_interviews"] == len(set(claim["interview_ids"]))
    plan = copy.deepcopy(S6.GOOD_PLAN)
    assert len(plan["claims"]) == len(document["cross_case_claims"])
