"""Étape 6 — préparation déterministe : matériau, index, représentation normalisée, identifiants. Aucun appel."""

import json

from core import cross_interview as X
from core import cross_interview_corpus as corpus
from core import cross_interview_material as cm
from tests import synthetic_stage6 as S6


def material(ids, uploads=None):
    checked = corpus.check_corpus(uploads or S6.uploads(ids))
    usable = {iid: f[corpus.KIND_TRAJECTORY][2] for iid, f in checked["usable"].items()}
    return cm.build_material(usable, checked["n_total"], checked["mode"]), checked


def test_material_keeps_usable_stage5_objects_with_their_review_reasons():
    m, _ = material(S6.IDS_17)
    assert m["interview_ids"] == S6.IDS_17 and m["aliases"]["ENT_A"] == "I01" and m["aliases"]["ENT_Q"] == "I17"
    assert (m["corpus_n_total"], m["corpus_n_usable"], m["mode"]) == (17, 17, "comparative")
    verif = m["claims"][S6.module_index("ENT_O")["VERIF"]["claims"][0]]
    assert verif["needs_review"] is True and verif["review_reasons"] == ["SINGLE_SUPPORT_PATTERN"]
    assert verif["claim_type"] == "recurring_accounting_move" and verif["move_types"] == ["appeal_to_verification"]
    temporal = m["claims"][S6.module_index("ENT_M")["TEMPORAL"]["claims"][0]]
    assert [a["text"] for a in temporal["anchors"]] == ["Au lycée", "Maintenant"]
    effort = m["criteria"][S6.module_index("ENT_A")["EFFORT"]["criteria"][0]]
    assert effort["criterion"] == "effort" and effort["evidence_turn_ids"] == ["ENT_A_T0006", "ENT_A_T0008"]
    assert m["summary"]["review_claim_count"] == 2 and m["summary"]["review_criterion_count"] == 2


def test_rejected_stage5_claims_are_counted_never_sent():
    doc = S6.document("ENT_X_INVALIDE")
    m = cm.build_material({"ENT_X_INVALIDE": doc, "ENT_A": S6.document("ENT_A")}, 2, "exploratory")
    info = m["interviews"]["ENT_X_INVALIDE"]
    assert info["rejected_claim_count"] == 1 and info["rejected_object_ids"] == ["ENT_X_INVALIDE_TC002"]
    assert "ENT_X_INVALIDE_TC002" not in m["claims"]
    payload = cm.serialize_payload(cm.build_payload(m))
    assert "n'existe pas" not in payload


def test_indexes_distinguish_presence_from_non_observation():
    m, _ = material(S6.IDS_8)
    presence = m["indexes"]["family_presence"]
    assert presence["explicit_temporal_change"] == {"explicit_presence": [], "not_observed": S6.IDS_8}
    assert presence["exception"]["explicit_presence"] == ["ENT_A", "ENT_B", "ENT_G"]
    assert "ENT_C" in presence["exception"]["not_observed"]
    criteria = m["indexes"]["criteria_by_interview"]
    labels = [m["criteria"][c]["criterion"] for c in criteria["ENT_H"]]
    assert labels == ["résultat rendu", "travaux notés faits sans l'outil"]
    assert m["indexes"]["interviews_by_move"]["appeal_to_effort"] == ["ENT_A", "ENT_C", "ENT_E", "ENT_F", "ENT_G"]
    assert set(m["indexes"]["boundaries"]["ENT_A"]) == set(m["indexes"]["claims_by_family"]["stable_boundary"]) & set(
        m["interviews"]["ENT_A"]["claim_ids"])
    assert m["indexes"]["ordinary_zones"]["ENT_D"] and m["indexes"]["tensions"]["ENT_C"]


def test_lexical_groups_require_identical_wording():
    m, _ = material(S6.IDS_17)
    groups = {g["key"]: g for g in m["indexes"]["criterion_labels"]}
    assert groups["effort"]["interview_ids"] == ["ENT_A", "ENT_C", "ENT_E", "ENT_F", "ENT_G", "ENT_K", "ENT_N"]
    # « résultat rendu » (ENT_H) n'est jamais rapproché d'« effort » : aucune équivalence supposée
    assert groups["resultat rendu"]["interview_ids"] == ["ENT_H"]
    contexts = {g["key"]: g for g in m["indexes"]["contexts"]}
    assert "dissertations" in contexts and "dissertation" not in contexts
    assert cm.fold_key("Les  Dissertations !") == "les dissertations"
    assert cm.public(m)["grouping_basis"] == "identical_wording"


def test_payload_is_normalized_compact_and_free_of_transcripts():
    m, _ = material(S6.IDS_8)
    payload = cm.build_payload(m)
    text = cm.serialize_payload(payload)
    assert len(payload["interviews"]) == 8 and text.count("\n") == 9  # un entretien par ligne
    first = payload["interviews"][0]
    assert first["id"] == "I01" and first["interview_id"] == "ENT_A"
    assert set(first["claims"]) == {"TC001", "TC002", "TC003", "TC004", "TC005"}
    assert "confidence" not in first["claims"]["TC002"]          # « medium » omis
    assert first["claims"]["TC001"]["confidence"] == "high"
    assert "review" not in first["claims"]["TC001"]
    for absent in ("comment ça se passe", "generated_at", "source_hashes", "trajectory_summary", "prompt_sha256",
                   "ENT_A_TC001", "validation_status"):
        assert absent not in text
    assert payload["indexes"]["families"]["explicit_temporal_change"] == {"not_observed": [f"I{n:02d}"
                                                                                           for n in range(1, 9)]}
    shared = {tuple(g["labels"]): g["refs"] for g in payload["indexes"]["shared_criterion_labels"]}
    assert shared[("effort",)] == ["I01:RC002", "I03:RC002", "I05:RC003", "I06:RC001", "I07:RC002"]


def test_payload_is_deterministic_and_independent_of_upload_order():
    first, _ = material(S6.IDS_8)
    second, _ = material(None, list(reversed(S6.uploads(S6.IDS_8))))
    assert cm.serialize_payload(cm.build_payload(first)) == cm.serialize_payload(cm.build_payload(second))
    assert X.build_request(first) == X.build_request(second)


def test_expand_ids_restores_full_identifiers_and_leaves_unknown_values():
    m, _ = material(S6.IDS_4)
    output = {"cross_case_claims": [{
        "support": [{"interview_id": "I02", "claim_ids": ["TC001", "I03:TC002"], "criterion_ids": ["RC001"],
                     "evidence_turn_ids": ["T0002"]},
                    {"interview_id": "I99", "claim_ids": ["TC001"], "criterion_ids": [], "evidence_turn_ids": []}],
        "counterexamples": [{"interview_id": "ENT_D", "claim_ids": ["TC001"], "criterion_ids": [],
                             "evidence_turn_ids": []}]}]}
    expanded = cm.expand_ids(output, m)["cross_case_claims"][0]
    assert expanded["support"][0] == {"interview_id": "ENT_B", "claim_ids": ["ENT_B_TC001", "ENT_C_TC002"],
                                      "criterion_ids": ["ENT_B_RC001"], "evidence_turn_ids": ["ENT_B_T0002"]}
    assert expanded["support"][1]["interview_id"] == "I99" and expanded["support"][1]["claim_ids"] == ["TC001"]
    assert expanded["counterexamples"][0]["claim_ids"] == ["ENT_D_TC001"]


def test_payload_sizes_for_2_8_and_17_interviews_stay_under_one_call():
    sizes = {}
    for ids in (S6.IDS_2, S6.IDS_8, S6.IDS_17):
        m, checked = material(ids)
        request = X.build_request(m)
        raw = cm.raw_size(checked["usable"])
        sizes[len(ids)] = (request["payload_chars"], request["estimated_tokens"], raw)
        assert request["payload_chars"] < 0.2 * raw  # ≈ 15 % des triplets bruts de l'étape 5
    assert sizes[2][0] < sizes[8][0] < sizes[17][0]
    assert sizes[17][1] < X.SINGLE_CALL_MAX_INPUT_TOKENS / 2  # ≈ 9 400 tokens estimés pour 17 entretiens
    print("\nTaille de la représentation (caractères, tokens estimés, JSON bruts de l'étape 5) :", json.dumps(sizes))
