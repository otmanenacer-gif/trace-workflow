"""Étape 6 — validateur déterministe : appuis, comptes, positions, revues, requalifications, vocabulaire."""

import pytest

from core import cross_interview_corpus as corpus
from core import cross_interview_material as cm
from core import cross_interview_validator as V
from tests import synthetic_stage6 as S6


def build(ids, extra=()):
    checked = corpus.check_corpus(S6.uploads([*ids, *extra]))
    usable = {iid: f[corpus.KIND_TRAJECTORY][2] for iid, f in checked["usable"].items()}
    excluded = {r["interview_id"] for r in checked["rows"] if r["status"] == corpus.STATUS_EXCLUDED}
    return cm.build_material(usable, checked["n_total"], checked["mode"]), excluded


M8, _ = build(S6.IDS_8)


def oid(iid, module, kind="claims", n=0):
    return S6.module_index(iid)[module][kind][n]


def sup(iid, *object_ids, turns=None, m=M8):
    objects = [m["claims"].get(o) or m["criteria"].get(o) for o in object_ids]
    return {"interview_id": iid, "claim_ids": [o for o in object_ids if "_TC" in o],
            "criterion_ids": [o for o in object_ids if "_RC" in o],
            "evidence_turn_ids": turns if turns is not None else [t for o in objects if o
                                                                  for t in o["evidence_turn_ids"]]}


def counter(iid, relation, description, *object_ids, m=M8):
    return {**sup(iid, *object_ids, m=m), "relation": relation, "description": description}


def draft(claim_type, description, support, counters=(), n=None, confidence="medium", needs_review=False,
          related=(), criterion_label=None, contexts=()):
    support = list(support)
    return {"claim_type": claim_type, "description": description, "criterion_label": criterion_label,
            "support": support, "counterexamples": list(counters), "contexts": list(contexts),
            "n_supporting_interviews": len({s["interview_id"] for s in support}) if n is None else n,
            "related_claim_numbers": list(related), "confidence": confidence, "needs_review": needs_review}


def validate(*drafts, m=M8, excluded=(), summary="Synthèse descriptive.", confidence="medium"):
    return V.validate_comparison({"cross_case_claims": list(drafts), "cross_case_summary": summary,
                                  "confidence": confidence, "needs_review": False, "comparator_notes": None},
                                 m, set(excluded))


def codes(result, object_id="CC001"):
    return [i["code"] for i in result["report"]["issues"] if i["object_id"] == object_id]


def claim(result, n=1):
    return result["document"]["cross_case_claims"][n - 1]


TIME = {i: oid(i, "TIME") for i in ("ENT_A", "ENT_B", "ENT_C", "ENT_E")}
TIME_DRAFT = draft("recurring_boundary", "Une frontière borne la rédaction au manque de temps.",
                   [sup(i, c) for i, c in TIME.items()], confidence="high")


def test_valid_regularity_is_kept_with_positions_and_counts():
    result = validate(TIME_DRAFT)
    c = claim(result)
    assert c["validation_status"] == "valid" and c["needs_review"] is False and codes(result) == []
    assert c["interview_ids"] == ["ENT_A", "ENT_B", "ENT_C", "ENT_E"] and c["n_supporting_interviews"] == 4
    assert (c["corpus_n_total"], c["corpus_n_usable"]) == (8, 8)
    assert c["interview_positions"]["ENT_D"] == "not_observed" and c["n_not_observed_interviews"] == 4
    assert result["document"]["recurring_boundaries"] == ["CC001"] and c["confidence"] == "high"
    assert c["support"][0]["evidence_turn_ids"] == ["ENT_A_T0002", "ENT_A_T0004"]


@pytest.mark.parametrize("entry, code", [
    (sup("ENT_Z", "ENT_Z_TC001", turns=[]), "UNKNOWN_INTERVIEW"),
    ({"interview_id": "ENT_A", "claim_ids": ["ENT_A_TC099"], "criterion_ids": [], "evidence_turn_ids": []},
     "UNKNOWN_OBJECT_ID"),
    ({"interview_id": "ENT_A", "claim_ids": [oid("ENT_B", "TIME")], "criterion_ids": [], "evidence_turn_ids": []},
     "FOREIGN_OBJECT_ID"),
    ({"interview_id": "ENT_A", "claim_ids": [], "criterion_ids": [], "evidence_turn_ids": []}, "EMPTY_SUPPORT_ENTRY"),
    (sup("ENT_A", oid("ENT_A", "TIME"), turns=["ENT_A_T0999"]), "UNKNOWN_EVIDENCE_TURN"),
])
def test_invented_support_rejects_the_claim(entry, code):
    result = validate(draft("recurring_boundary", "Frontière.", [sup("ENT_B", TIME["ENT_B"]),
                                                                 sup("ENT_C", TIME["ENT_C"]), entry]))
    assert code in codes(result) and claim(result)["validation_status"] == "rejected"
    assert result["document"]["recurring_boundaries"] == []


def test_excluded_interview_and_rejected_stage5_object_never_support():
    m, excluded = build(S6.IDS_4, extra=["ENT_X_INVALIDE"])
    result = validate(draft("recurring_boundary", "Frontière.", [sup("ENT_A", TIME["ENT_A"], m=m),
                                                                 sup("ENT_B", TIME["ENT_B"], m=m),
                                                                 sup("ENT_X_INVALIDE", "ENT_X_INVALIDE_TC001",
                                                                     turns=[], m=m)]), m=m, excluded=excluded)
    assert "EXCLUDED_INTERVIEW" in codes(result) and claim(result)["validation_status"] == "rejected"
    rejected = cm.build_material({"ENT_A": S6.document("ENT_A"), "ENT_X_INVALIDE": S6.document("ENT_X_INVALIDE")}, 2,
                                 "exploratory")
    entry = {"interview_id": "ENT_X_INVALIDE", "claim_ids": ["ENT_X_INVALIDE_TC002"], "criterion_ids": [],
             "evidence_turn_ids": []}
    assert "STAGE5_REJECTED_OBJECT" in codes(validate(draft("minority_configuration", "Cas.", [entry]), m=rejected))


def test_counts_are_on_distinct_interviews_never_on_entries_or_objects():
    a2 = oid("ENT_A", "TIME", n=1)
    entries = [sup("ENT_A", TIME["ENT_A"]), sup("ENT_A", a2), sup("ENT_B", TIME["ENT_B"])]
    result = validate(draft("recurring_boundary", "Frontière.", entries, n=3))
    c = claim(result)
    assert c["n_supporting_interviews"] == 2 and c["model_n_supporting_interviews"] == 3
    assert {"DUPLICATE_SUPPORT_INTERVIEW", "COUNT_MISMATCH"} <= set(codes(result))
    assert c["support"][0]["claim_ids"] == [TIME["ENT_A"], a2]


def test_single_interview_regularity_is_requalified_as_minority():
    result = validate(draft("recurring_boundary", "Frontière.", [sup("ENT_A", TIME["ENT_A"])], confidence="high"))
    c = claim(result)
    assert c["claim_type"] == "minority_configuration" and c["model_claim_type"] == "recurring_boundary"
    assert c["confidence"] == "low" and c["model_confidence"] == "high"
    assert "RECURRENCE_SINGLE_INTERVIEW" in codes(result)
    assert result["document"]["minority_configurations"] == ["CC001"]


def test_counterexample_without_stage5_element_is_not_a_refusal():
    result = validate(draft("recurring_student_role_criterion", "Critère de l'effort.",
                            [sup("ENT_A", oid("ENT_A", "EFFORT", "criteria")),
                             sup("ENT_C", oid("ENT_C", "EFFORT", "criteria"))],
                            [counter("ENT_B", "explicit_refusal", "ENT_B ne parle pas d'effort.")],
                            criterion_label="effort"))
    c = claim(result)
    assert c["interview_positions"]["ENT_B"] == "not_observed"
    assert c["counterexamples"][0]["validation_status"] == "unsupported"
    assert "COUNTEREXAMPLE_WITHOUT_SUPPORT" in codes(result)
    negative = result["document"]["negative_cases"]
    assert [(n["interview_id"], n["validation_status"], n["needs_review"]) for n in negative] == [
        ("ENT_B", "unsupported", True)]  # conservé pour une vérification humaine, jamais compté comme refus


def test_invented_counterexample_is_an_error_and_never_a_negative_case():
    result = validate(draft("recurring_boundary", "Frontière.", [sup("ENT_A", TIME["ENT_A"]),
                                                                 sup("ENT_B", TIME["ENT_B"])],
                            [{"interview_id": "ENT_D", "relation": "contrary_case", "description": "Cas.",
                              "claim_ids": ["ENT_D_TC042"], "criterion_ids": [], "evidence_turn_ids": []}]))
    assert "COUNTEREXAMPLE_UNKNOWN" in codes(result) and claim(result)["validation_status"] == "rejected"
    assert result["document"]["negative_cases"] == []


def test_negative_cases_are_kept_even_when_their_claim_is_rejected():
    d = oid("ENT_D", "ALWAYS")
    result = validate(draft("recurring_boundary", "Les étudiants de type pressé délèguent.",
                            [sup("ENT_A", TIME["ENT_A"]), sup("ENT_B", TIME["ENT_B"])],
                            [counter("ENT_D", "contrary_case", "Rédaction « même quand j'ai le temps ».", d)]))
    assert claim(result)["validation_status"] == "rejected" and "TYPOLOGY_OF_PERSONS" in codes(result)
    negative = result["document"]["negative_cases"]
    assert len(negative) == 1 and negative[0]["interview_id"] == "ENT_D" and negative[0]["needs_review"] is True
    assert negative[0]["related_cross_claim_ids"] == ["CC001"] and negative[0]["support_ids"] == [d]


def test_negative_case_claim_and_counterexample_are_merged_not_duplicated():
    d = oid("ENT_D", "ALWAYS")
    result = validate(draft("recurring_boundary", "Frontière.", [sup("ENT_A", TIME["ENT_A"]),
                                                                 sup("ENT_B", TIME["ENT_B"])],
                            [counter("ENT_D", "contrary_case", "Contraire.", d)]),
                      draft("negative_case", "ENT_D complique la frontière.", [sup("ENT_D", d)], related=[1]))
    negative = result["document"]["negative_cases"]
    assert len(negative) == 1 and negative[0]["relations"] == ["contrary_case", "negative_case"]
    assert negative[0]["sources"] == ["CC001.counterexamples", "CC002"]
    unrelated = validate(draft("negative_case", "ENT_D.", [sup("ENT_D", d)]))
    assert "NEGATIVE_CASE_UNRELATED" in codes(unrelated) and unrelated["document"]["negative_cases"]


def test_review_only_support_forces_review_and_caps_confidence():
    m, _ = build(S6.IDS_17)
    o, p = oid("ENT_O", "VERIF"), oid("ENT_P", "VERIF")
    result = validate(draft("recurring_accounting_move", "Rapport à la vérification.",
                            [sup("ENT_O", o, m=m), sup("ENT_P", p, m=m)], confidence="high"), m=m)
    c = claim(result)
    assert c["needs_review"] is True and c["confidence"] == "low" and c["model_confidence"] == "high"
    assert c["n_review_only_interviews"] == 2 and c["review_only_interview_ids"] == ["ENT_O", "ENT_P"]
    assert {"SUPPORTED_ONLY_BY_REVIEW_ITEMS", "STRONG_PATTERN_ON_REVIEW_ITEMS"} <= set(c["review_reasons"])
    assert c["stage5_review_reasons"] == ["SINGLE_SUPPORT_PATTERN"]
    assert all(s["needs_review"] and s["stage5_review_reasons"] == ["SINGLE_SUPPORT_PATTERN"] for s in c["support"])


def test_regularity_with_one_clean_interview_cannot_be_strong():
    m, _ = build(S6.IDS_17)
    o = oid("ENT_O", "VERIF")
    a = [c for c in m["interviews"]["ENT_A"]["claim_ids"]
         if m["claims"][c]["claim_type"] == "recurring_accounting_move"]
    result = validate(draft("recurring_accounting_move", "Rapport à la vérification.",
                            [sup("ENT_O", o, m=m), sup("ENT_A", a[0], m=m)], confidence="high"), m=m)
    c = claim(result)
    assert "RECURRENCE_RESTS_ON_REVIEW_ITEMS" in c["review_reasons"] and c["confidence"] == "low"


def test_temporal_pattern_requires_validated_temporal_changes():
    m, _ = build(S6.IDS_17)
    temporal = [sup(i, oid(i, "TEMPORAL"), m=m) for i in ("ENT_M", "ENT_N")]
    assert codes(validate(draft("temporal_pattern", "Changement daté.", temporal), m=m)) == []
    fake = validate(draft("temporal_pattern", "Changement.", [sup("ENT_E", oid("ENT_E", "MATHS"), m=m),
                                                              sup("ENT_G", oid("ENT_G", "MATHS"), m=m)]), m=m)
    assert claim(fake)["claim_type"] == "contextual_association" and "TEMPORAL_PATTERN_REQUALIFIED" in codes(fake)
    mixed = validate(draft("temporal_pattern", "Changement.", [*temporal, sup("ENT_E", oid("ENT_E", "MATHS"), m=m)]),
                     m=m)
    assert "TEMPORAL_SUPPORT_NOT_VALIDATED" in codes(mixed)


def test_criterion_pattern_must_cite_criteria():
    result = validate(draft("recurring_student_role_criterion", "Effort.", [sup("ENT_A", oid("ENT_A", "EFFORT")),
                                                                            sup("ENT_C", oid("ENT_C", "EFFORT"))],
                            criterion_label="effort"))
    assert "CRITERION_PATTERN_WITHOUT_CRITERIA" in codes(result)


def test_exploratory_corpus_caps_confidence():
    m, _ = build(S6.IDS_2)
    result = validate(draft("recurring_boundary", "Présent dans les deux entretiens disponibles.",
                            [sup("ENT_A", TIME["ENT_A"], m=m), sup("ENT_B", TIME["ENT_B"], m=m)], confidence="high"),
                      m=m, confidence="high")
    assert claim(result)["confidence"] == "medium" and "EXPLORATORY_CONFIDENCE_CAPPED" in codes(result)
    assert result["document"]["confidence"] == "medium" and "EXPLORATORY_CORPUS" in codes(result, "corpus")


SUPPORT = [sup("ENT_A", TIME["ENT_A"]), sup("ENT_B", TIME["ENT_B"]), sup("ENT_C", TIME["ENT_C"])]


@pytest.mark.parametrize("description, code, severity", [
    ("Les étudiants consciencieux bornent la rédaction.", "TYPOLOGY_OF_PERSONS", "error"),
    ("On observe un type d'étudiant contextuel.", "TYPOLOGY_OF_PERSONS", "error"),
    ("Cette frontière tient à la personnalité des enquêtés.", "PERSON_OR_SOCIAL_ATTRIBUTION", "error"),
    ("Cette frontière varie selon le milieu social.", "PERSON_OR_SOCIAL_ATTRIBUTION", "error"),
    ("Cette frontière traduit une stratégie de légitimation.", "PSYCHOLOGICAL_VOCABULARY", "warning"),
    ("La frontière s'explique par la charge de travail.", "CAUSAL_CLAIM", "warning"),
    ("En général, les étudiants délèguent faute de temps.", "GENERALIZATION_BEYOND_N", "warning"),
    ("Cette frontière apparaît 24 fois.", "OCCURRENCE_COUNT", "warning"),
    ("Cette frontière apparaît dans 5 des 8 entretiens.", "COUNT_IN_TEXT_MISMATCH", "warning"),
    ("ENT_D ne possède pas ce critère.", "NOT_OBSERVED_AS_ABSENCE", "warning"),
    ("La frontière apparaît aussi dans ENT_F.", "INTERVIEW_MENTION_NOT_SUPPORTED", "warning"),
    ("Formulée « uniquement le dimanche ».", "QUOTE_NOT_IN_STAGE5", "error"),
])
def test_text_checks(description, code, severity):
    result = validate(draft("recurring_boundary", description, SUPPORT))
    issue = next(i for i in result["report"]["issues"] if i["code"] == code)
    assert issue["severity"] == severity
    assert claim(result)["validation_status"] == ("rejected" if severity == "error" else "needs_review")


def test_correct_descriptions_raise_nothing():
    for description in ("Cette frontière apparaît dans 3 des 8 entretiens (ENT_A, ENT_B, ENT_C) : « seulement quand "
                        "je manque de temps ».",
                        "Trois entretiens sur 8 formulent cette frontière ; l'intelligence artificielle y est un "
                        "outil de rédaction.",
                        "Dans les entretiens I01, I02 et I03, la rédaction est bornée."):
        result = validate(draft("recurring_boundary", description, SUPPORT))
        assert codes(result) == [], (description, codes(result))


def test_summary_checks():
    result = validate(TIME_DRAFT, summary="Les étudiants font généralement ceci, à cause de leur discipline ; "
                                          "un type d'étudiant pressé apparaît dans I09, « jamais le soir ».")
    found = set(codes(result, "corpus"))
    assert {"SUMMARY_GENERALIZATION", "SUMMARY_CAUSAL", "SUMMARY_VOCABULARY", "SUMMARY_QUOTE_NOT_IN_STAGE5",
            "SUMMARY_UNKNOWN_INTERVIEW"} <= found
    assert result["document"]["needs_review"] is True
    clean = validate(TIME_DRAFT, summary="Dans 4 des 8 entretiens (I01, I02, I03, I05), une frontière borne la "
                                         "rédaction au manque de temps.")
    assert codes(clean, "corpus") == [] and clean["document"]["needs_review"] is False
    assert "ENT_A, ENT_B, ENT_C, ENT_E" in clean["document"]["cross_case_summary_display"]


def test_configuration_distribution_is_counted_in_interviews():
    m, _ = build(S6.IDS_17)
    distribution = V.configuration_distribution(m)
    assert distribution["mixed"] == {"n_interviews": 2, "interview_ids": ["ENT_M", "ENT_N"],
                                     "share": "2 entretien(s) sur 17 exploitable(s)"}
    assert sum(v["n_interviews"] for v in distribution.values()) == 17


def test_non_observed_count_is_accepted_only_when_written_as_such():
    ok = validate(draft("recurring_boundary", "Frontière dans 3 des 8 entretiens ; 5 des 8 entretiens ne la "
                                              "mentionnent pas (non observée).", SUPPORT))
    assert "COUNT_IN_TEXT_MISMATCH" not in codes(ok)
    wrong = validate(draft("recurring_boundary", "Cette frontière apparaît dans 5 des 8 entretiens.", SUPPORT))
    assert "COUNT_IN_TEXT_MISMATCH" in codes(wrong)
