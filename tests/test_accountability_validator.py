"""Étape 4 — Episode Validator déterministe (aucun LLM)."""

import copy

from core import accountability_candidates as ac
from core.accountability_episode_validator import validate_episodes
from tests.test_accountability_candidates import IID, PRACTICES, S, T, tid

SIGNALS = [
    S(1, [4], "restriction", [(4, "Pour reformuler oui, mais je ne veux pas qu'il écrive à ma place.")]),
    S(2, [6, 12], "cross_turn_contradiction",
      [(6, "Normalement je fais mes plans moi-même."), (12, "Une fois je lui ai demandé un plan")]),
    S(3, [8], "preference_statement", [(8, "Je lui fais corriger mes mails")]),
]
BUILT = ac.build_candidates(T, PRACTICES, SIGNALS, {})
CANDIDATES = BUILT["candidates"]


def cand(practice_number: int) -> dict:
    return next(c for c in CANDIDATES if f"{IID}_P{practice_number:03d}" in c["practice_ids"])


def episode(candidate: dict, **fields) -> dict:
    base = {
        "candidate_ids": [candidate["candidate_id"]], "turn_start": candidate["turn_ids"][0],
        "turn_end": candidate["turn_ids"][-1], "practice_ids": candidate["practice_ids"],
        "signal_ids": candidate["signal_ids"], "episode_status": "accountability_episode",
        "accountability_problem": "Jusqu'où l'outil peut-il écrire ?",
        "accounting_moves": [{"type": "restriction", "description": "L'étudiant limite l'usage à la reformulation.",
                              "evidence_turn_ids": [tid(4)]}],
        "boundary_objects": ["écrire / reformuler"], "student_role_reference": None, "external_reference": None,
        "episode_summary": "L'étudiant accepte la reformulation et exclut l'écriture.", "confidence": "high",
        "needs_review": False,
        "evidence": [{"turn_id": tid(4), "quote": "Pour reformuler oui, mais je ne veux pas qu'il écrive à ma place."}],
    }
    base.update(fields)
    return base


def good_set() -> list[dict]:
    return [
        episode(cand(2)),
        episode(cand(4), accounting_moves=[
            {"type": "general_rule", "description": "L'étudiant énonce ce qu'il fait d'ordinaire.", "evidence_turn_ids": [tid(6)]},
            {"type": "exception", "description": "L'étudiant rapporte une seule fois.", "evidence_turn_ids": [tid(12)]}],
                boundary_objects=["faire / faire faire"], evidence=[
            {"turn_id": tid(6), "quote": "Normalement je fais mes plans moi-même."},
            {"turn_id": tid(12), "quote": "Une fois je lui ai demandé un plan parce que j'étais en retard."}]),
        episode(cand(5), episode_status="ordinary_practice", accountability_problem=None, accounting_moves=[],
                boundary_objects=[], episode_summary="L'étudiant raconte faire corriger ses mails.", confidence="medium",
                evidence=[{"turn_id": tid(8), "quote": "Je lui fais corriger mes mails avec ChatGPT."}]),
    ]


def run(episodes, warnings=None):
    return validate_episodes(episodes, T, PRACTICES, SIGNALS, CANDIDATES, warnings)


def codes(result, index=0):
    return set(result["episodes"][index]["review_reasons"])


def by_candidate(result, practice_number):
    cid = cand(practice_number)["candidate_id"]
    return next(e for e in result["episodes"] if cid in e["candidate_ids"])


def test_well_formed_episodes_pass_without_issue():
    result = run(good_set())
    assert result["report"]["error_count"] == result["report"]["warning_count"] == 0
    assert [e["validation_status"] for e in result["episodes"]] == ["valid"] * 3
    assert [e["episode_id"] for e in result["episodes"]] == [f"{IID}_E001", f"{IID}_E002", f"{IID}_E003"]
    assert result["report"]["unaddressed_candidate_ids"] == []
    contradiction = by_candidate(result, 4)
    assert {e["turn_id"] for e in contradiction["evidence"]} == {tid(6), tid(12)}  # deux citations éloignées gardées


def test_fabricated_quote_is_rejected():
    episodes = good_set()
    episodes[0]["evidence"] = [{"turn_id": tid(4), "quote": "Je ne veux surtout pas qu'il écrive."}]
    result = run(episodes)
    e = by_candidate(result, 2)
    assert e["validation_status"] == "rejected" and {"QUOTE_NOT_FOUND", "NO_VALID_EVIDENCE"} <= set(e["review_reasons"])
    assert result["report"]["invalid_evidence_count"] == 1


def test_unknown_practice_signal_candidate_and_turn_ids_are_rejected():
    episodes = good_set()
    episodes[0]["practice_ids"] = [*episodes[0]["practice_ids"], f"{IID}_P099"]
    episodes[1]["signal_ids"] = [f"{IID}_S099"]
    episodes[2]["candidate_ids"] = [f"{IID}_C099"]
    result = run(episodes)
    assert "UNKNOWN_PRACTICE_ID" in by_candidate(result, 2)["review_reasons"]
    assert "UNKNOWN_SIGNAL_ID" in by_candidate(result, 4)["review_reasons"]
    assert all(e["validation_status"] == "rejected" for e in result["episodes"])
    assert cand(5)["candidate_id"] in result["report"]["unaddressed_candidate_ids"]


def test_turn_range_and_move_turns_are_checked():
    episodes = good_set()
    episodes[0].update(turn_start=tid(4), turn_end=tid(3))
    episodes[1]["accounting_moves"][0]["evidence_turn_ids"] = [f"{IID}_T0999"]
    episodes[2]["turn_start"] = "AUTRE_T0001"
    result = run(episodes)
    assert "INVALID_TURN_RANGE" in by_candidate(result, 2)["review_reasons"]
    assert "UNKNOWN_MOVE_TURN" in by_candidate(result, 4)["review_reasons"]
    assert "FOREIGN_INTERVIEW_TURN" in by_candidate(result, 5)["review_reasons"]


def test_accountability_episode_on_interviewer_question_only_is_rejected():
    """La catégorie de l'enquêteur (« triche ») ne devient jamais une position de l'enquêté·e."""
    bad = episode(cand(2), accounting_moves=[{"type": "appeal_to_external_judgment",
                                              "description": "L'étudiant qualifie l'usage de triche.",
                                              "evidence_turn_ids": [tid(13)]}],
                  episode_summary="L'étudiant considère que c'est de la triche.",
                  evidence=[{"turn_id": tid(13), "quote": "Tu ne trouves pas que c'est de la triche ?"}],
                  turn_start=tid(3), turn_end=tid(13))
    result = run([bad])
    e = result["episodes"][0]
    assert e["validation_status"] == "rejected"
    assert {"NO_INTERVIEWEE_EVIDENCE", "MOVE_WITHOUT_INTERVIEWEE_TURN"} <= set(e["review_reasons"])


def test_accountability_episode_without_moves_or_substantial_material_is_rejected():
    no_moves = episode(cand(2), accounting_moves=[])
    plain = episode(cand(5), signal_ids=[], accounting_moves=[{"type": "other_explicit_move", "description": "d",
                                                              "evidence_turn_ids": [tid(8)]}],
                    evidence=[{"turn_id": tid(8), "quote": "Je lui fais corriger mes mails avec ChatGPT."}],
                    boundary_objects=[])
    result = run([no_moves, plain])
    assert "NO_ACCOUNTING_MOVES" in by_candidate(result, 2)["review_reasons"]
    assert "INSUFFICIENT_MATERIAL" in by_candidate(result, 5)["review_reasons"]
    assert result["report"]["rejected_episode_ids"] == [e["episode_id"] for e in result["episodes"]]


def test_ordinary_practice_cannot_carry_five_moves():
    episodes = good_set()
    episodes[2]["accounting_moves"] = [{"type": t, "description": "d", "evidence_turn_ids": [tid(8)]}
                                       for t in ("restriction", "distinction", "refusal", "preference", "comparison")]
    e = by_candidate(run(episodes), 5)
    assert "ORDINARY_PRACTICE_WITH_ACCOUNTING" in e["review_reasons"] and e["validation_status"] == "needs_review"


def test_uncertain_is_always_flagged_for_review():
    episodes = good_set()
    episodes[2].update(episode_status="uncertain", needs_review=False, confidence="medium")
    e = by_candidate(run(episodes), 5)
    assert e["needs_review"] is True and {"UNCERTAIN_NOT_FLAGGED", "UNCERTAIN_CONFIDENCE"} <= set(e["review_reasons"])


def test_speaker_warning_is_propagated_by_trace_even_if_the_model_ignores_it():
    warnings = {tid(4): {"suggested_speaker": "enqueteur", "confidence": "medium"}}
    e = by_candidate(run(good_set(), warnings), 2)
    assert e["speaker_warnings"] == [{"turn_id": tid(4), "suggested_speaker": "enqueteur", "confidence": "medium"}]
    assert e["needs_review"] is True and "SPEAKER_WARNING_PROPAGATED" in e["review_reasons"]


def test_psychologizing_vocabulary_is_flagged():
    for wording in ("L'étudiant met en œuvre une stratégie défensive.", "Une rationalisation de l'usage.",
                    "Il cherche à protéger son identité morale.", "Une tentative de se justifier.",
                    "Une honte implicite apparaît.", "Une volonté de légitimer l'usage.", "Il fait preuve de mauvaise foi.",
                    "Une forme de manipulation.", "Une peur implicite du jugement."):
        episodes = good_set()
        episodes[0]["episode_summary"] = wording
        e = by_candidate(run(episodes), 2)
        assert "INTERPRETIVE_VOCABULARY" in e["review_reasons"], wording
    episodes = good_set()
    episodes[1]["accounting_moves"][0]["description"] = "Stratégie de légitimation."
    assert "INTERPRETIVE_VOCABULARY" in by_candidate(run(episodes), 4)["review_reasons"]


def test_vocabulary_used_by_the_interviewee_is_not_flagged_but_interviewer_wording_is():
    texts = list(T["turns"])
    t = {"interview_id": IID, "turns": [*texts, {"turn_id": tid(17), "speaker": "enquete",
                                                  "text": "J'ai un peu honte de lui faire écrire mes plans."}]}
    ep = episode(cand(2), episode_summary="L'étudiant dit avoir « un peu honte ».",
                 evidence=[{"turn_id": tid(4), "quote": "Pour reformuler oui"},
                           {"turn_id": tid(17), "quote": "J'ai un peu honte"}], turn_end=tid(17))
    result = validate_episodes([ep], t, PRACTICES, SIGNALS, CANDIDATES)
    assert "INTERPRETIVE_VOCABULARY" not in result["episodes"][0]["review_reasons"]
    # mot présent seulement dans la question de l'enquêteur : signalé
    ep = episode(cand(2), episode_summary="Le passage relève de la manipulation.",
                 evidence=[{"turn_id": tid(4), "quote": "Pour reformuler oui"}])
    assert "INTERPRETIVE_VOCABULARY" in run([ep])["episodes"][0]["review_reasons"]


def test_justification_requires_an_explicit_reason():
    episodes = good_set()
    episodes[0]["episode_summary"] = "Le passage produit une justification explicite."
    assert "UNSUPPORTED_JUSTIFICATION" in by_candidate(run(episodes), 2)["review_reasons"]
    episodes = good_set()
    episodes[1]["episode_summary"] = "Le passage produit une justification explicite : « parce que j'étais en retard »."
    assert "UNSUPPORTED_JUSTIFICATION" not in by_candidate(run(episodes), 4)["review_reasons"]


def test_merging_different_tasks_or_intervening_practices_is_flagged():
    merged = copy.deepcopy(good_set()[1])
    merged["candidate_ids"] = [cand(4)["candidate_id"], cand(5)["candidate_id"]]
    merged["practice_ids"] = [*merged["practice_ids"], f"{IID}_P005"]
    result = run([good_set()[0], merged])
    e = by_candidate(result, 4)
    assert {"MERGED_DIFFERENT_TASKS", "INTERVENING_PRACTICE_MERGED"} <= set(e["review_reasons"])


def test_candidate_not_addressed_or_addressed_twice_is_reported():
    episodes = good_set()
    result = run(episodes[:2])
    assert result["report"]["unaddressed_candidate_ids"] == [cand(5)["candidate_id"]]
    result = run([*episodes, copy.deepcopy(episodes[0])])
    assert any(i["code"] == "CANDIDATE_IN_SEVERAL_EPISODES" for i in result["report"]["issues"])
