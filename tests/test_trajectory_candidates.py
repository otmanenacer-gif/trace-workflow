"""Étape 5 — préparation déterministe (core/trajectory_candidates.py). Aucun LLM."""

import json

import pytest

from core import trajectory_candidates as tc
from tests import synthetic_stage5 as S5

ID = "ENTRETIEN_UNITAIRE"


def tid(n: int) -> str:
    return f"{ID}_T{n:04d}"


# --- Ancrages temporels ------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Au lycée je lui faisais tout faire.", [("Au lycée", "period")]),
    ("Avant, je lui demandais tout ; maintenant je relis.", [("Avant", "past"), ("maintenant", "present")]),
    ("Avant je faisais autrement.", [("Avant", "past")]),
    ("L'année dernière je l'utilisais beaucoup.", [("L'année dernière", "past")]),
    ("Je ne l'utilise plus pour les maths.", [("ne l'utilise plus", "transition")]),
    ("Depuis septembre, je lui demande des quiz.", [("Depuis", "present")]),
    ("En première année j'étais perdue, en troisième année non.",
     [("En première année", "period"), ("en troisième année", "period")]),
    ("Petit à petit j'ai arrêté.", [("Petit à petit", "transition"), ("j'ai arrêté", "transition")]),
])
def test_explicit_temporal_expressions_are_found_with_their_exact_text(text, expected):
    found = tc.find_temporal(text)
    assert [(f["text"], f["kind"]) for f in found] == expected
    assert all(text[f["start"]:f["end"]] == f["text"] for f in found)  # copie exacte du tour


@pytest.mark.parametrize("text", [
    "Je le relis avant de rendre le devoir.",          # « avant de » : pas un ancrage biographique
    "Je lui demande avant les partiels.",
    "Il est plus rapide que le correcteur.",
    "Je ne l'utilise pas plus que ça.",                 # « ne … pas plus » : pas un changement
    "Une fois, faute de temps, je lui ai demandé un plan.",  # fréquence, pas temporalité biographique
    "En maths je lui demande la réponse.",
])
def test_no_temporal_anchor_without_an_explicit_expression(text):
    assert tc.find_temporal(text) == []


def test_periods_have_a_biographical_rank():
    assert tc.anchor_kind("en première") == ("period", 2)
    assert tc.anchor_kind("en première année") == ("period", 3)
    assert tc.anchor_kind("exercices") is None


@pytest.mark.parametrize("anchors, basis", [
    ([("period", 2), ("present", None)], "past_present"),
    ([("past", None), ("present", None)], "past_present"),
    ([("transition", None)], "transition"),
    ([("period", 2), ("period", 5)], "ordered_periods"),
    ([("period", 2)], None),                     # un seul état daté : rien n'ordonne l'autre
    ([("present", None), ("present", None)], None),
    ([("period", 2), ("period", 2)], None),
    ([], None),
])
def test_ordering_basis(anchors, basis):
    assert tc.ordering_basis(anchors) == basis


# --- Matériau (identifiants entiers) -----------------------------------------------------------------

def _transcript(turns):
    return {"interview_id": ID, "turns": [{"turn_id": tid(i), "speaker": s, "text": t}
                                          for i, (s, t) in enumerate(turns, start=1)]}


def _practice(n, quote, pid, **fields):
    return {"practice_id": f"{ID}_P{pid:03d}", "turn_start": tid(n), "turn_end": tid(n), "use_status": "use",
            "practice_domain": "academic", "evidence": [{"turn_id": tid(n), "quote": quote}], **fields}


def _episode(eid, n, quote, practice_ids, usable=True, status="accountability_episode", **fields):
    return {"episode_id": f"{ID}_E{eid:03d}", "turn_start": tid(n), "turn_end": tid(n), "practice_ids": practice_ids,
            "signal_ids": [], "episode_status": status, "accounting_moves": fields.pop("moves", []),
            "boundary_objects": [], "episode_summary": "Résumé.", "confidence": "medium",
            "needs_review": fields.pop("needs_review", False), "review_reasons": [], "speaker_warnings": [],
            "validation_status": "valid" if usable else "rejected", "usable_for_next_stages": usable,
            "evidence": [{"turn_id": tid(n), "quote": quote}], **fields}


LONG_TURN = ("Au lycée, on avait beaucoup de devoirs. " + "Après il y avait les trajets, les repas et le reste. " * 8
             + "Pour les fiches, je lui demande des résumés.")
TURNS = [
    ("enqueteur", "Et maintenant, comment tu fais ?"),
    ("enquete", "Maintenant je lui demande seulement de relire mes copies."),
    ("enqueteur", "Et avant ?"),
    ("enquete", LONG_TURN),
    ("enqueteur", "Je lui demande moi aussi, avant, des choses."),
    ("enquete", "Je ne fais jamais faire mes plans, normalement."),
]


def _material(**overrides):
    practices = [_practice(2, "je lui demande seulement de relire mes copies", 1, academic_task="copies"),
                 _practice(4, "Pour les fiches, je lui demande des résumés.", 2, academic_task="fiches"),
                 _practice(5, "Je lui demande moi aussi", 3, academic_task="divers"),
                 _practice(6, "Je ne fais jamais faire mes plans", 4, use_status="refusal", academic_task="plans")]
    doc = {"episodes": [
        _episode(1, 2, "Maintenant je lui demande seulement de relire mes copies.", [f"{ID}_P001"],
                 moves=[{"type": "restriction", "evidence_turn_ids": [tid(2)]}]),
        _episode(2, 6, "Je ne fais jamais faire mes plans", [f"{ID}_P004"], usable=False)],
        "unmarked_practices": [{"practice_id": f"{ID}_P002"}, {"practice_id": f"{ID}_P003"}]}
    doc.update(overrides)
    return tc.build_material(_transcript(TURNS), doc, practices, [])


def test_only_usable_episodes_and_voiced_unmarked_practices_are_items():
    material = _material()
    assert list(material["items"]) == [f"{ID}_E001", f"{ID}_P002"]
    assert material["excluded_episodes"] == [{"episode_id": f"{ID}_E002", "validation_status": "rejected",
                                              "review_reasons": []}]
    # P003 : seule citation dans un tour de l'enquêteur → n'appuie rien
    assert material["summary"] == {"usable_episode_count": 1, "excluded_episode_count": 1, "unmarked_practice_count": 1,
                                   "temporal_anchor_count": 1, "needs_review_episode_count": 0, "item_count": 2}


def test_anchors_come_from_interviewee_turns_near_the_item_quotes_only():
    material = _material()
    # « Maintenant » (T0002) près de la citation ; « Au lycée » ouvre un tour long dont la citation est loin
    # (plus de 200 caractères) ; « maintenant » et « avant » des questions de l'enquêteur ne comptent jamais
    assert [(a["turn_id"], a["text"], a["kind"]) for a in material["temporal_anchors"]] == [
        (tid(2), "Maintenant", "present")]
    assert material["temporal_anchors"][0]["sentence"] == "Maintenant je lui demande seulement de relire mes copies."
    assert material["items"][f"{ID}_E001"]["anchor_ids"] == ["A001"]


def test_rule_and_case_flags():
    material = _material()
    episode = material["items"][f"{ID}_E001"]
    assert episode["rule"] is True and episode["case"] is False  # restriction (« seulement »)
    assert material["items"][f"{ID}_P002"]["rule"] is False


# --- Représentation envoyée ----------------------------------------------------------------------------

def test_payload_is_normalized_compact_and_limited_to_this_interview():
    material = _material()
    payload = tc.build_payload(_transcript(TURNS), material)
    text = tc.serialize_payload(payload)
    assert payload["id_prefix"] == f"{ID}_" and payload["interview_id"] == ID
    assert [e["id"] for e in payload["episodes"]] == ["E001"] and [p["id"] for p in payload["unmarked_practices"]] == ["P002"]
    assert payload["episodes"][0]["moves"] == [["restriction", "T0002"]]
    assert "E002" not in text  # épisode rejeté : jamais envoyé
    assert "les trajets, les repas" not in text  # jamais le texte complet d'un tour hors citation
    assert f"{ID}_T" not in text.replace('"id_prefix":"' + ID + '_"', "")  # identifiants abrégés partout
    assert ", " not in text.split('"quotes_by_id"')[0].replace(", ", "")  # JSON compact (hors texte cité)
    json.loads(text)


def test_expand_ids_restores_the_prefix():
    output = {"claims": [{"episode_ids": ["E001"], "practice_ids": ["P002"], "evidence_turn_ids": ["T0002"],
                          "temporal_anchors": [{"text": "Maintenant", "turn_id": "T0002"}]}],
              "student_role_criteria": [{"episode_ids": ["E001"], "evidence_turn_ids": ["T0002", f"{ID}_T0006"]}]}
    expanded = tc.expand_ids(output, ID)
    claim = expanded["claims"][0]
    assert claim["episode_ids"] == [f"{ID}_E001"] and claim["practice_ids"] == [f"{ID}_P002"]
    assert claim["temporal_anchors"][0]["turn_id"] == tid(2)
    assert expanded["student_role_criteria"][0]["evidence_turn_ids"] == [tid(2), tid(6)]
    assert output["claims"][0]["episode_ids"] == ["E001"]  # l'original n'est pas modifié


def test_regularities_on_a_real_stage4_output(tmp_path):
    run, _ = S5.case_to_stage4(tmp_path, S5.STABLE)
    import core.trajectory as trajectory
    prepared = trajectory.prepare_stage5(run["files"][0]["ingestion"])
    reg = prepared.material["regularities"]
    assert list(reg["repeated_moves"]) == ["restriction"] and len(reg["repeated_moves"]["restriction"]) == 3
    assert reg["ordinary_item_ids"] == [f"{S5.STABLE.interview_id}_P004"]
    assert reg["signaled_tensions"] == [] and reg["polarity_contrasts"] == []
