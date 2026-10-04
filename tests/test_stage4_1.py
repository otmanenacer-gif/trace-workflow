"""Étape 4.1 — stabilisation après le premier run réel : proximité intra-tour, fusions, représentation, vocabulaire.

Aucun appel réel : agents simulés. Les nombres décrivent les fixtures synthétiques, pas des quotas.
"""

import json
from pathlib import Path

import pytest

from core import accountability, config
from core import accountability_candidates as ac
from core.accountability_episode_validator import validate_episodes
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from core.interaction_chunking import estimate_tokens
from tests import synthetic_interviews as si
from tests import synthetic_stage4_otmane as O
from tests.fake_llm import ACCOUNTABILITY, FakeAgents, fake_settings, agent_error
from tests.synthetic_interviews import practice, signal

IID = "LONG"
Q, A = "enqueteur", "enquete"
FILLER = ("Après, ça dépend beaucoup des semaines et de la charge de travail, parce qu'en licence on a souvent plusieurs "
          "rendus en même temps, des lectures à préparer et des exposés à organiser avec des camarades, donc on "
          "s'organise comme on peut avec un agenda et des listes de choses à faire mises à jour chaque soir.")
LONG = (f"Pour les fiches, je lui demande des questions. {FILLER} Pour les lectures, je lui demande des résumés, enfin "
        f"plutôt des plans. {FILLER} Pour l'anglais, je lui demande du vocabulaire. {FILLER} Bon, dit comme ça, ça a "
        "l'air décousu ce que je raconte.")


def tid(n: int) -> str:
    return f"{IID}_T{n:04d}"


def transcript(texts):
    return {"interview_id": IID, "turns": [{"turn_id": tid(i), "speaker": s, "text": t}
                                           for i, (s, t) in enumerate(texts, start=1)]}


def P(n, turn, quote, **fields):
    return {"practice_id": f"{IID}_P{n:03d}", **practice(turn_start=tid(turn - 1), turn_end=tid(turn),
                                                        evidence=[{"turn_id": tid(turn), "quote": quote}], **fields)}


def S(n, turn, signal_type, quote, **fields):
    return {"signal_id": f"{IID}_S{n:03d}", **signal(turn_ids=[tid(turn)], signal_type=signal_type, surface_form=quote,
                                                    description="d", evidence=[{"turn_id": tid(turn), "quote": quote}],
                                                    **fields)}


T_LONG = transcript([(Q, "Et pour tes cours ?"), (A, LONG)])
P_LONG = [P(1, 2, "Pour les fiches, je lui demande des questions.", academic_task="fiches"),
          P(2, 2, "Pour les lectures, je lui demande des résumés", academic_task="lectures"),
          P(3, 2, "Pour l'anglais, je lui demande du vocabulaire.", academic_task="anglais")]


# --- 1. Proximité dans un même tour -------------------------------------------------------------

def test_signal_attaches_only_to_the_practice_close_to_it_within_a_long_turn():
    built = ac.build_candidates(T_LONG, P_LONG, [S(1, 2, "self_reformulation", "enfin plutôt des plans")])
    [cand] = built["candidates"]
    assert cand["practice_ids"] == [f"{IID}_P002"] and "signal_proximity" in cand["trigger_types"]
    assert set(built["unmarked_practice_ids"]) == {f"{IID}_P001", f"{IID}_P003"}


def test_signal_far_from_every_practice_of_its_turn_creates_no_candidate():
    built = ac.build_candidates(T_LONG, P_LONG, [S(1, 2, "metadiscursive_self_evaluation",
                                                   "Bon, dit comme ça, ça a l'air décousu ce que je raconte.")])
    assert built["candidates"] == []
    assert built["unattached_signals"] == [{"signal_id": f"{IID}_S001", "reason": "FAR_WITHIN_TURN"}]
    assert built["summary"]["far_within_turn_signal_count"] == 1


def test_intra_turn_threshold_is_inclusive_and_documented():
    def turn_with_gap(gap):
        text = "Je lui demande des résumés." + "x" * gap + "Enfin plutôt des plans."
        t = transcript([(Q, "Et ?"), (A, text)])
        p = P(1, 2, "Je lui demande des résumés.")
        return ac.build_candidates(t, [p], [S(1, 2, "self_reformulation", "Enfin plutôt des plans.")])["candidates"]
    assert ac.INTRA_TURN_MAX_GAP_CHARS == 200
    assert len(turn_with_gap(200)) == 1 and turn_with_gap(201) == []


def test_explicit_triggers_do_not_depend_on_intra_turn_proximity():
    texts = [(Q, "Et pour tes cours ?"),
             (A, f"Pour les fiches, je les fais moi-même. {FILLER} Pour les lectures, je lui demande des résumés.")]
    t = transcript(texts)
    practices = [P(1, 2, "Pour les fiches, je les fais moi-même.", use_status="non_use", non_use_reason="not_stated",
                   academic_task="fiches"),
                 P(2, 2, "Pour les lectures, je lui demande des résumés.", academic_task="lectures")]
    built = ac.build_candidates(t, practices, [])
    [boundary] = [c for c in built["candidates"] if "boundary_formulation" in c["trigger_types"]]
    assert f"{IID}_P001" in boundary["practice_ids"]  # frontière explicite : candidat, même sans signal proche


# --- 2. Relations entre candidats --------------------------------------------------------------

def cand(cid, practices=(), signals=(), turns=(), tasks=(), contradiction=()):
    return {"candidate_id": cid, "practice_ids": list(practices), "substantive_signal_ids": list(signals),
            "evidence_turn_ids": list(turns), "tasks": list(tasks), "contradiction_turn_ids": list(contradiction)}


def test_candidate_links_are_explicit_only():
    a = cand("C1", ["P1"], ["S1"], ["T1"], ["mails"])
    assert ac.candidate_links(a, cand("C2", ["P1"])) == ["shared_practice"]
    assert ac.candidate_links(a, cand("C2", signals=["S1"])) == ["shared_signal"]
    assert ac.candidate_links(a, cand("C2", turns=["T1"], tasks=["mails"])) == ["same_turns_same_task"]
    assert ac.candidate_links(cand("C2", contradiction=["T1", "T9"]), a) == ["cross_turn_contradiction"]
    # mêmes tours mais autre tâche ; tours voisins ; même outil : aucun lien
    assert ac.candidate_links(a, cand("C2", turns=["T1"], tasks=["projet"])) == []
    assert ac.candidate_links(a, cand("C2", ["P2"], ["S2"], ["T3"], ["projet"])) == []


def test_components_allow_chains_but_never_join_unrelated_candidates():
    c1, c2, c3 = cand("C1", ["P1"]), cand("C2", ["P1", "P2"]), cand("C3", ["P2"])
    c4 = cand("C4", ["P9"], turns=["T50"], tasks=["mails"])
    assert ac.candidate_components([c1, c2, c3, c4]) == {"C1": "K01", "C2": "K01", "C3": "K01", "C4": "K02"}


# --- 3. Validation des fusions ------------------------------------------------------------------

def otmane_run(tmp_path, mode="good", **settings):
    run = si.make_ingested_run(tmp_path, O.files())
    cache = AnalysisCache(tmp_path / "cache")
    run = analyze_run(run, settings=fake_settings(), client=FakeAgents(O.stage3_responders()), cache=cache)
    transport = FakeAgents({ACCOUNTABILITY: O.builder(mode)})
    run = accountability.analyze_run_stage4(run, settings=fake_settings(**settings), client=transport, cache=cache)
    out = Path(run["files"][0]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    doc = json.loads((out / config.ACCOUNTABILITY_EPISODES_FILENAME).read_text(encoding="utf-8"))
    return run, doc, transport, cache


def test_otmane_regression_good_builder(tmp_path):
    run, doc, transport, _ = otmane_run(tmp_path)
    prepared = accountability.prepare_stage4(run["files"][0]["ingestion"])
    assert len(transport.calls) == len(prepared.requests) == doc["chunk_count"] > 1  # blocs d'au plus 4 candidats
    summary = doc["candidates"]["summary"]
    assert summary["practice_count"] == O.PRACTICE_COUNT == 50
    assert 70 <= summary["signal_count"] <= 80
    assert 25 <= doc["candidate_count"] <= 30
    candidates = doc["candidates"]["candidates"]
    # pas de signal_proximity faux dans les longs tours : une seule pratique, celle du passage proche
    for n in O.LONG_TURNS:
        long_cands = [c for c in candidates if c["evidence_turn_ids"] == [O.tid(n)]]
        assert len(long_cands) == 1 and len(long_cands[0]["practice_ids"]) == 1, n
    assert summary["far_within_turn_signal_count"] == len(O.LONG_TURNS)
    # paires proches dans le temps mais de tâches différentes : composantes distinctes
    comp = {c["evidence_turn_ids"][0]: c["component_id"] for c in candidates}
    for first, (_, second, _) in O.PAIRS.items():
        assert comp[O.tid(first)] != comp[O.tid(second)]
    # deux candidats réellement reliés (pratique partagée) : même composante, fusion conservée et valide
    linked = [c for c in candidates if O.tid(120) in c["evidence_turn_ids"]]
    assert len(linked) == 2 and linked[0]["component_id"] == linked[1]["component_id"]
    [merged] = [e for e in doc["episodes"] if len(e["candidate_ids"]) == 2]
    assert merged["validation_status"] == "valid" and merged["usable_for_next_stages"] is True
    assert doc["validation_error_count"] == doc["validation_warning_count"] == 0
    assert doc["usable_episode_count"] == doc["episode_count"] == doc["candidate_count"] - 1
    # le mot de l'enquêteur (« coupable ») n'est jamais attribué à l'étudiant
    assert "coupable" not in json.dumps(doc["episodes"], ensure_ascii=False)


def test_otmane_regression_adversarial_builder_is_caught(tmp_path):
    _, doc, _, _ = otmane_run(tmp_path, "adversarial")
    by_reason = lambda code: [e for e in doc["episodes"] if code in e["review_reasons"]]  # noqa: E731
    disconnected = by_reason("DISCONNECTED_MERGE")
    assert len(disconnected) == len(O.PAIRS)
    assert all(e["validation_status"] == "rejected" and e["usable_for_next_stages"] is False and e["needs_review"]
               for e in disconnected)
    [interviewer] = by_reason("INTERVIEWER_TERM_ATTRIBUTED")
    assert "INTERPRETIVE_VOCABULARY" in interviewer["review_reasons"] and interviewer["needs_review"]
    # la fusion réellement justifiée (pratique partagée) reste valide
    assert [e for e in doc["episodes"] if len(e["candidate_ids"]) == 2 and e["validation_status"] == "valid"]
    assert doc["usable_episode_count"] == doc["episode_count"] - len(O.PAIRS)


def test_otmane_cost_normalized_payload(tmp_path, monkeypatch):
    # format de la représentation mesuré à découpage égal (un seul bloc) : le découpage de l'étape 4.2 est testé
    # dans tests/test_stage4_blocks.py
    monkeypatch.setattr(ac, "BLOCK_MAX_CANDIDATES", ac.SINGLE_CALL_MAX_CANDIDATES)
    run, doc, transport, cache = otmane_run(tmp_path)
    prepared = accountability.prepare_stage4(run["files"][0]["ingestion"])
    legacy = estimate_tokens(O.legacy_user_message(prepared.transcript, prepared.built, prepared.speaker_warnings))
    normalized = prepared.estimated_input_tokens
    reduction = 1 - normalized / legacy
    print(f"\nOTMANE synthétique : {doc['candidate_count']} candidats ; ancien format {legacy} tokens estimés, "
          f"format normalisé {normalized} tokens estimés, réduction {reduction:.0%} ; appels {len(transport.calls)}")
    assert reduction >= 0.30
    assert normalized < ac.SINGLE_CALL_MAX_INPUT_TOKENS and len(prepared.requests) == 1 and len(transport.calls) == 1
    # chaque citation, pratique, signal et tour n'est envoyé qu'une fois
    payload = O.sent_payload(transport.calls[0]["params"])
    quotes = [(e["turn_id"], e["quote"]) for e in payload["evidence_by_id"].values()]
    assert len(quotes) == len(set(quotes))
    sent = transport.calls[0]["params"]["messages"][0]["content"]
    for pid in payload["practices_by_id"]:
        assert sent.count(f'"{pid}":{{') == 1
    assert O.INTERVIEW_ID + "_T" not in sent.split("<candidates>")[1]  # identifiants abrégés, préfixe déclaré une fois
    # relance : 0 appel
    again = FakeAgents({ACCOUNTABILITY: O.builder()})
    accountability.analyze_run_stage4(run, settings=fake_settings(), client=again, cache=cache)
    assert again.calls == []


# --- 4. Découpage par composantes (seulement au-delà du seuil) ------------------------------------

def test_oversized_payload_is_split_by_whole_components(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "SINGLE_CALL_MAX_INPUT_TOKENS", 2500)
    run, doc, transport, cache = otmane_run(tmp_path)
    calls = transport.calls
    assert len(calls) > 1 and doc["chunk_count"] == len(calls)
    # une composante n'est jamais coupée : les deux candidats reliés partent dans le même appel
    per_call = [{c["component_id"] for c in O.sent_payload(call["params"])["candidates"]} for call in calls]
    assert all(not (a & b) for i, a in enumerate(per_call) for b in per_call[i + 1:])
    assert doc["candidate_count"] == sum(len(O.sent_payload(c["params"])["candidates"]) for c in calls)
    assert doc["status"] == "SUCCESS" and doc["validation_error_count"] == 0
    assert doc["usable_episode_count"] == doc["candidate_count"] - 1  # même résultat qu'en un appel
    again = FakeAgents({ACCOUNTABILITY: O.builder()})
    accountability.analyze_run_stage4(run, settings=fake_settings(), client=again, cache=cache)
    assert again.calls == []  # cache par bloc


def test_one_failed_chunk_gives_a_partial_result(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "SINGLE_CALL_MAX_INPUT_TOKENS", 2500)
    good = O.builder()
    state = {"n": 0}

    def flaky(params):
        state["n"] += 1
        return agent_error() if state["n"] == 1 else good(params)

    run = si.make_ingested_run(tmp_path, O.files())
    cache = AnalysisCache(tmp_path / "cache")
    run = analyze_run(run, settings=fake_settings(), client=FakeAgents(O.stage3_responders()), cache=cache)
    run = accountability.analyze_run_stage4(run, settings=fake_settings(),
                                            client=FakeAgents({ACCOUNTABILITY: flaky}), cache=cache)
    summary = run["files"][0]["accountability"]
    assert summary["status"] == "PARTIAL" and summary["analysis_complete"] is False
    assert summary["error"]["code"] == "PARTIAL_ANALYSIS" and summary["episode_count"] > 0


# --- 5. Vocabulaire ----------------------------------------------------------------------------

T_VOC = transcript([(Q, "Tu te sens coupable quand tu l'utilises ?"), (A, "Non. Je l'utilise pour vérifier, c'est tout."),
                    (Q, "Et pour le reste ?"), (A, "J'aurais peur que le prof pense que je n'ai rien fait.")])
P_VOC = [P(1, 2, "Je l'utilise pour vérifier", academic_task="calculs"),
         P(2, 4, "J'aurais peur que le prof pense que je n'ai rien fait.", use_status="non_use",
           non_use_reason="other", academic_task="dossiers")]
S_VOC = [S(1, 2, "restriction", "c'est tout"), S(2, 4, "explicit_emotion", "J'aurais peur", explicit_affect="peur")]


def voc_episode(candidate, summary, status="accountability_episode"):
    return {"candidate_ids": [candidate["candidate_id"]], "turn_start": candidate["turn_ids"][0],
            "turn_end": candidate["turn_ids"][-1], "practice_ids": candidate["practice_ids"],
            "signal_ids": candidate["signal_ids"], "episode_status": status,
            "accountability_problem": None if status == "ordinary_practice" else "Jusqu'où ?",
            "accounting_moves": [] if status == "ordinary_practice" else [
                {"type": "restriction", "description": "L'étudiant limite l'usage.",
                 "evidence_turn_ids": [candidate["evidence_turn_ids"][0]]}],
            "boundary_objects": [], "student_role_reference": None, "external_reference": None,
            "episode_summary": summary, "confidence": "medium", "needs_review": False,
            "evidence": [{"turn_id": candidate["evidence_turn_ids"][0],
                          "quote": T_VOC["turns"][int(candidate["evidence_turn_ids"][0][-4:]) - 1]["text"]}]}


def run_voc(summary, index=0, status="accountability_episode"):
    built = ac.build_candidates(T_VOC, P_VOC, S_VOC)
    candidate = built["candidates"][index]
    return validate_episodes([voc_episode(candidate, summary, status)], T_VOC, P_VOC, S_VOC,
                             built["candidates"])["episodes"][0]


def test_interviewer_term_is_never_attributed_to_the_student():
    bad = run_voc("L'étudiant dit ne pas se sentir coupable et limite l'usage à la vérification.")
    assert {"INTERPRETIVE_VOCABULARY", "INTERVIEWER_TERM_ATTRIBUTED"} <= set(bad["review_reasons"])
    good = run_voc("L'étudiant répond « Non » à la question de l'enquêteur et limite l'usage à la vérification.")
    assert good["review_reasons"] == [] and good["validation_status"] == "valid"


def test_affect_said_by_the_student_is_allowed_only_quoted_from_them():
    assert run_voc("L'étudiant dit qu'il aurait « peur » que le prof pense qu'il n'a rien fait.", index=1)[
        "review_reasons"] == []
    for wording in ("Le passage exprime une forme de culpabilité.", "On perçoit de la honte.",
                    "L'intention de l'étudiant est de bien faire.", "L'usage est raconté sans réserve portant sur sa légitimité."):
        assert "INTERPRETIVE_VOCABULARY" in run_voc(wording)["review_reasons"], wording


def test_recommended_ordinary_practice_wording_passes():
    episode = run_voc(O.ORDINARY_SUMMARY, status="ordinary_practice")
    assert episode["review_reasons"] == [] and episode["validation_status"] == "valid"


# --- 6. Identifiants abrégés --------------------------------------------------------------------

def test_short_ids_are_expanded_deterministically():
    output = {"episodes": [{"candidate_ids": ["C001"], "practice_ids": ["P002", "X_P003"], "signal_ids": ["S001"],
                            "turn_start": "T0003", "turn_end": "T0004",
                            "evidence": [{"turn_id": "T0004", "quote": "q"}],
                            "accounting_moves": [{"type": "restriction", "description": "d", "evidence_turn_ids": ["T0004"]}]}],
              "builder_notes": None}
    episode = ac.expand_ids(output, "X")["episodes"][0]
    assert episode["candidate_ids"] == ["X_C001"] and episode["practice_ids"] == ["X_P002", "X_P003"]
    assert episode["turn_start"] == "X_T0003" and episode["evidence"][0]["turn_id"] == "X_T0004"
    assert episode["accounting_moves"][0]["evidence_turn_ids"] == ["X_T0004"]
    assert output["episodes"][0]["turn_start"] == "T0003"  # la sortie du modèle (cache) n'est jamais modifiée


@pytest.mark.parametrize("name", ["candidates", "practices_by_id", "signals_by_id", "evidence_by_id", "turns_by_id"])
def test_payload_sections_are_compact_and_parseable(name):
    built = ac.build_candidates(T_LONG, P_LONG, [S(1, 2, "self_reformulation", "enfin plutôt des plans")])
    payload = ac.build_payload(T_LONG, built)
    serialized = ac.serialize_payload(payload)
    assert json.loads(serialized)[name] == payload[name] and ": " not in serialized.split('"turns_by_id"')[0]
    assert payload["id_prefix"] == f"{IID}_" and "description" not in json.dumps(payload["signals_by_id"])
