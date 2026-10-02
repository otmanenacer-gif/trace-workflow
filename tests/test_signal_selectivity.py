"""Étape 3.7 — Interaction Signal Reader : règle de pertinence, micro-marqueurs, lecture à longue distance.

agents simulés uniquement : ces tests vérifient les consignes envoyées au modèle et la couche
DÉTERMINISTE (core/signal_selectivity.py), pas la qualité d'un vrai modèle.
"""

import json
from pathlib import Path

import pytest

from agents import interaction_signal_reader as reader
from core import analysis, interaction_chunking, signal_selectivity
from core.analysis import analyze_run
from core.analysis_cache import AnalysisCache
from tests import synthetic_interviews as si
from tests import synthetic_stage37 as S
from tests.fake_llm import (AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, FakeAgents, fake_settings,
                            text_response)

EMPTY_AUDIT = {"assessments": [], "audit_notes": None}
THEORY_TERMS = ("accountab", "garfinkel", "breach", "réparation", "métier d'étudiant", "régime", "coulon",
                "identité", "autonomie", "résistance", "morale")


# --- Consignes ------------------------------------------------------------------------------

def test_prompt_states_the_relevance_rule_and_micro_marker_policy():
    prompt = reader.SPEC.system_prompt
    assert "Tu ne fais **pas un inventaire linguistique**" in prompt
    assert "**Préférer un signal plus substantiel à plusieurs micro-signaux redondants.**" in prompt
    assert "Il n'y a ni quota ni plafond" in prompt and "pas de nombre maximal" not in prompt
    for marker in ("« euh »", "« ben »", "« voilà »", "« enfin »", "« juste »", "« un peu »", "« on va dire »"):
        assert marker in prompt, marker
    assert "Un simple « euh » en début de réponse : **aucun signal**." in prompt
    for negative in ("« Ben oui. » → aucun signal", "« Euh je l'utilise beaucoup. »",
                     "« J'utilise juste ChatGPT pour chercher des films. »"):
        assert negative in prompt, negative
    for positive in ("enfin, sauf une fois pour mon rapport. » → `self_correction` et `exception`",
                     "« Je préfère le faire moi-même. » → `preference_statement`",
                     "c'est un peu facile ce que je dis. » → `metadiscursive_self_evaluation`",
                     "`reference_to_teacher_judgment` et `explicit_emotion`",
                     "« Je ne veux pas qu'il réfléchisse à ma place. » → `normative_formulation`",
                     "« Je lui fais juste reformuler, jamais écrire. » → `restriction`"):
        assert positive in prompt, positive
    assert "Chaque tour cité dans `evidence` y figure obligatoirement." in prompt
    assert "présent mot pour mot dans une des citations du signal" in prompt
    assert "au moins une citation provient d'un tour `enquete`" in prompt
    assert "« justification », « stratégie », « défense », « légitimation », « normalisation »" in prompt
    lowered = prompt.casefold()
    for term in THEORY_TERMS:
        assert term not in lowered, term
    assert reader.SPEC.version == "1.3" and reader.SPEC.schema_version == "1.2"  # schéma (types) inchangé


def test_chunk_message_no_longer_asks_for_every_signal():
    template = reader.CHUNK_USER_TEMPLATE
    assert "Relève tous les signaux" not in template
    assert "relève les signaux pertinents pour le récit des pratiques, pas chaque marqueur" in template


def test_long_distance_prompt_is_stricter():
    prompt = analysis.LONG_DISTANCE.system_prompt
    assert "« je l'utilise beaucoup » ; plus loin : « je l'utilise surtout pour réviser » — c'est une précision" in prompt
    assert "réellement incompatibles ou en forte tension" in prompt and "En cas de doute, ne relève rien." in prompt
    assert reader.LONG_DISTANCE_VERSION == "1.1"


# --- Micro-marqueurs : règle déterministe ------------------------------------------------------

@pytest.mark.parametrize("surface, bare", [
    ("Euh…", True), ("ben", True), ("Ben voilà", True), ("enfin", True), ("juste", True), ("un peu", True),
    ("on va dire", True), ("énormément", True), ("…", True), ("euuh", True),
    ("enfin sauf", False), ("Enfin non", False), ("juste reformuler, jamais écrire", False), ("jamais", False),
    ("tout le temps", False), ("je préfère", False), ("(rires)", False),
])
def test_bare_filler_detection(surface, bare):
    assert signal_selectivity.is_bare_filler(surface) is bare


TRANSCRIPT = {"interview_id": "X", "turns": [
    {"turn_id": "X_T0001", "speaker": "enqueteur", "text": "Tu l'utilises pour écrire ?"},
    {"turn_id": "X_T0002", "speaker": "enquete", "text": "Euh je l'utilise beaucoup."},
    {"turn_id": "X_T0003", "speaker": "enqueteur", "text": "Et pour tes devoirs ?"},
    {"turn_id": "X_T0004", "speaker": "enquete", "text": "Euh… je l'utilise jamais pour écrire… enfin, sauf une fois."},
    {"turn_id": "X_T0005", "speaker": "enqueteur", "text": "Moi personnellement je lui fais juste reformuler."},
    {"turn_id": "X_T0006", "speaker": "enquete", "text": "Je lui fais juste reformuler, jamais écrire."},
]}


def sig(turn, signal_type, surface, quote, **fields):
    return si.signal(turn_ids=[f"X_T{turn:04d}"], signal_type=signal_type, surface_form=surface,
                     description="d", evidence=[{"turn_id": f"X_T{turn:04d}", "quote": quote}], **fields)


def reasons(result):
    return [(s["turn_ids"][0][-2:], s["signal_type"], s["set_aside_reason"]) for s in result["set_aside"]]


def test_isolated_bare_markers_are_set_aside_with_their_reason():
    signals = [sig(2, "hesitation", "Euh", "Euh"), sig(2, "intensification", "beaucoup", "beaucoup")]
    result = signal_selectivity.apply(signals, TRANSCRIPT)
    assert result["signals"] == []
    assert reasons(result) == [("02", "hesitation", "ISOLATED_MICRO_MARKER"),
                               ("02", "intensification", "ISOLATED_MICRO_MARKER")]
    assert all(e["validation"]["valid"] for s in result["set_aside"] for e in s["evidence"])  # citations vérifiées
    assert result["summary"]["set_aside_by_reason"] == {"ISOLATED_MICRO_MARKER": 2}


def test_hesitation_before_a_self_correction_is_kept_but_redundant_marker_is_not():
    signals = [sig(4, "hesitation", "Euh…", "Euh…"),
               sig(4, "self_correction", "enfin, sauf une fois", "enfin, sauf une fois."),
               sig(4, "exception", "sauf une fois", "sauf une fois"),
               sig(4, "self_correction", "enfin", "enfin")]                     # « enfin » seul : redondant
    result = signal_selectivity.apply(signals, TRANSCRIPT)
    assert [s["surface_form"] for s in result["signals"]] == ["Euh…", "enfin, sauf une fois", "sauf une fois"]
    assert reasons(result) == [("04", "self_correction", "REDUNDANT_MICRO_MARKER")]


def test_bounding_juste_is_kept_as_restriction_and_bare_juste_is_redundant():
    signals = [sig(6, "restriction", "juste reformuler, jamais écrire", "Je lui fais juste reformuler, jamais écrire."),
               sig(6, "minimization", "juste", "juste")]
    result = signal_selectivity.apply(signals, TRANSCRIPT)
    assert [s["signal_type"] for s in result["signals"]] == ["restriction"]
    assert reasons(result) == [("06", "minimization", "REDUNDANT_MICRO_MARKER")]


@pytest.mark.parametrize("fields", [{"needs_human_review": True}, {"explicit_affect": "honte"},
                                    {"cross_turn_reference": "x"}])
def test_markers_that_call_for_review_are_never_set_aside(fields):
    result = signal_selectivity.apply([sig(2, "hesitation", "Euh", "Euh", **fields)], TRANSCRIPT)
    assert len(result["signals"]) == 1 and result["set_aside"] == []


def test_marker_with_invalid_quote_is_kept_for_validation():
    result = signal_selectivity.apply([sig(2, "hesitation", "Euh", "Phrase inventée.")], TRANSCRIPT)
    assert len(result["signals"]) == 1


def test_interviewer_only_signal_is_set_aside_unless_the_turn_is_flagged_by_the_audit():
    signal = sig(5, "restriction", "juste reformuler", "Moi personnellement je lui fais juste reformuler.")
    result = signal_selectivity.apply([signal], TRANSCRIPT)
    assert reasons(result) == [("05", "restriction", "INTERVIEWER_ONLY_EVIDENCE")]
    flagged = signal_selectivity.apply([signal], TRANSCRIPT, {"X_T0005": {"suggested_speaker": "enquete",
                                                                         "confidence": "high"}})
    assert len(flagged["signals"]) == 1  # réponse probable de l'enquêté·e : appui possible, à revoir
    with_answer = dict(signal, turn_ids=["X_T0005", "X_T0006"],
                       evidence=[*signal["evidence"], {"turn_id": "X_T0006", "quote": "jamais écrire"}])
    assert len(signal_selectivity.apply([with_answer], TRANSCRIPT)["signals"]) == 1


def test_turn_ids_are_completed_with_every_cited_turn():
    signal = si.signal(turn_ids=["X_T0006"], signal_type="restriction", surface_form="jamais écrire", description="d",
                       evidence=[{"turn_id": "X_T0005", "quote": "Moi personnellement"},
                                 {"turn_id": "X_T0006", "quote": "jamais écrire"},
                                 {"turn_id": "Y_T0009", "quote": "x"}])
    result = signal_selectivity.apply([signal], TRANSCRIPT)
    kept, = result["signals"]
    assert kept["turn_ids"] == ["X_T0005", "X_T0006"] and kept["turn_ids_added"] == ["X_T0005"]
    assert result["summary"]["signals_with_turn_ids_completed"] == 1
    assert signal["turn_ids"] == ["X_T0006"]  # l'objet reçu n'est pas modifié ; un tour inconnu n'est pas ajouté


# --- Entretien long saturé de remplisseurs ----------------------------------------------------

@pytest.fixture
def long_run(tmp_path):
    return si.make_ingested_run(tmp_path, S.files())


def run_reader(tmp_path, run, interaction):
    transport = FakeAgents({PRACTICE: S.practice_reader, INTERACTION: interaction,
                               LONG_DISTANCE: S.long_distance_reader, AUDITOR: lambda p: text_response(EMPTY_AUDIT)})
    run = analyze_run(run, settings=fake_settings(), client=transport, cache=AnalysisCache(tmp_path / "cache"))
    directory = Path(run["files"][0]["analysis"]["analysis_dir"])
    return run, transport, json.loads((directory / "interaction_signals.json").read_text(encoding="utf-8"))


def kinds(document):
    return {(s["signal_type"], s["turn_ids"][0]) for s in document["signals"]}


EXPECTED_PHENOMENA = {
    ("self_correction", S.tid(10)), ("exception", S.tid(10)), ("preference_statement", S.tid(40)),
    ("metadiscursive_self_evaluation", S.tid(60)), ("explicit_emotion", S.tid(80)),
    ("reference_to_teacher_judgment", S.tid(80)), ("normative_formulation", S.tid(120)),
    ("restriction", S.tid(140)), ("contrast", S.tid(116)), ("contrast", S.tid(100)), ("cross_turn_contradiction", S.tid(24)),
    ("cross_turn_contradiction", S.tid(70)),
}


def test_fillers_do_not_become_dozens_of_signals_even_from_an_over_coding_reader(tmp_path, long_run):
    run, transport, doc = run_reader(tmp_path, long_run, S.naive_reader)
    assert len(transport.calls_for(INTERACTION)) == 3  # plusieurs blocs
    markers = S.micro_marker_count()
    info, selectivity = doc["chunking"], doc["selectivity"]
    assert markers > 600 and info["signals_before_dedup"] > markers  # le lecteur simulé a tout codé
    assert doc["item_count"] * 20 < markers  # nombre de signaux sans commune mesure avec les remplisseurs
    assert EXPECTED_PHENOMENA <= kinds(doc)  # les phénomènes significatifs restent
    assert not [s for s in doc["signals"] if signal_selectivity.is_bare_micro_marker(s)]
    # rien n'est perdu en silence : tout signal écarté est conservé, avec sa raison
    assert selectivity["signals_received"] == info["signals_after_dedup"] == doc["item_count"] + len(doc["set_aside_signals"])
    assert set(selectivity["set_aside_by_reason"]) == {"ISOLATED_MICRO_MARKER", "REDUNDANT_MICRO_MARKER",
                                                       "INTERVIEWER_ONLY_EVIDENCE"}
    assert selectivity["set_aside_by_reason"]["ISOLATED_MICRO_MARKER"] > 600
    assert doc["status"] == "SUCCESS" and doc["validation_issue_count"] == 0


def test_selective_reader_keeps_the_significant_phenomena(tmp_path, long_run):
    _, _, doc = run_reader(tmp_path, long_run, S.selective_reader)
    assert kinds(doc) == EXPECTED_PHENOMENA | {("preference_statement", S.tid(201)), ("restriction", S.tid(300))}
    assert doc["item_count"] == S.EXPECTED_LOCAL_SIGNAL_COUNT + 2 and doc["set_aside_signals"] == []
    assert doc["chunking"]["duplicates_removed"] == 1  # T0116, dans le chevauchement des blocs 1 et 2
    emotion = next(s for s in doc["signals"] if s["signal_type"] == "explicit_emotion")
    assert emotion["explicit_affect"] == "peur" and emotion["review_reasons"] == []


def test_short_interview_selectivity_leaves_legitimate_signals_untouched(tmp_path):
    run = si.make_ingested_run(tmp_path)
    transport = FakeAgents({PRACTICE: lambda p: text_response(si.GOOD_PRACTICES),
                               INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)})
    run = analyze_run(run, settings=fake_settings(), client=transport, cache=AnalysisCache(tmp_path / "cache"))
    directory = Path(run["files"][0]["analysis"]["analysis_dir"])
    doc = json.loads((directory / "interaction_signals.json").read_text(encoding="utf-8"))
    # « un peu » (à côté de « scrupules »), « Euh… » (avant « Enfin non »), « juste » : chacun sur un tour qui porte
    # un autre signal sans les contenir : conservés
    assert doc["item_count"] == 10 and doc["set_aside_signals"] == [] and doc["selectivity"]["signals_set_aside"] == 0


# --- Lecture à longue distance plus stricte ------------------------------------------------------

def test_long_distance_requires_two_cited_turns_in_different_blocks(long_run):
    transcript = json.loads((Path(long_run["files"][0]["ingestion"]["output_dir"]) / "structured_transcript.json")
                            .read_text(encoding="utf-8"))
    chunks = interaction_chunking.plan_chunks(transcript, 5000)
    position = {t["turn_id"]: i for i, t in enumerate(transcript["turns"])}
    real, other = S.CONTRADICTIONS
    assert interaction_chunking.is_long_distance(real, chunks, position)
    one_quote = dict(real, evidence=real["evidence"][:1])  # deux tours annoncés, un seul cité
    assert not interaction_chunking.is_long_distance(one_quote, chunks, position)
    near = dict(real, turn_ids=[S.tid(10), S.tid(24)],
                evidence=[S.ev(10, "Je ne l'utilise jamais pour mes devoirs"), real["evidence"][0]])
    assert not interaction_chunking.is_long_distance(near, chunks, position)  # même bloc : déjà lu


def test_long_distance_pass_finds_both_distant_contradictions(tmp_path, long_run):
    _, transport, doc = run_reader(tmp_path, long_run, S.selective_reader)
    sent = {t["turn_id"] for t in json.loads(
        transport.calls_for(LONG_DISTANCE)[0]["params"]["messages"][0]["content"]
        .split("<transcript>\n")[1].split("\n</transcript>")[0])["turns"]}
    assert {S.tid(24), S.tid(300), S.tid(70), S.tid(260)} <= sent and len(sent) < S.TURN_COUNT / 4
    contradictions = [s for s in doc["signals"] if s["signal_type"] == "cross_turn_contradiction"]
    assert [c["turn_ids"] for c in contradictions] == [[S.tid(24), S.tid(300)], [S.tid(70), S.tid(260)]]
    assert all(c["provenance"]["pass"] == "long_distance" and not c["needs_review"] for c in contradictions)
