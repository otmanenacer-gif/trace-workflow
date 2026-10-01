"""Étape 3.6 — Interaction Signal Reader sur un entretien long : blocs, longue distance, fusion, cache.

LLM simulé uniquement (tests/fake_llm.py) : aucun appel réel.
"""

import json
from pathlib import Path

import pytest

from agents.base import render_user_message
from core import analysis, interaction_chunking as chunking
from core.analysis import analyze_run, plan_analysis, plan_interaction_chunking
from core.analysis_cache import AnalysisCache
from core.llm_client import LLMSettings
from tests import synthetic_interviews as si
from tests import synthetic_long_interview as L
from tests.fake_llm import (AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, FakeTransport, fake_settings,
                            text_response)

EMPTY_PRACTICES = {"practices": [], "extraction_notes": None}
EMPTY_AUDIT = {"assessments": [], "audit_notes": None}


def responders(interaction=L.simulated_chunk_reader, long_distance=L.simulated_long_distance_reader):
    return {PRACTICE: lambda p: text_response(EMPTY_PRACTICES), INTERACTION: interaction,
            LONG_DISTANCE: long_distance, AUDITOR: lambda p: text_response(EMPTY_AUDIT)}


@pytest.fixture
def long_run(tmp_path):
    return si.make_ingested_run(tmp_path, L.long_files())


def analyze(tmp_path, run, transport, **settings):
    return analyze_run(run, settings=fake_settings(**settings), transport=transport,
                       cache=AnalysisCache(tmp_path / "cache"))


def outputs(run):
    directory = Path(run["files"][0]["analysis"]["analysis_dir"])
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))  # noqa: E731
    return {"document": read("interaction_signals.json"), "manifest": read("interaction_manifest.json"),
            "validation": read("evidence_validation.json")}


def transcript_of(run):
    ingestion = run["files"][0]["ingestion"]
    return json.loads((Path(ingestion["output_dir"]) / "structured_transcript.json").read_text(encoding="utf-8"))


def of_type(document, signal_type):
    return [s for s in document["signals"] if s["signal_type"] == signal_type]


def practice_blocks(run) -> int:
    """Étape 3.7 : le Practice Extractor lit aussi un entretien long par blocs (un appel par bloc)."""
    from core import practice_chunking
    return len(practice_chunking.plan_chunks(transcript_of(run), fake_settings().practice_chunk_tokens))


# --- Découpage ----------------------------------------------------------------------------

def test_long_interview_is_split_on_turn_boundaries_with_overlap(long_run):
    transcript = transcript_of(long_run)
    chunks = chunking.plan_chunks(transcript, 5000)
    ids = [t["turn_id"] for t in transcript["turns"]]
    assert len(chunks) == 5
    # couverture complète, ordonnée, sans tour coupé : les parties propres des blocs se suivent exactement
    assert [i for c in chunks for i in ids[c.core_start:c.end]] == ids
    assert all(list(c.turn_ids) == ids[c.start:c.end] for c in chunks)
    assert chunks[0].overlap_turns == 0
    assert all(c.overlap_turns == chunking.OVERLAP_TURNS for c in chunks[1:])
    assert all(c.estimated_tokens <= 5000 * 1.4 for c in chunks)
    assert chunking.max_interaction_calls(chunks) == 6  # 5 blocs + 1 lecture à longue distance


def test_short_interview_is_never_split(tmp_path):
    transcript = transcript_of(si.make_ingested_run(tmp_path))
    assert len(chunking.plan_chunks(transcript, 5000)) == 1
    assert len(chunking.plan_chunks(transcript, 500)) == 1  # même avec la plus petite taille autorisée


def test_tiny_tail_is_merged_into_previous_chunk(long_run):
    full = transcript_of(long_run)
    transcript = {"interview_id": full["interview_id"], "turns": full["turns"][:100]}
    total = sum(chunking.turn_costs(transcript))
    chunks = chunking.plan_chunks(transcript, total // 2 - 50)  # 2 blocs pleins + un reste minuscule
    assert len(chunks) == 2 and chunks[-1].end == len(transcript["turns"])


def test_chunk_size_setting_is_read_from_environment():
    settings = LLMSettings.from_env({"TRACE_INTERACTION_CHUNK_TOKENS": "3000"})
    assert settings.interaction_chunk_tokens == 3000
    assert LLMSettings.from_env({}).interaction_chunk_tokens == 5000
    clamped = LLMSettings.from_env({"TRACE_INTERACTION_CHUNK_TOKENS": "10"})
    assert clamped.interaction_chunk_tokens == 500 and clamped.problems
    assert "interaction_chunk_tokens" not in json.dumps(settings.request_params())  # n'entre pas dans les clés


# --- A. Entretien court : 1 appel, comportement inchangé -----------------------------------

def test_short_interview_single_call_unchanged(tmp_path):
    run = si.make_ingested_run(tmp_path)
    transport = FakeTransport({PRACTICE: lambda p: text_response(si.GOOD_PRACTICES),
                               INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)})
    run = analyze(tmp_path, run, transport)
    calls = transport.calls_for(INTERACTION)
    assert len(calls) == 1 and transport.calls_for(LONG_DISTANCE) == []
    from core.analysis import prepare_interview
    prepared = prepare_interview(run["files"][0]["ingestion"])
    assert calls[0]["params"]["messages"][0]["content"] == render_user_message(prepared.agent_input,
                                                                               prepared.transcript_json)
    out = outputs(run)
    assert out["manifest"]["chunking"]["chunking_used"] is False and out["manifest"]["chunking"]["chunk_count"] == 1
    assert out["manifest"]["cache_key"] is not None and out["manifest"]["status"] == "SUCCESS"
    assert [s["signal_id"] for s in out["document"]["signals"]] == [
        f"{si.INTERVIEW_ID}_S{i:03d}" for i in range(1, len(si.GOOD_SIGNALS["signals"]) + 1)]
    assert all("provenance" not in s for s in out["document"]["signals"])
    summary = run["files"][0]["analysis"]["agents"]["interaction_signal_reader"]
    assert summary["chunking"]["chunk_count"] == 1 and summary["analysis_complete"] is True


# --- B. Entretien long : plusieurs blocs, tous analysés, fusion -----------------------------

def test_long_interview_all_chunks_analyzed_and_merged(tmp_path, long_run):
    transport = FakeTransport(responders())
    run = analyze(tmp_path, long_run, transport)
    chunk_calls = transport.calls_for(INTERACTION)
    assert len(chunk_calls) == 5 and len(transport.calls_for(LONG_DISTANCE)) == 1
    for call in chunk_calls:  # même agent : mêmes consignes, même schéma, limite de sortie inchangée
        assert call["params"]["system"][0]["text"] == analysis.INTERACTION.system_prompt
        assert call["params"]["output_config"]["format"]["schema"] == analysis.INTERACTION.output_schema
        assert call["params"]["max_tokens"] == fake_settings().max_tokens
        content = call["params"]["messages"][0]["content"]
        assert "EXTRAIT" in content and len(L.sent_turns(call["params"])) < L.TURN_COUNT
    out = outputs(run)
    doc, manifest = out["document"], out["manifest"]
    assert doc["status"] == "SUCCESS" and doc["analysis_complete"] is True and manifest["analysis_complete"] is True
    chunk_info = manifest["chunking"]
    assert chunk_info["chunking_used"] is True and chunk_info["chunk_count"] == 5
    assert chunk_info["chunks_succeeded"] == 5 and chunk_info["failed_chunks"] == []
    assert chunk_info["overlap_turns"] == chunking.OVERLAP_TURNS and len(chunk_info["chunk_ranges"]) == 5
    assert chunk_info["llm_calls_this_run"] == 6 == manifest["api_calls"]
    # étape 3.7 : les « Euh… » isolés du lecteur simulé (surcodage) sont écartés APRÈS la fusion, jamais perdus
    set_aside = doc["selectivity"]["signals_set_aside"]
    assert chunk_info["signals_after_dedup"] == doc["item_count"] + set_aside == len(doc["signals"]) + set_aside
    assert [s["signal_id"] for s in doc["signals"]] == [f"{L.LONG_INTERVIEW_ID}_S{i:03d}"
                                                         for i in range(1, len(doc["signals"]) + 1)]
    # ordre de l'entretien
    position = {t["turn_id"]: i for i, t in enumerate(transcript_of(run)["turns"])}
    firsts = [min(position[t] for t in s["turn_ids"]) for s in doc["signals"]]
    assert firsts == sorted(firsts)
    # une hésitation par tour concerné, ni perdue ni dupliquée (étape 3.7 : écartée comme micro-marqueur isolé)
    expected = [t["turn_id"] for t in transcript_of(run)["turns"] if t["text"].startswith("Euh…")]
    hesitations = [s for s in doc["set_aside_signals"] if s["signal_type"] == "hesitation"]
    assert sorted(t for s in hesitations for t in s["turn_ids"]) == sorted(expected) and not of_type(doc, "hesitation")
    assert {s["set_aside_reason"] for s in hesitations} == {"ISOLATED_MICRO_MARKER"}
    usage = run["last_analysis"]["usage"]
    blocks = practice_blocks(run)
    assert usage["api_calls"] == blocks + 6 and usage["output_tokens"] == 300 * blocks + 5 * 900 + 300
    assert run["pipeline"]["Analyse interactionnelle"] == "terminé (1/1 entretien(s))"


# --- C. Chevauchement : même signal dans deux blocs → un seul ------------------------------

def test_overlap_duplicate_is_merged_once(tmp_path, long_run):
    run = analyze(tmp_path, long_run, FakeTransport(responders()))
    doc = outputs(run)["document"]
    info = doc["chunking"]
    assert info["signals_before_dedup"] - info["signals_after_dedup"] == info["duplicates_removed"] >= 2
    overlap_turn = L.ltid(154)  # T0150–T0155 : fin du bloc 2 et début du bloc 3
    signals = [s for s in doc["signals"] + doc["set_aside_signals"] if s["turn_ids"] == [overlap_turn]]
    assert len(signals) == 1
    assert signals[0]["provenance"] == {"pass": "local", "chunks": [2, 3], "merged_duplicates": 1}
    assert len(signals[0]["evidence"]) == 1 and signals[0]["evidence"][0]["validation"]["valid"]


# --- D. Signaux différents dans deux blocs → tous conservés --------------------------------

def _sig(turn_ids, signal_type, surface, quotes, review=False):
    return {"turn_ids": turn_ids, "signal_type": signal_type, "surface_form": surface, "description": "d",
            "topic": None, "evidence": [{"turn_id": t, "quote": q} for t, q in quotes], "explicit_affect": None,
            "cross_turn_reference": None, "explicitness": "direct", "needs_human_review": review}


def test_merge_keeps_distinct_signals_and_merges_only_true_duplicates(long_run):
    transcript = transcript_of(long_run)
    chunks = chunking.plan_chunks(transcript, 5000)
    shared = L.ltid(154)  # dans les blocs 2 et 3
    text = transcript["turns"][153]["text"]
    a = _sig([shared], "hesitation", "Euh…", [(shared, "Euh…")])
    b_same = _sig([shared], "hesitation", "euh…", [(shared, "Euh… Alors")], review=True)  # même signal
    b_type = _sig([shared], "minimization", "Euh…", [(shared, "Euh…")])                 # autre type
    b_form = _sig([shared], "hesitation", "puis", [(shared, text[-30:])])                # autre passage
    elsewhere = _sig([L.ltid(14)], "hesitation", "Euh…", [(L.ltid(14), "Euh…")])        # bloc 1 seul
    merged = chunking.merge_signals([
        {"label": "bloc2", "chunk": 2, "signals": [a, elsewhere]},
        {"label": "bloc3", "chunk": 3, "signals": [b_same, b_type, b_form]},
    ], chunks, transcript)
    assert merged["before"] == 5 and merged["after"] == 4 and merged["duplicates_removed"] == 1
    kept = {(tuple(s["turn_ids"]), s["signal_type"], s["surface_form"]) for s in merged["signals"]}
    assert kept == {((shared,), "hesitation", "Euh…"), ((shared,), "minimization", "Euh…"),
                    ((shared,), "hesitation", "puis"), ((L.ltid(14),), "hesitation", "Euh…")}
    first = next(s for s in merged["signals"] if s["surface_form"] == "Euh…" and s["signal_type"] == "hesitation"
                 and s["turn_ids"] == [shared])
    assert first["needs_human_review"] is True  # OU logique
    assert [e["quote"] for e in first["evidence"]] == ["Euh…", "Euh… Alors"]  # toutes les citations conservées
    assert first["provenance"]["chunks"] == [2, 3]


def test_same_signal_in_one_chunk_or_outside_overlap_is_not_merged(long_run):
    transcript = transcript_of(long_run)
    chunks = chunking.plan_chunks(transcript, 5000)
    shared = L.ltid(154)
    a = _sig([shared], "hesitation", "Euh…", [(shared, "Euh…")])
    # deux relevés identiques d'un MÊME bloc : l'agent les a distingués, TRACE ne fusionne pas
    merged = chunking.merge_signals([{"label": "bloc2", "chunk": 2, "signals": [a, dict(a)]}], chunks, transcript)
    assert merged["after"] == 2
    # même relevé attribué à deux blocs qui ne contiennent pas tous deux le tour : pas un doublon de chevauchement
    far = L.ltid(14)
    b = _sig([far], "hesitation", "Euh…", [(far, "Euh…")])
    merged = chunking.merge_signals([{"label": "bloc1", "chunk": 1, "signals": [b]},
                                     {"label": "bloc3", "chunk": 3, "signals": [dict(b)]}], chunks, transcript)
    assert merged["after"] == 2


# --- E. Contradiction entre deux passages éloignés ------------------------------------------

def test_long_distance_contradiction_is_detected(tmp_path, long_run):
    transport = FakeTransport(responders())
    run = analyze(tmp_path, long_run, transport)
    doc = outputs(run)["document"]
    contradictions = of_type(doc, "cross_turn_contradiction")
    assert len(contradictions) == 1
    found = contradictions[0]
    assert found["turn_ids"] == [L.ltid(L.EARLY_TURN), L.ltid(L.LATE_TURN)]
    assert found["provenance"] == {"pass": "long_distance", "chunks": [], "merged_duplicates": 0}
    assert all(e["validation"]["valid"] for e in found["evidence"])
    assert "CONTRADICTION_SINGLE_TURN" not in found["review_reasons"]
    # lecture légère : une sélection compacte, jamais l'entretien entier
    call = transport.calls_for(LONG_DISTANCE)[0]["params"]
    sent = L.sent_turns(call)
    assert len(sent) < L.TURN_COUNT / 2
    assert {L.ltid(L.EARLY_TURN), L.ltid(L.LATE_TURN)} <= {t["turn_id"] for t in sent}
    assert all(t["blocks"] for t in sent)
    assert call["system"][0]["text"] == analysis.LONG_DISTANCE.system_prompt
    enum = call["output_config"]["format"]["schema"]["$defs"]["LongDistanceSignal"]["properties"]["signal_type"]["enum"]
    assert set(enum) == {"cross_turn_contradiction", "significant_repetition", "vocabulary_shift"}
    info = doc["chunking"]["long_distance"]
    assert info["status"] == "SUCCESS" and info["signals_kept"] == 1
    assert info["selected_turn_count"] == len(sent)


def test_long_distance_pass_keeps_only_cross_chunk_signals(tmp_path, long_run):
    local = dict(L.CONTRADICTION, turn_ids=[L.ltid(10), L.ltid(12)],
                 evidence=[{"turn_id": L.ltid(10), "quote": L.EARLY_QUOTE}, {"turn_id": L.ltid(12), "quote": "Euh…"}])
    transport = FakeTransport(responders(long_distance=lambda p: text_response(
        {"signals": [L.CONTRADICTION, local], "reading_notes": None})))
    run = analyze(tmp_path, long_run, transport)
    doc = outputs(run)["document"]
    assert len(of_type(doc, "cross_turn_contradiction")) == 1
    info = doc["chunking"]["long_distance"]
    assert info["signals_returned"] == 2 and info["signals_kept"] == 1
    assert info["signals_discarded_not_long_distance"] == 1


def test_selection_respects_budget_and_skips_single_block(long_run):
    transcript = transcript_of(long_run)
    chunks = chunking.plan_chunks(transcript, 5000)
    selection = chunking.select_long_distance_turns(transcript, chunks, [], budget_tokens=20)
    assert selection["estimated_tokens"] <= 20 and selection["dropped_count"] > 0
    full = chunking.select_long_distance_turns(transcript, chunks, [])
    assert full["estimated_tokens"] <= chunking.LONG_DISTANCE_MAX_TOKENS
    assert L.TURN_COUNT // 10 > len(full["positions"]) > 0  # compacte : quelques tours, pas l'entretien
    # aucun candidat hors d'un même bloc → aucun appel
    one_block = {"interview_id": transcript["interview_id"], "turns": transcript["turns"][:20]}
    assert not chunking.select_long_distance_turns(one_block, chunking.plan_chunks(one_block, 5000), [])["needs_llm"]


# --- F. Bloc tronqué : statut explicite, relance ciblée -------------------------------------

def truncating_reader(turn_id):
    def respond(params):
        if any(t["turn_id"] == turn_id for t in L.sent_turns(params)):
            return text_response('{"signals": [', stop_reason="max_tokens", output_tokens=32000)
        return L.simulated_chunk_reader(params)
    return respond


def test_truncated_chunk_makes_result_explicitly_partial_then_relaunch_redoes_only_it(tmp_path, long_run):
    transport = FakeTransport(responders(interaction=truncating_reader(L.ltid(200))))  # bloc 3 seulement
    run = analyze(tmp_path, long_run, transport)
    out = outputs(run)
    doc, manifest = out["document"], out["manifest"]
    assert doc["status"] == manifest["status"] == "PARTIAL"
    assert doc["analysis_complete"] is False and manifest["analysis_complete"] is False
    assert manifest["error"]["code"] == "PARTIAL_ANALYSIS" and "INCOMPLÈTE" in manifest["error"]["message"]
    assert "4/5" in manifest["error"]["message"] and "tronquée" in manifest["error"]["message"]
    info = manifest["chunking"]
    assert info["truncated_chunks"] == [3] and info["failed_chunks"] == [3] and info["chunks_succeeded"] == 4
    chunk3 = info["chunks"][2]
    assert chunk3["status"] == "TRUNCATED" and chunk3["stop_reason"] == "max_tokens"
    assert info["long_distance"]["status"] == "SKIPPED_INCOMPLETE" and transport.calls_for(LONG_DISTANCE) == []
    # les blocs réussis restent utilisables (signaux valides), le bloc 3 n'a rien produit
    assert doc["signals"] and all(e["validation"]["valid"] for s in doc["signals"] for e in s["evidence"])
    assert all(3 not in s["provenance"]["chunks"] or len(s["provenance"]["chunks"]) > 1 for s in doc["signals"])
    assert out["validation"]["agents"]["interaction_signal_reader"]["analysis_complete"] is False
    summary = run["files"][0]["analysis"]["agents"]["interaction_signal_reader"]
    assert summary["analysis_complete"] is False and summary["chunking"]["truncated_chunks"] == [3]
    assert run["last_analysis"]["usage"]["partial"] == 1
    assert "incomplet" in run["pipeline"]["Analyse interactionnelle"]

    # relance : seuls le bloc 3 et la lecture à longue distance sont appelés
    plan = plan_analysis(run, [L.LONG_INTERVIEW_ID], fake_settings(), AnalysisCache(tmp_path / "cache"))
    assert plan["calls"] == 2 and plan["exact"] is False  # bloc 3 + longue distance (au plus)
    again = FakeTransport(responders())
    run = analyze(tmp_path, run, again)
    assert len(again.calls_for(INTERACTION)) == 1 and len(again.calls_for(LONG_DISTANCE)) == 1
    assert again.calls_for(PRACTICE) == []
    out = outputs(run)
    assert out["manifest"]["status"] == "SUCCESS" and out["document"]["analysis_complete"] is True
    assert out["manifest"]["error"] is None
    assert [c["cache_hit"] for c in out["manifest"]["chunking"]["chunks"]] == [True, True, False, True, True]


def test_all_chunks_failing_is_a_failure_without_output(tmp_path, long_run):
    transport = FakeTransport(responders(interaction=lambda p: text_response("x", stop_reason="max_tokens")))
    run = analyze(tmp_path, long_run, transport)
    directory = Path(run["files"][0]["analysis"]["analysis_dir"])
    manifest = json.loads((directory / "interaction_manifest.json").read_text())
    assert manifest["status"] == "FAILED" and manifest["error"]["code"] == "TRUNCATED"
    assert not (directory / "interaction_signals.json").exists()
    assert run["files"][0]["analysis"]["agents"]["practice_extractor"]["status"] == "SUCCESS"


# --- G. Cache par bloc ------------------------------------------------------------------------

def test_identical_relaunch_makes_no_call(tmp_path, long_run):
    cache = AnalysisCache(tmp_path / "cache")
    before = plan_analysis(long_run, [L.LONG_INTERVIEW_ID], fake_settings(), cache)
    blocks = practice_blocks(long_run)
    assert before["calls"] == blocks + 6 and before["exact"] is False
    run = analyze(tmp_path, long_run, FakeTransport(responders()))
    first = outputs(run)["document"]
    assert plan_analysis(run, [L.LONG_INTERVIEW_ID], fake_settings(), cache) == {
        "interviews": 1, "calls": 0, "cached": blocks + 6, "audit_calls": 0, "candidate_turns": 0, "exact": True}
    second = FakeTransport(responders())
    run = analyze(tmp_path, run, second)
    assert second.calls == []
    out = outputs(run)
    assert out["manifest"]["status"] == "CACHED" and out["manifest"]["cache_hit"] is True
    assert out["manifest"]["api_calls"] == 0 and out["manifest"]["billed_this_run"] is False
    assert out["document"]["signals"] == first["signals"]
    usage = run["last_analysis"]["usage"]
    assert usage["api_calls"] == 0 and usage["input_tokens"] == 0


def test_only_the_changed_chunk_is_recomputed(tmp_path, long_run):
    analyze(tmp_path, long_run, FakeTransport(responders()))
    text = L.long_text()
    assert text.count("Au semestre 170 ") == 1
    edited = text.replace("Au semestre 170 ", "Au semestre 170 bis ")  # tour T0340, dans le seul bloc 5
    assert edited != text
    run2 = si.make_ingested_run(tmp_path / "edited", [(L.LONG_FILENAME, edited.encode("utf-8"))])
    transport = FakeTransport(responders())
    run2 = analyze(tmp_path, run2, transport)
    assert len(transport.calls_for(INTERACTION)) == 1
    assert L.ltid(340) in {t["turn_id"] for t in L.sent_turns(transport.calls_for(INTERACTION)[0]["params"])}
    assert len(transport.calls_for(LONG_DISTANCE)) <= 1
    assert [c["cache_hit"] for c in outputs(run2)["manifest"]["chunking"]["chunks"]] == [True] * 4 + [False]


def test_plan_interaction_chunking_announces_blocks(tmp_path, long_run):
    plans = plan_interaction_chunking(long_run, [L.LONG_INTERVIEW_ID], fake_settings())
    assert plans[0]["chunking_used"] is True and plans[0]["chunk_count"] == 5 and plans[0]["max_calls"] == 6
    short = si.make_ingested_run(tmp_path / "short")
    plans = plan_interaction_chunking(short, [si.INTERVIEW_ID], fake_settings())
    assert plans[0] == {**plans[0], "chunking_used": False, "chunk_count": 1, "max_calls": 1}


# --- H. Citations : toutes valides après fusion ---------------------------------------------

def test_all_citations_valid_after_merge(tmp_path, long_run):
    run = analyze(tmp_path, long_run, FakeTransport(responders()))
    out = outputs(run)
    signals = out["document"]["signals"]
    assert signals and all(e["validation"]["valid"] for s in signals for e in s["evidence"])
    report = out["validation"]["agents"]["interaction_signal_reader"]
    assert report["invalid_evidence_count"] == 0 and report["error_count"] == 0
    assert report["evidence_count"] == sum(len(s["evidence"]) for s in signals)
    assert report["analysis_complete"] is True
    assert out["validation"]["total_invalid_evidence"] == 0
    turns = {t["turn_id"] for t in transcript_of(run)["turns"]}
    assert all(t in turns for s in signals for t in s["turn_ids"])


def test_fabricated_quote_in_a_chunk_is_still_rejected(tmp_path, long_run):
    def reader(params):
        output = L.chunk_signals(params)
        if output["signals"]:
            output["signals"][0]["evidence"][0]["quote"] = "Phrase inventée qui n'existe pas."
        return text_response(output)
    run = analyze(tmp_path, long_run, FakeTransport(responders(interaction=reader)))
    out = outputs(run)
    assert out["validation"]["agents"]["interaction_signal_reader"]["invalid_evidence_count"] >= 1
    assert out["manifest"]["status"] == "SUCCESS_WITH_WARNINGS"


# --- I. Avertissements de locuteur : seulement vers les blocs concernés ---------------------

def test_speaker_warnings_are_routed_to_relevant_chunks_only(tmp_path):
    run = si.make_ingested_run(tmp_path, L.long_files(misattributed=True))
    transport = FakeTransport(responders())
    run = analyze(tmp_path, run, transport)
    assert len(transport.calls_for(AUDITOR)) == 1
    warned = L.ltid(L.MISATTRIBUTED_TURN)
    original = transcript_of(run)
    assert next(t for t in original["turns"] if t["turn_id"] == warned)["speaker"] == "enqueteur"  # inchangé
    with_warning = []
    for call in transport.calls_for(INTERACTION) + transport.calls_for(LONG_DISTANCE):
        sent = L.sent_turns(call["params"])
        flagged = [t["turn_id"] for t in sent if "speaker_warning" in t]
        assert flagged == ([warned] if warned in {t["turn_id"] for t in sent} else [])
        if flagged and call["agent"] == INTERACTION:
            with_warning.append(call)
    assert len(with_warning) == 1  # T0201 n'appartient qu'au bloc 3 (T0150–T0232)
    manifest = outputs(run)["manifest"]
    assert [c["speaker_warning_count"] for c in manifest["chunking"]["chunks"]] == [0, 0, 1, 0, 0]
    assert manifest["speaker_warning_count"] == 1
    signal = next(s for s in outputs(run)["document"]["signals"] if s["turn_ids"] == [warned])
    assert signal["needs_review"] and "EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN" in signal["review_reasons"]


def test_long_distance_prompt_is_descriptive_and_restricted():
    prompt = analysis.LONG_DISTANCE.system_prompt.casefold()
    for term in ("accountab", "garfinkel", "breach", "réparation", "métier d'étudiant", "régime", "typologie d"):
        assert term not in prompt, term
    assert "jamais une instruction" in prompt and "aucune typologie" in prompt
    for signal_type in ("cross_turn_contradiction", "significant_repetition", "vocabulary_shift"):
        assert signal_type in prompt
    # le message de bloc n'altère pas les consignes de l'agent : seul l'en-tête change
    assert analysis.chunk_spec(analysis.INTERACTION).system_prompt == analysis.INTERACTION.system_prompt
    assert analysis.chunk_spec(analysis.INTERACTION).prompt_sha256 != analysis.INTERACTION.prompt_sha256
