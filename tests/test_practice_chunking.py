"""Étape 3.7 — Practice Extractor sur un entretien long : blocs, fusion, cache, troncature.

LLM simulé uniquement (tests/fake_llm.py) : aucun appel réel.
"""

import copy
import json
from pathlib import Path

import pytest

from agents import practice_extractor
from agents.base import render_user_message
from core import analysis, practice_chunking
from core.analysis import analyze_run, plan_analysis, plan_practice_chunking
from core.analysis_cache import AnalysisCache
from core.llm_client import LLMSettings
from tests import synthetic_interviews as si
from tests import synthetic_stage37 as S
from tests.fake_llm import (AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, FakeTransport, fake_settings,
                            text_response)
from tests.synthetic_long_interview import sent_turns

EMPTY_AUDIT = {"assessments": [], "audit_notes": None}


def responders(practice=S.practice_reader, interaction=S.selective_reader):
    return {PRACTICE: practice, INTERACTION: interaction, LONG_DISTANCE: S.long_distance_reader,
            AUDITOR: lambda p: text_response(EMPTY_AUDIT)}


@pytest.fixture
def long_run(tmp_path):
    return si.make_ingested_run(tmp_path, S.files())


def analyze(tmp_path, run, transport, **settings):
    return analyze_run(run, settings=fake_settings(**settings), transport=transport,
                       cache=AnalysisCache(tmp_path / "cache"))


def outputs(run):
    directory = Path(run["files"][0]["analysis"]["analysis_dir"])
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))  # noqa: E731
    return {"document": read("practice_extractor.json"), "manifest": read("practice_manifest.json"),
            "validation": read("evidence_validation.json")}


def transcript_of(run):
    ingestion = run["files"][0]["ingestion"]
    return json.loads((Path(ingestion["output_dir"]) / "structured_transcript.json").read_text(encoding="utf-8"))


def find(document, turn, status):
    return [p for p in document["practices"] if any(e["turn_id"] == S.tid(turn) for e in p["evidence"])
            and p["use_status"] == status]


# --- Découpage ------------------------------------------------------------------------------

def test_practice_chunk_size_setting():
    assert LLMSettings.from_env({}).practice_chunk_tokens == 4000
    assert LLMSettings.from_env({"TRACE_PRACTICE_CHUNK_TOKENS": "3000"}).practice_chunk_tokens == 3000
    clamped = LLMSettings.from_env({"TRACE_PRACTICE_CHUNK_TOKENS": "50"})
    assert clamped.practice_chunk_tokens == 500 and clamped.problems
    assert "practice_chunk_tokens" not in json.dumps(fake_settings().request_params())  # hors clé de cache
    assert fake_settings().max_tokens == 32000  # la limite de sortie n'est pas augmentée


def test_long_interview_is_split_between_turns_with_light_overlap(long_run):
    transcript = transcript_of(long_run)
    chunks = practice_chunking.plan_chunks(transcript, 4000)
    ids = [t["turn_id"] for t in transcript["turns"]]
    assert len(chunks) == 4
    assert [i for c in chunks for i in ids[c.core_start:c.end]] == ids  # couverture exacte, aucun tour coupé
    assert all(list(c.turn_ids) == ids[c.start:c.end] for c in chunks)
    assert chunks[0].overlap_turns == 0
    assert all(c.overlap_turns == practice_chunking.PRACTICE_OVERLAP_TURNS == 8 for c in chunks[1:])
    assert all(c.estimated_tokens <= 4000 * 1.4 for c in chunks)


# --- A. Petit entretien : 1 appel, comportement inchangé -------------------------------------

def test_short_interview_single_practice_call_unchanged(tmp_path):
    run = si.make_ingested_run(tmp_path)
    transport = FakeTransport({PRACTICE: lambda p: text_response(si.GOOD_PRACTICES),
                               INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)})
    run = analyze(tmp_path, run, transport)
    calls = transport.calls_for(PRACTICE)
    assert len(calls) == 1
    prepared = analysis.prepare_interview(run["files"][0]["ingestion"])
    assert calls[0]["params"]["messages"][0]["content"] == render_user_message(prepared.agent_input,
                                                                               prepared.transcript_json)
    out = outputs(run)
    assert out["manifest"]["chunking"]["chunking_used"] is False and out["manifest"]["chunking"]["chunk_count"] == 1
    assert out["manifest"]["cache_key"] == analysis.compute_cache_key(
        analysis.cache_key_fields(analysis.PRACTICE, prepared, fake_settings()))  # même clé qu'avant l'étape 3.7
    assert out["document"]["item_count"] == 6 and all("provenance" not in p for p in out["document"]["practices"])


# --- B. Entretien long : plusieurs blocs -----------------------------------------------------

def test_long_interview_practice_chunks_all_analyzed(tmp_path, long_run):
    transport = FakeTransport(responders())
    run = analyze(tmp_path, long_run, transport)
    calls = transport.calls_for(PRACTICE)
    assert len(calls) == 4
    for call in calls:  # même agent : mêmes consignes, même schéma, même limite de sortie
        assert call["params"]["system"][0]["text"] == practice_extractor.SPEC.system_prompt
        assert call["params"]["output_config"]["format"]["schema"] == practice_extractor.SPEC.output_schema
        assert call["params"]["max_tokens"] == fake_settings().max_tokens
        content = call["params"]["messages"][0]["content"]
        assert "EXTRAIT" in content and "un usage et un non-usage restent deux pratiques" in content
        assert len(sent_turns(call["params"])) < S.TURN_COUNT
    out = outputs(run)
    doc, info = out["document"], out["manifest"]["chunking"]
    assert doc["status"] == out["manifest"]["status"] == "SUCCESS" and doc["analysis_complete"] is True
    assert info["chunking_used"] is True and info["chunk_count"] == 4 and info["chunks_succeeded"] == 4
    assert info["truncated_chunks"] == [] and info["overlap_turns"] == 8 and info["practice_calls_max"] == 4
    assert info["practices_after_dedup"] == doc["item_count"] == S.EXPECTED_PRACTICE_COUNT
    assert [p["practice_id"] for p in doc["practices"]] == [f"{S.INTERVIEW_ID}_P{i:03d}"
                                                             for i in range(1, doc["item_count"] + 1)]
    position = {t["turn_id"]: i for i, t in enumerate(transcript_of(run)["turns"])}
    starts = [position[p["turn_start"]] for p in doc["practices"]]
    assert starts == sorted(starts)  # ordre de l'entretien
    summary = run["files"][0]["analysis"]["agents"]["practice_extractor"]
    assert summary["chunking"]["chunk_count"] == 4
    assert summary["chunking"]["practices_before_dedup"] == S.EXPECTED_PRACTICE_COUNT + 1  # + 1 doublon (T0090)


# --- C. Doublon dans le chevauchement → une pratique -----------------------------------------

def test_overlap_duplicate_practice_is_merged_once(tmp_path, long_run):
    doc = outputs(analyze(tmp_path, long_run, FakeTransport(responders())))["document"]
    info = doc["chunking"]
    assert info["practices_before_dedup"] - info["practices_after_dedup"] == info["duplicates_removed"] == 1
    mails, = find(doc, 90, "use")  # T0089–T0090 : fin du bloc 1 et début du bloc 2
    assert mails["provenance"] == {"chunks": [1, 2], "merged_duplicates": 1}
    assert len(mails["evidence"]) == 1 and mails["evidence"][0]["validation"]["valid"]


def _practice(turn, quote, status="use", **fields):
    return si.practice(summary=fields.pop("summary", "Résumé."), turn_start=S.tid(turn - 1), turn_end=S.tid(turn),
                       use_status=status, evidence=[S.ev(turn, quote)], **fields)


def _merge(long_run, first, second):
    transcript = transcript_of(long_run)
    chunks = practice_chunking.plan_chunks(transcript, 4000)
    return practice_chunking.merge_practices([{"chunk": 1, "practices": first}, {"chunk": 2, "practices": second}],
                                             chunks, transcript)


def test_merge_never_joins_different_conducts(long_run):
    quote = "je lui demande de corriger l'orthographe"  # T0090, dans la zone commune aux blocs 1 et 2
    use = _practice(90, quote, academic_task="mails")
    for first, other in ((use, dict(use, use_status="non_use", non_use_reason="not_stated")),     # use / non_use
                         (use, dict(use, use_status="refusal", non_use_reason="personal_rule")),  # use / refusal
                         (use, dict(use, use_status="past_use")),                                 # past_use / use
                         (use, dict(use, practice_domain="personal")),                            # études / perso
                         (use, dict(use, academic_task="exposés")),                               # autre situation
                         (dict(use, ai_tool=["ChatGPT"]), dict(use, ai_tool=["Gemini"]))):        # autre outil
        merged = _merge(long_run, [first], [other])
        assert merged["after"] == 2 and merged["duplicates_removed"] == 0, other
    assert _merge(long_run, [use], [copy.deepcopy(use)])["after"] == 1  # témoin : la même conduite est fusionnée
    past = _practice(100, "Avant je lui faisais faire mes fiches de lecture", "past_use")
    now = _practice(100, "maintenant je les fais moi-même", "non_use", non_use_reason="not_stated")
    assert _merge(long_run, [past, now], [])["after"] == 2  # même tour : deux pratiques


def test_merge_requires_a_shared_anchor_in_the_overlap(long_run):
    far = _practice(20, "Il m'arrive de lui faire rédiger des paragraphes quand je bloque sur un devoir.")
    assert _merge(long_run, [far], [copy.deepcopy(far)])["after"] == 2  # T0020 hors de la zone commune
    same_chunk = _practice(90, "je lui demande de corriger l'orthographe")
    merged = practice_chunking.merge_practices(
        [{"chunk": 1, "practices": [same_chunk, copy.deepcopy(same_chunk)]}],
        practice_chunking.plan_chunks(transcript_of(long_run), 4000), transcript_of(long_run))
    assert merged["after"] == 2  # deux pratiques d'un MÊME bloc ne sont jamais fusionnées


def test_merge_keeps_all_evidence_turns_notes_and_most_cautious_reading(long_run):
    a = _practice(90, "je lui demande de corriger l'orthographe", academic_task="mails", explicitness="direct",
                  uncertainty_note="Note du bloc 1.", ai_action=["corrige"])
    b = si.practice(summary="Autre résumé.", turn_start=S.tid(88), turn_end=S.tid(90), use_status="use",
                    academic_task="mails", explicitness="unclear", uncertainty_note="Note du bloc 2.",
                    ai_action=["corrige", "relit"], assessment_context="ungraded",
                    evidence=[S.ev(90, "Pour les mails, je lui demande de corriger l'orthographe."),
                              S.ev(89, S.turn_text(89))])
    merged = _merge(long_run, [a], [b])
    assert merged["after"] == 1 and merged["duplicates_removed"] == 1
    p = merged["practices"][0]
    assert p["summary"] == "Résumé." and (p["turn_start"], p["turn_end"]) == (S.tid(88), S.tid(90))
    assert [e["turn_id"] for e in p["evidence"]] == [S.tid(90), S.tid(90), S.tid(89)]  # toutes les citations
    assert p["ai_action"] == ["corrige", "relit"] and p["assessment_context"] == "ungraded"
    assert p["explicitness"] == "unclear" and p["uncertainty_note"] == "Note du bloc 1. | Note du bloc 2."
    assert p["provenance"] == {"chunks": [1, 2], "merged_duplicates": 1}


def test_one_practice_absorbs_at_most_one_practice_per_chunk(long_run):
    sentence = "Pour les mails, je lui demande de corriger l'orthographe."
    a1 = _practice(90, sentence, summary="Correction de l'orthographe des mails.", academic_task="mails")
    a2 = _practice(90, sentence, summary="Relecture des mails.", academic_task="mails")
    b1 = _practice(90, sentence, summary="Correction de l'orthographe des mails.", academic_task="mails")
    b2 = _practice(90, sentence, summary="Relecture des mails.", academic_task="mails")
    merged = _merge(long_run, [a1, a2], [b2, b1])
    assert merged["after"] == 2
    assert sorted(p["summary"] for p in merged["practices"]) == ["Correction de l'orthographe des mails.",
                                                                 "Relecture des mails."]


# --- D. use + refusal / non_use dans des blocs différents → pratiques distinctes ----------------

def test_use_and_distant_refusal_stay_separate_practices(tmp_path, long_run):
    doc = outputs(analyze(tmp_path, long_run, FakeTransport(responders())))["document"]
    use, = find(doc, 20, "use")                 # « Il m'arrive de lui faire rédiger » (bloc 1)
    non_use, = find(doc, 150, "non_use")        # « Normalement mes travaux je les fais moi-même » (bloc 2)
    refusal, = find(doc, 24, "refusal")         # « Je ne fais jamais rédiger mes devoirs »
    assert use["provenance"]["chunks"] == [1] and non_use["provenance"]["chunks"] == [2]
    assert len({use["practice_id"], non_use["practice_id"], refusal["practice_id"]}) == 3
    # général / exception dans un même tour : deux pratiques ; usage borné / refus : deux pratiques
    assert find(doc, 10, "non_use") and find(doc, 10, "use")
    assert find(doc, 140, "use") and find(doc, 140, "refusal")
    # usage personnel et non-usage hors des cours, à distance : deux pratiques, aucune lecture de l'écart
    assert find(doc, 70, "non_use")[0]["practice_domain"] == find(doc, 260, "use")[0]["practice_domain"] == "personal"
    for practice in doc["practices"]:
        assert "contradiction" not in json.dumps(practice, ensure_ascii=False).casefold()


# --- E. past_use + non_use actuel → deux pratiques ---------------------------------------------

def test_past_use_and_current_non_use_stay_two_practices(tmp_path, long_run):
    doc = outputs(analyze(tmp_path, long_run, FakeTransport(responders())))["document"]
    past, = find(doc, 100, "past_use")
    now, = find(doc, 100, "non_use")
    assert past["practice_id"] != now["practice_id"] and now["non_use_reason"] == "not_stated"


# --- F. Bloc tronqué → PARTIAL, jamais présenté comme complet ------------------------------------

def truncating_practice_reader(turn):
    def respond(params):
        if any(t["turn_id"] == S.tid(turn) for t in sent_turns(params)):
            return text_response('{"practices": [', stop_reason="max_tokens", output_tokens=32000)
        return S.practice_reader(params)
    return respond


def test_truncated_practice_chunk_is_partial_then_relaunch_redoes_only_it(tmp_path, long_run):
    transport = FakeTransport(responders(practice=truncating_practice_reader(140)))  # bloc 2 seulement
    run = analyze(tmp_path, long_run, transport)
    out = outputs(run)
    doc, manifest = out["document"], out["manifest"]
    assert doc["status"] == manifest["status"] == "PARTIAL"
    assert doc["analysis_complete"] is False and manifest["analysis_complete"] is False
    assert manifest["error"]["code"] == "PARTIAL_ANALYSIS" and "3/4" in manifest["error"]["message"]
    assert "TRACE_PRACTICE_CHUNK_TOKENS" in manifest["error"]["message"]
    info = manifest["chunking"]
    assert info["truncated_chunks"] == [2] and info["chunks_succeeded"] == 3
    assert info["chunks"][1]["status"] == "TRUNCATED" and info["chunks"][1]["stop_reason"] == "max_tokens"
    assert doc["practices"] and not find(doc, 140, "use")  # les blocs réussis restent, le bloc 2 manque
    assert out["validation"]["agents"]["practice_extractor"]["analysis_complete"] is False
    assert run["last_analysis"]["usage"]["partial"] == 1 and "incomplet" in run["pipeline"]["Extraction des pratiques"]
    assert run["files"][0]["analysis"]["agents"]["interaction_signal_reader"]["status"] == "SUCCESS"

    plan = plan_analysis(run, [S.INTERVIEW_ID], fake_settings(), AnalysisCache(tmp_path / "cache"))
    assert plan["calls"] == 1 and plan["exact"] is True  # seul le bloc 2 manque
    again = FakeTransport(responders())
    run = analyze(tmp_path, run, again)
    assert len(again.calls_for(PRACTICE)) == 1 and again.calls_for(INTERACTION) == []
    out = outputs(run)
    assert out["manifest"]["status"] == "SUCCESS" and out["document"]["item_count"] == S.EXPECTED_PRACTICE_COUNT
    assert [c["cache_hit"] for c in out["manifest"]["chunking"]["chunks"]] == [True, False, True, True]


def test_all_practice_chunks_failing_is_a_failure_without_output(tmp_path, long_run):
    transport = FakeTransport(responders(practice=lambda p: text_response("x", stop_reason="max_tokens")))
    run = analyze(tmp_path, long_run, transport)
    directory = Path(run["files"][0]["analysis"]["analysis_dir"])
    manifest = json.loads((directory / "practice_manifest.json").read_text())
    assert manifest["status"] == "FAILED" and manifest["error"]["code"] == "TRUNCATED"
    assert not (directory / "practice_extractor.json").exists()


# --- G. Cache par bloc ---------------------------------------------------------------------------

def test_identical_relaunch_makes_no_practice_call(tmp_path, long_run):
    cache = AnalysisCache(tmp_path / "cache")
    run = analyze(tmp_path, long_run, FakeTransport(responders()))
    first = outputs(run)["document"]["practices"]
    plan = plan_analysis(run, [S.INTERVIEW_ID], fake_settings(), cache)
    assert plan["calls"] == 0 and plan["exact"] is True
    second = FakeTransport(responders())
    run = analyze(tmp_path, run, second)
    assert second.calls == []
    out = outputs(run)
    assert out["manifest"]["status"] == "CACHED" and out["manifest"]["api_calls"] == 0
    assert out["document"]["practices"] == first


def test_local_edit_recomputes_only_the_touched_practice_chunk(tmp_path, long_run):
    analyze(tmp_path, long_run, FakeTransport(responders()))
    edit = ("Le week-end je lui demande toujours des recettes de cuisine.",
            "Le week-end je lui demande toujours des recettes de cuisine végétarienne.")  # T0260, bloc 3 seul
    run2 = si.make_ingested_run(tmp_path / "edited", S.files(edit))
    transport = FakeTransport(responders())
    run2 = analyze(tmp_path, run2, transport)
    calls = transport.calls_for(PRACTICE)
    assert len(calls) == 1 and S.tid(260) in {t["turn_id"] for t in sent_turns(calls[0]["params"])}
    assert [c["cache_hit"] for c in outputs(run2)["manifest"]["chunking"]["chunks"]] == [True, True, False, True]


def test_plan_practice_chunking_announces_blocks(tmp_path, long_run):
    plan, = plan_practice_chunking(long_run, [S.INTERVIEW_ID], fake_settings())
    assert plan["chunking_used"] is True and plan["chunk_count"] == 4 and plan["max_calls"] == 4
    short, = plan_practice_chunking(si.make_ingested_run(tmp_path / "short"), [si.INTERVIEW_ID], fake_settings())
    assert (short["chunking_used"], short["chunk_count"], short["max_calls"]) == (False, 1, 1)


# --- H. Citations ------------------------------------------------------------------------------

def test_all_practice_citations_valid_after_merge(tmp_path, long_run):
    out = outputs(analyze(tmp_path, long_run, FakeTransport(responders())))
    practices = out["document"]["practices"]
    assert practices and all(e["validation"]["valid"] for p in practices for e in p["evidence"])
    report = out["validation"]["agents"]["practice_extractor"]
    assert report["invalid_evidence_count"] == 0 and report["error_count"] == 0 and report["warning_count"] == 0
    assert out["document"]["non_use_cues"] == {"cue_turn_count": 7, "uncovered_turn_ids": []}


def test_fabricated_practice_quote_in_a_chunk_is_rejected(tmp_path, long_run):
    def reader(params):
        output = S.practice_output(params)
        if output["practices"]:
            output["practices"][0] = copy.deepcopy(output["practices"][0])
            output["practices"][0]["evidence"][0]["quote"] = "Phrase inventée qui n'existe pas."
        return text_response(output)
    out = outputs(analyze(tmp_path, long_run, FakeTransport(responders(practice=reader))))
    assert out["validation"]["agents"]["practice_extractor"]["invalid_evidence_count"] >= 1
    assert out["manifest"]["status"] == "SUCCESS_WITH_WARNINGS" and out["manifest"]["validation_issue_count"] >= 1


def test_uncovered_non_use_cue_is_reported_not_corrected(tmp_path, long_run):
    def reader(params):  # le lecteur « oublie » le non-usage du tour T0150
        output = S.practice_output(params)
        output["practices"] = [p for p in output["practices"] if p["evidence"][0]["turn_id"] != S.tid(150)]
        return text_response(output)
    doc = outputs(analyze(tmp_path, long_run, FakeTransport(responders(practice=reader))))["document"]
    assert doc["non_use_cues"]["uncovered_turn_ids"] == [S.tid(150)]
    assert doc["item_count"] == S.EXPECTED_PRACTICE_COUNT - 1 and doc["status"] == "SUCCESS"  # information seulement


def test_practice_system_prompt_is_unchanged_by_chunking():
    """La définition des catégories (use, past_use, non_use, refusal…) n'est pas touchée : seul le message de bloc."""
    spec = analysis.chunk_spec(analysis.PRACTICE)
    assert spec.system_prompt == practice_extractor.SPEC.system_prompt
    assert spec.output_schema == practice_extractor.SPEC.output_schema and spec.version == "1.2"
    assert spec.prompt_sha256 != practice_extractor.SPEC.prompt_sha256  # gabarit de bloc ≠ gabarit d'entretien
