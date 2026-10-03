"""Performance du runner local : fenêtre de contexte par appel, génération bornée, mesures, corrections, ordre des
appels. Faux Ollama derrière le vrai LocalAgentRunner : aucun modèle, aucun réseau."""

import asyncio
import json

import pytest

from core import local_pipeline as lp
from core.llm_client import CTX_BUCKETS, LLMError
from core.local_agent_runner import (CTX_MARGIN, MIN_OUTPUT_TOKENS, PrefixAwareSlots, call_summary_line,
                                     context_plan)
from agents.practice_extractor import SPEC as PRACTICE_SPEC
from tests import synthetic_interviews as si
from tests import synthetic_long_interview as L
from tests.fake_llm import (INTERACTION, LONG_DISTANCE, PRACTICE, FakeLocalAgentRunner, OllamaRaw, fake_settings,
                            text_response, use_fake_runtime)


def complete(runner, user="<transcript>…</transcript>", label="TEST/practice_extractor"):
    return asyncio.run(runner.complete_json(system_prompt=PRACTICE_SPEC.system_prompt, user_content=user,
                                            output_schema=PRACTICE_SPEC.output_schema,
                                            response_model=PRACTICE_SPEC.output_model, label=label))


GOOD = text_response(si.GOOD_PRACTICES)


# --- Fenêtre de contexte : la plus petite qui contient tout, jamais tronquée -------------------------------------

def test_context_plan_picks_the_smallest_window_that_holds_prompt_and_answer():
    assert context_plan(2000, 2048, 32768) == (8192, 2048)
    assert context_plan(9900, 6144, 32768) == (20480, 6144)
    for prompt in (500, 3000, 9000, 12300, 20000):
        num_ctx, num_predict = context_plan(prompt, 6144, 32768)
        assert num_ctx in CTX_BUCKETS and prompt + num_predict + CTX_MARGIN <= num_ctx <= 32768


def test_context_plan_never_shrinks_below_the_loaded_window_nor_exceeds_the_maximum():
    assert context_plan(2000, 2048, 32768, floor=20480)[0] == 20480  # pas de rechargement du modèle
    assert context_plan(2000, 2048, 16384, floor=65536)[0] == 16384
    num_ctx, num_predict = context_plan(14000, 6144, 16384)  # la réponse réservée est réduite, jamais le prompt
    assert num_ctx == 16384 and num_predict == 16384 - 14000 - CTX_MARGIN >= MIN_OUTPUT_TOKENS


def test_a_prompt_longer_than_the_maximum_window_is_refused_before_any_call():
    with pytest.raises(LLMError) as info:
        context_plan(32000, 2048, 32768)
    assert info.value.code == "PROMPT_TOO_LONG" and "TRACE_OLLAMA_NUM_CTX" in info.value.user_message
    runner = FakeLocalAgentRunner({PRACTICE: GOOD}, settings=fake_settings(num_ctx=4096))
    with pytest.raises(LLMError) as info:
        complete(runner, user="x" * 30000)
    assert info.value.code == "PROMPT_TOO_LONG" and runner.calls == []  # rien n'est envoyé tronqué


def test_each_call_gets_its_own_window_and_a_bounded_answer():
    runner = FakeLocalAgentRunner({PRACTICE: GOOD})
    complete(runner, user="court")
    complete(runner, user="x" * 30000, label="TEST/practice_extractor/bloc2")
    complete(runner, user="court", label="TEST/practice_extractor/bloc3")
    options = [c["payload"]["options"] for c in runner.calls]
    assert [o["num_ctx"] for o in options] == [16384, 24576, 24576]  # croissante seulement : aucun rechargement
    assert all(o["num_predict"] == 8192 for o in options)
    assert all(c["payload"]["keep_alive"] == "30m" and c["payload"]["stream"] is True for c in runner.calls)


# --- Mesures ------------------------------------------------------------------------------------------------------

def test_every_attempt_is_measured_in_records_journal_and_events(tmp_path):
    events = []
    runner = FakeLocalAgentRunner({PRACTICE: GOOD}, journal_dir=tmp_path)
    runner.on_event = events.append
    result = complete(runner)
    [record] = runner.records.values()
    for key in ("prompt_chars", "estimated_input_tokens", "num_ctx", "num_predict", "input_tokens", "output_tokens",
                "duration_seconds", "generation_seconds", "tokens_per_second", "load_seconds", "response_chars",
                "started_at", "finished_at"):
        assert record[key] is not None, key
    assert record["tokens_per_second"] == 50.0 and record["agent_label"] == "Practice Extractor"
    journal = json.loads((tmp_path / f"{result.request_id}.json").read_text(encoding="utf-8"))
    [attempt] = journal["attempt_log"]
    for key in ("started_at", "finished_at", "duration_seconds", "prompt_chars", "estimated_input_tokens", "num_ctx",
                "num_predict", "prompt_eval_count", "prompt_eval_seconds", "eval_count", "eval_seconds",
                "tokens_per_second", "load_seconds", "response_chars", "done_reason"):
        assert key in attempt, key
    assert journal["options"]["num_ctx"] == record["num_ctx"] and journal["options"]["num_ctx_max"] == 32768
    assert [e["type"] for e in events] == ["call_started", "tokens", "call_finished"]
    line = call_summary_line(record)
    assert line.startswith("Practice Extractor — TEST — ") and "tokens entrée" in line and "tok/s" in line


def test_runner_logs_one_line_per_call(caplog):
    caplog.set_level("INFO", logger="trace.local")
    complete(FakeLocalAgentRunner({PRACTICE: GOOD}))
    assert any("tokens entrée" in r.getMessage() and "Practice Extractor" in r.getMessage() for r in caplog.records)


# --- Corrections : budgets séparés, une réponse tronquée est redemandée sans être renvoyée -------------------------

def test_a_truncated_answer_is_asked_again_with_a_larger_budget_without_resending_it():
    runner = FakeLocalAgentRunner({PRACTICE: [OllamaRaw('{"practices": [', done_reason="length"), GOOD]})
    complete(runner)
    first, second = runner.calls
    assert len(second["payload"]["messages"]) == 2  # requête d'origine, sans la réponse coupée
    assert second["payload"]["options"]["num_predict"] == 2 * first["payload"]["options"]["num_predict"]
    [record] = runner.records.values()
    assert record["correction_reasons"] == ["réponse tronquée"] and record["outcome"] == "accepted"


def test_format_and_method_corrections_have_separate_budgets():
    blocking = ["QUOTE_NOT_FOUND — P001 : citation absente du tour cité"]
    runner = FakeLocalAgentRunner({PRACTICE: ["pas du JSON", "toujours pas", GOOD, GOOD]},
                                  checker=lambda spec, label, content, output: list(blocking))
    complete(runner)
    # 2 corrections de format (défaut), puis 1 correction méthodologique (défaut) : 4 générations au plus
    assert len(runner.calls) == 4
    [record] = runner.records.values()
    assert record["correction_reasons"] == ["INVALID_JSON", "INVALID_JSON", "validateur méthodologique"]
    assert record["outcome"] == "accepted_with_issues" and record["remaining_issues"] == blocking


def test_method_corrections_can_be_disabled_and_the_validator_still_reports():
    runner = FakeLocalAgentRunner({PRACTICE: GOOD}, settings=fake_settings(max_method_corrections=0),
                                  checker=lambda spec, label, content, output: ["QUOTE_NOT_FOUND — P001"])
    complete(runner)
    assert len(runner.calls) == 1 and next(iter(runner.records.values()))["outcome"] == "accepted_with_issues"


# --- Ordre des appels : un agent après l'autre (préfixe réutilisé par Ollama) --------------------------------------

def test_calls_of_the_same_agent_are_grouped(tmp_path, monkeypatch):
    ollama = use_fake_runtime(monkeypatch, {PRACTICE: lambda p: text_response({"practices": [],
                                                                                 "extraction_notes": None}),
                                            INTERACTION: L.simulated_chunk_reader,
                                            LONG_DISTANCE: L.simulated_long_distance_reader})
    lp.run_stage("3", si.make_ingested_run(tmp_path, L.long_files()))
    agents = [c["agent"] for c in ollama.calls]
    switches = sum(a != b for a, b in zip(agents, agents[1:]))
    assert agents.count(PRACTICE) > 1 and agents.count(INTERACTION) > 1 and switches == 2  # P… → I… → longue distance


def test_prefix_aware_slots_serve_the_same_agent_first():
    async def scenario():
        slots, order = PrefixAwareSlots(1), []

        async def call(key, name):
            async with slots.slot(key):
                order.append(name)
                await asyncio.sleep(0)

        await asyncio.gather(call("A", "a1"), call("B", "b1"), call("A", "a2"), call("B", "b2"), call("A", "a3"))
        return order
    assert asyncio.run(scenario()) == ["a1", "a2", "a3", "b1", "b2"]
