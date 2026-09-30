"""Tests du client LLM : configuration, réessais, délais, réponses invalides. Aucun appel réel."""

import asyncio
import json
from types import SimpleNamespace

import anthropic
import pytest

import core.llm_client as llm_client
from agents.practice_extractor import SPEC as PRACTICE_SPEC
from core.llm_client import AnthropicTransport, LLMClient, LLMError, LLMSettings, backoff_delay
from tests import synthetic_interviews as si
from tests.fake_llm import (PRACTICE, FakeTransport, fake_settings, no_sleep, rate_limit_error, server_error,
                            status_error, text_response, timeout_error)

KEY = "sk-ant-test-FAKE-KEY-000"


def complete(client, **overrides):
    kwargs = {"system_prompt": PRACTICE_SPEC.system_prompt, "user_content": "<transcript>…</transcript>",
              "output_schema": PRACTICE_SPEC.output_schema, "response_model": PRACTICE_SPEC.output_model,
              "label": "TEST/practice_extractor"}
    kwargs.update(overrides)
    return asyncio.run(client.complete_json(**kwargs))


def client_with(responses, **settings):
    transport = FakeTransport({PRACTICE: responses})
    return LLMClient(fake_settings(**settings), transport=transport, sleep=no_sleep), transport


# --- Configuration ----------------------------------------------------------------------

def test_missing_key_and_model_disable_analysis():
    settings = LLMSettings.from_env({})
    assert not settings.enabled
    assert settings.missing == ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"]
    assert "ANTHROPIC_API_KEY" in settings.disabled_reason() and "ingestion" in settings.disabled_reason()
    with pytest.raises(LLMError) as info:
        LLMClient(settings)
    assert info.value.code == "NOT_CONFIGURED"


def test_missing_model_disables_analysis():
    settings = LLMSettings.from_env({"ANTHROPIC_API_KEY": KEY})
    assert settings.missing == ["ANTHROPIC_MODEL"]
    assert KEY not in settings.disabled_reason()
    with pytest.raises(LLMError):
        LLMClient(settings)


def test_settings_read_from_env_without_leaking_key():
    settings = LLMSettings.from_env({"ANTHROPIC_API_KEY": f"  {KEY} ", "ANTHROPIC_MODEL": "fake-model",
                                     "TRACE_MAX_CONCURRENCY": "3", "TRACE_LLM_EFFORT": "LOW"})
    assert settings.enabled and settings.api_key == KEY and settings.model == "fake-model"
    assert settings.max_concurrency == 3 and settings.effort == "low" and settings.temperature is None
    assert KEY not in repr(settings)


def test_invalid_settings_fall_back_with_explicit_problems():
    settings = LLMSettings.from_env({"TRACE_MAX_CONCURRENCY": "50", "TRACE_LLM_MAX_RETRIES": "abc",
                                     "TRACE_LLM_EFFORT": "turbo", "TRACE_LLM_TIMEOUT_SECONDS": "-1"})
    assert settings.max_concurrency == llm_client.MAX_CONCURRENCY_LIMIT
    assert settings.max_retries == llm_client.DEFAULT_MAX_RETRIES
    assert settings.effort is None and settings.timeout_seconds == llm_client.DEFAULT_TIMEOUT_SECONDS
    assert len(settings.problems) == 4


def test_no_model_identifier_hardcoded_in_code():
    """Le modèle vient uniquement de ANTHROPIC_MODEL : aucun identifiant claude-* dans le code."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for folder in ("core", "agents"):
        for path in (root / folder).glob("*.py"):
            assert "claude-" not in path.read_text(encoding="utf-8"), path
    assert "claude-" not in (root / "app.py").read_text(encoding="utf-8")


# --- Requête ----------------------------------------------------------------------------

def test_request_uses_structured_output_and_settings():
    client, transport = client_with([text_response(si.GOOD_PRACTICES)])
    complete(client)
    params = transport.calls[0]["params"]
    assert params["model"] == "fake-model"
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": PRACTICE_SPEC.output_schema}
    assert "effort" not in params["output_config"] and "extra_body" not in params
    assert params["system"][0]["text"] == PRACTICE_SPEC.system_prompt
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"] == [{"role": "user", "content": "<transcript>…</transcript>"}]


def test_optional_effort_and_temperature_are_sent_only_when_configured():
    client, transport = client_with([text_response(si.GOOD_PRACTICES)], effort="low", temperature=0.0)
    complete(client)
    params = transport.calls[0]["params"]
    assert params["output_config"]["effort"] == "low"
    assert params["extra_body"] == {"temperature": 0.0}


def test_valid_response_is_parsed_and_usage_recorded():
    client, transport = client_with([text_response(si.GOOD_PRACTICES, input_tokens=18432, output_tokens=3210,
                                                   cache_read=100)])
    result = complete(client)
    assert len(result.data.practices) == len(si.GOOD_PRACTICES["practices"])
    assert result.usage == {"input_tokens": 18432, "output_tokens": 3210, "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 100}
    assert result.attempts == 1 and result.response_model == "fake-model" and result.request_id == "req_fake"


# --- Réponses invalides (jamais acceptées silencieusement) -------------------------------

@pytest.mark.parametrize("response, code", [
    (text_response("ceci n'est pas du JSON"), "INVALID_JSON"),
    (text_response({"practices": [{"summary": "incomplet"}], "extraction_notes": None}), "SCHEMA_VALIDATION"),
    (text_response({"practices": [], "extraction_notes": None, "extra": 1}), "SCHEMA_VALIDATION"),
    (text_response(si.GOOD_PRACTICES, stop_reason="max_tokens"), "TRUNCATED"),
    (text_response(si.GOOD_PRACTICES, stop_reason="refusal"), "REFUSAL"),
    (text_response(""), "EMPTY_RESPONSE"),
])
def test_invalid_responses_are_rejected_without_retry(response, code):
    client, transport = client_with([response])
    with pytest.raises(LLMError) as info:
        complete(client)
    assert info.value.code == code
    assert len(transport.calls) == 1              # pas de réessai payant
    assert info.value.usage["input_tokens"] == 1200  # le coût de l'échec reste visible


def test_schema_error_message_never_contains_transcript_content():
    secret_text = "PHRASE CONFIDENTIELLE DE L'ENTRETIEN"
    bad = {"practices": [{**si.GOOD_PRACTICES["practices"][0], "use_status": secret_text}], "extraction_notes": None}
    client, _ = client_with([text_response(bad)])
    with pytest.raises(LLMError) as info:
        complete(client)
    assert info.value.code == "SCHEMA_VALIDATION"
    assert "use_status" in info.value.user_message and secret_text not in info.value.user_message


# --- Erreurs transitoires : réessais bornés ----------------------------------------------

def test_429_then_success_is_retried():
    client, transport = client_with([rate_limit_error("1"), text_response(si.GOOD_PRACTICES)])
    result = complete(client)
    assert result.attempts == 2 and len(transport.calls) == 2


def test_429_retries_are_bounded():
    client, transport = client_with([rate_limit_error() for _ in range(10)], max_retries=2)
    with pytest.raises(LLMError) as info:
        complete(client)
    assert info.value.code == "RATE_LIMITED" and info.value.status_code == 429
    assert len(transport.calls) == 3 and info.value.attempts == 3


def test_5xx_then_success_is_retried():
    client, transport = client_with([server_error(503), status_error(anthropic.OverloadedError, 529),
                                     text_response(si.GOOD_PRACTICES)])
    assert complete(client).attempts == 3


def test_simulated_sdk_timeout_is_retried_then_reported():
    client, transport = client_with([timeout_error() for _ in range(5)], max_retries=1)
    with pytest.raises(LLMError) as info:
        complete(client)
    assert info.value.code == "TIMEOUT" and len(transport.calls) == 2


def test_wall_clock_timeout_is_enforced():
    async def slow(_params):
        await asyncio.sleep(5)

    client, transport = client_with(slow, timeout_seconds=0.05, max_retries=0)
    with pytest.raises(LLMError) as info:
        complete(client)
    assert info.value.code == "TIMEOUT"


@pytest.mark.parametrize("error, code", [
    (status_error(anthropic.AuthenticationError, 401), "AUTHENTICATION_ERROR"),
    (status_error(anthropic.PermissionDeniedError, 403), "PERMISSION_DENIED"),
    (status_error(anthropic.NotFoundError, 404), "MODEL_NOT_FOUND"),
    (status_error(anthropic.BadRequestError, 400), "BAD_REQUEST"),
])
def test_client_errors_are_not_retried(error, code):
    client, transport = client_with([error, text_response(si.GOOD_PRACTICES)])
    with pytest.raises(LLMError) as info:
        complete(client)
    assert info.value.code == code and len(transport.calls) == 1


def test_api_key_is_redacted_from_error_messages():
    error = status_error(anthropic.BadRequestError, 400, message=f"invalid header x-api-key: {KEY}")
    client, _ = client_with([error])
    with pytest.raises(LLMError) as info:
        complete(client)
    assert KEY not in info.value.user_message and "[clé masquée]" in info.value.user_message
    assert KEY not in json.dumps(info.value.to_dict())


def test_backoff_is_exponential_bounded_and_honours_retry_after(monkeypatch):
    monkeypatch.setattr(llm_client, "BACKOFF_BASE_SECONDS", 2.0)
    monkeypatch.setattr(llm_client, "RETRY_AFTER_MAX_SECONDS", 60.0)
    assert 1.5 <= backoff_delay(1) <= 2.5
    assert 3.0 <= backoff_delay(2) <= 5.0
    assert backoff_delay(10) <= llm_client.BACKOFF_MAX_SECONDS
    assert backoff_delay(1, retry_after=7.0) == 7.0
    assert backoff_delay(1, retry_after=3600.0) == 60.0


# --- Concurrence ------------------------------------------------------------------------

def test_concurrency_is_limited_by_semaphore():
    async def slow_ok(_params):
        await asyncio.sleep(0.01)
        return text_response(si.GOOD_PRACTICES)

    client, transport = client_with(slow_ok, max_concurrency=2)

    async def many():
        kwargs = {"system_prompt": "s", "user_content": "u", "output_schema": PRACTICE_SPEC.output_schema,
                  "response_model": PRACTICE_SPEC.output_model, "label": "x"}
        await asyncio.gather(*(client.complete_json(**kwargs) for _ in range(6)))

    asyncio.run(many())
    assert len(transport.calls) == 6 and transport.max_active == 2


# --- Transport réel (SDK simulé : aucune connexion) ---------------------------------------

def test_real_transport_passes_only_the_explicit_key_and_disables_sdk_retries(monkeypatch):
    created = {}

    class FakeStream:
        request_id = "req_stream"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_final_message(self):
            return SimpleNamespace(content=[])

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.messages = self

        def stream(self, **params):
            created["params"] = params
            return FakeStream()

        async def close(self):
            created["closed"] = True

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsyncAnthropic)
    transport = AnthropicTransport(fake_settings())
    assert created["api_key"] == KEY and created["max_retries"] == 0
    assert asyncio.run(transport({"model": "fake-model"}))._request_id == "req_stream"
    asyncio.run(transport.aclose())
    assert created["params"] == {"model": "fake-model"} and created["closed"]


def test_tests_cannot_reach_the_real_api():
    """Garde-fou de conftest : sans transport injecté, le client réel est interdit."""
    with pytest.raises(AssertionError):
        LLMClient(fake_settings())


def test_real_sdk_streaming_path_offline():
    """Chemin SDK réel (flux SSE) sur un faux serveur HTTP en mémoire : requête et réponse de bout en bout."""
    import httpx2

    captured = {}
    payload = json.dumps(si.GOOD_PRACTICES, ensure_ascii=False)

    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def handler(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        body = "".join([
            sse("message_start", {"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "fake-model", "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 321, "output_tokens": 1, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0}}}),
            sse("content_block_start", {"type": "content_block_start", "index": 0,
                                        "content_block": {"type": "text", "text": ""}}),
            sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": payload[:40]}}),
            sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": payload[40:]}}),
            sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                  "usage": {"output_tokens": 654}}),
            sse("message_stop", {"type": "message_stop"}),
        ])
        return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_offline"},
                               content=body.encode("utf-8"))

    async def scenario():
        http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        settings = fake_settings()
        client = LLMClient(settings, transport=AnthropicTransport(settings, http_client=http_client), sleep=no_sleep)
        try:
            return await client.complete_json(
                system_prompt=PRACTICE_SPEC.system_prompt, user_content="<transcript>…</transcript>",
                output_schema=PRACTICE_SPEC.output_schema, response_model=PRACTICE_SPEC.output_model, label="offline")
        finally:
            await client.aclose()

    result = asyncio.run(scenario())
    assert len(result.data.practices) == 6 and result.request_id == "req_offline"
    assert result.usage["input_tokens"] == 321 and result.usage["output_tokens"] == 654
    body, headers = captured["body"], captured["headers"]
    assert headers["x-api-key"] == KEY and "authorization" not in headers
    assert body["model"] == "fake-model" and body["stream"] is True
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in body
