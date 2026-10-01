"""Faux transport LLM pour les tests : aucune connexion réseau, aucun token consommé.

Il remplace l'appel HTTP au niveau le plus bas (LLMClient reste réel : réessais,
validation JSON, sémaphore, usage). Les réponses imitent la forme des objets
`Message` du SDK (content, usage, stop_reason, model, _request_id).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import anthropic
import httpx2

PRACTICE = "practice_extractor"
INTERACTION = "interaction_signal_reader"
AUDITOR = "speaker_attribution_auditor"
LONG_DISTANCE = "interaction_signal_reader_long_distance"  # lecture à longue distance (entretien long)


def text_response(payload, *, input_tokens=1200, output_tokens=300, stop_reason="end_turn", model="fake-model",
                  cache_read=0, cache_creation=0, request_id="req_fake"):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=cache_creation, cache_read_input_tokens=cache_read),
        stop_reason=stop_reason,
        model=model,
    )
    response._request_id = request_id
    return response


def _request():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, status: int, headers: dict | None = None, message: str = "error"):
    response = httpx2.Response(status, request=_request(), headers={"request-id": "req_err", **(headers or {})})
    return cls(message, response=response, body=None)


def rate_limit_error(retry_after: str | None = None):
    return status_error(anthropic.RateLimitError, 429, {"retry-after": retry_after} if retry_after else None)


def server_error(status: int = 500):
    return status_error(anthropic.InternalServerError, status)


def timeout_error():
    return anthropic.APITimeoutError(request=_request())


def agent_of(params: dict) -> str:
    properties = params["output_config"]["format"]["schema"]["properties"]
    if "assessments" in properties:
        return AUDITOR
    if params["system"][0]["text"].startswith("# Interaction Signal Reader — lecture à longue distance"):
        return LONG_DISTANCE
    return PRACTICE if "practices" in properties else INTERACTION


class FakeTransport:
    """Répond selon l'agent (déduit du schéma demandé) : PRACTICE, INTERACTION, AUDITOR ou LONG_DISTANCE.

    Un appel à un agent sans réponse prévue échoue (KeyError) : un test qui ne prévoit
    pas d'audit des locuteurs vérifie donc aussi qu'aucun appel d'audit n'a lieu.

    `responders[agent]` peut être : une réponse, une exception, une liste
    (consommée dans l'ordre) ou une fonction (params) -> réponse (éventuellement async).
    """

    def __init__(self, responders: dict):
        self.responders = {k: (list(v) if isinstance(v, list) else v) for k, v in responders.items()}
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def __call__(self, params: dict):
        agent = agent_of(params)
        self.calls.append({"agent": agent, "params": params})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            handler = self.responders[agent]
            if isinstance(handler, list):
                handler = handler.pop(0)
            result = handler(params) if callable(handler) and not isinstance(handler, BaseException) else handler
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            self.active -= 1

    async def aclose(self):
        self.closed = True

    def calls_for(self, agent: str) -> list[dict]:
        return [c for c in self.calls if c["agent"] == agent]


def fake_settings(**overrides):
    from core.llm_client import LLMSettings

    values = {"api_key": "sk-ant-test-FAKE-KEY-000", "model": "fake-model", "max_concurrency": 2, "max_retries": 2,
              "timeout_seconds": 5.0}
    values.update(overrides)
    return LLMSettings(**values)


async def no_sleep(_delay):
    return None
