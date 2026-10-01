"""Garde-fous communs : AUCUN test ne peut appeler la vraie API Anthropic.

- les variables ANTHROPIC_* / TRACE_* de l'environnement sont retirées ;
- le fichier .env local n'est jamais chargé ;
- le transport HTTP réel est remplacé par une classe qui échoue ;
- toute connexion réseau TCP est refusée ;
- le cache d'analyse pointe vers un dossier temporaire.
"""

import socket

import pytest

import core.llm_client as llm_client
from core import config

_ENV_VARS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL", "ANTHROPIC_PROFILE",
    llm_client.ENV_MAX_CONCURRENCY, llm_client.ENV_TIMEOUT, llm_client.ENV_MAX_RETRIES,
    llm_client.ENV_MAX_TOKENS, llm_client.ENV_EFFORT, llm_client.ENV_TEMPERATURE,
    llm_client.ENV_INTERACTION_CHUNK_TOKENS,
)

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


class RealTransportForbidden:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Transport Anthropic réel utilisé dans un test : interdit.")


def _guard(real):
    def guarded(self, address, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"Connexion réseau interdite pendant les tests : {address!r}")
        return real(self, address, *args, **kwargs)
    return guarded


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch, tmp_path):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_env_file", lambda path=None: [])
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "analysis_cache")
    monkeypatch.setattr(llm_client, "AnthropicTransport", RealTransportForbidden)
    monkeypatch.setattr(socket.socket, "connect", _guard(_real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(_real_connect_ex))
    # Réessais sans attente réelle
    monkeypatch.setattr(llm_client, "BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setattr(llm_client, "RETRY_AFTER_MAX_SECONDS", 0.0)
