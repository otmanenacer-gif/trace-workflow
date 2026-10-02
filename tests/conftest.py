"""Garde-fous communs : AUCUN test ne peut joindre le réseau ni dépendre de l'environnement local.

- les variables TRACE_* lues par TRACE sont retirées de l'environnement ;
- le fichier .env local n'est jamais chargé ;
- toute connexion réseau TCP (IPv4 / IPv6) est refusée immédiatement ;
- le cache d'analyse pointe vers un dossier temporaire.

Les agents sont simulés (tests/fake_llm.FakeAgents) ou joués par des réponses écrites dans les paquets de tâche
du workflow (core/claude_code_workflow.py) : aucun SDK de modèle, aucune clé.
"""

import socket

import pytest

import core.llm_client as llm_client
from core import config

_ENV_VARS = (llm_client.ENV_INTERACTION_CHUNK_TOKENS, llm_client.ENV_PRACTICE_CHUNK_TOKENS)

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _guard(real):
    def guarded(self, address, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"Connexion réseau interdite pendant les tests : {address!r}")
        return real(self, address, *args, **kwargs)
    return guarded


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_env_file", lambda path=None: [])
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "analysis_cache")
    monkeypatch.setattr(socket.socket, "connect", _guard(_real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(_real_connect_ex))
