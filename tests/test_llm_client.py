"""Échange avec les agents : validation des réponses, erreurs, paramètres, schémas stricts. Aucun SDK, aucun réseau."""

import ast
import asyncio
import json
import socket
from pathlib import Path

import pytest

import core.llm_client as llm_client
from agents import (accountability_episode_builder, cross_interview_comparator, interaction_signal_reader,
                    practice_extractor, trajectory_mapper)
from agents.base import strict_json_schema
from agents.practice_extractor import SPEC as PRACTICE_SPEC
from core import speaker_attribution_auditor
from core.llm_client import LLMError, LLMSettings, parse_json_output
from tests import synthetic_interviews as si
from tests.fake_llm import PRACTICE, FakeAgents, agent_error, fake_settings, text_response

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = Path(__file__).resolve().parent / "data" / "agent_schemas_snapshot.json"
ALL_SPECS = (practice_extractor.SPEC, interaction_signal_reader.SPEC, interaction_signal_reader.LONG_DISTANCE_SPEC,
             speaker_attribution_auditor.SPEC, accountability_episode_builder.SPEC, trajectory_mapper.SPEC,
             cross_interview_comparator.SPEC)


def complete(client, **overrides):
    kwargs = {"system_prompt": PRACTICE_SPEC.system_prompt, "user_content": "<transcript>…</transcript>",
              "output_schema": PRACTICE_SPEC.output_schema, "response_model": PRACTICE_SPEC.output_model,
              "label": "TEST/practice_extractor"}
    kwargs.update(overrides)
    return asyncio.run(client.complete_json(**kwargs))


# --- Paramètres ---------------------------------------------------------------------------------

def test_settings_need_no_key_and_carry_the_workflow_label():
    settings = LLMSettings.from_env({})
    assert settings.model == llm_client.AGENT_RUNNER_MODEL == "workflow_claude_code"
    assert settings.interaction_chunk_tokens == llm_client.DEFAULT_INTERACTION_CHUNK_TOKENS
    assert settings.practice_chunk_tokens == llm_client.DEFAULT_PRACTICE_CHUNK_TOKENS
    assert settings.problems == ()
    assert settings.request_params() == {"effort": None, "temperature": None}  # champ legacy, neutre
    for name in ("api_key", "enabled", "missing", "max_concurrency", "max_retries", "timeout_seconds"):
        assert not hasattr(settings, name), name


def test_chunk_settings_read_from_env_with_explicit_problems():
    settings = LLMSettings.from_env({"TRACE_INTERACTION_CHUNK_TOKENS": "7000", "TRACE_PRACTICE_CHUNK_TOKENS": "3000"})
    assert (settings.interaction_chunk_tokens, settings.practice_chunk_tokens) == (7000, 3000)
    bad = LLMSettings.from_env({"TRACE_INTERACTION_CHUNK_TOKENS": "abc", "TRACE_PRACTICE_CHUNK_TOKENS": "99999"})
    assert bad.interaction_chunk_tokens == llm_client.DEFAULT_INTERACTION_CHUNK_TOKENS
    assert bad.practice_chunk_tokens == llm_client.INTERACTION_CHUNK_TOKENS_MAX
    assert len(bad.problems) == 2


def test_api_variables_are_ignored():
    """Les anciennes variables d'API n'ont plus aucun effet."""
    env = {"ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_MODEL": "m", "TRACE_MAX_CONCURRENCY": "8",
           "TRACE_LLM_EFFORT": "max", "TRACE_LLM_TEMPERATURE": "0"}
    assert LLMSettings.from_env(env) == LLMSettings.from_env({})


def test_no_model_identifier_hardcoded_in_code():
    """Aucun identifiant de modèle claude-* dans le code de TRACE."""
    for folder in ("core", "agents"):
        for path in (ROOT / folder).glob("*.py"):
            assert "claude-" not in path.read_text(encoding="utf-8"), path
    assert "claude-" not in (ROOT / "app.py").read_text(encoding="utf-8")


# --- Aucune dépendance à un SDK de modèle -------------------------------------------------------------

def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_no_model_sdk_or_http_client_is_imported():
    forbidden = {"anthropic", "openai", "google", "ollama", "httpx", "httpx2", "requests", "urllib3", "aiohttp"}
    paths = [ROOT / "app.py", *(ROOT / "core").glob("*.py"), *(ROOT / "agents").glob("*.py"),
             *(ROOT / "scripts").glob("*.py"), *(ROOT / "tests").rglob("*.py")]
    for path in paths:
        assert not (_imported_modules(path) & forbidden), path


def test_requirements_do_not_list_a_model_sdk():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "anthropic" not in requirements and "openai" not in requirements
    assert "pydantic" in requirements


def test_legacy_api_code_is_gone():
    for name in ("AnthropicTransport", "LLMClient", "Transport", "classify_exception", "backoff_delay", "redact"):
        assert not hasattr(llm_client, name), name
    assert not (ROOT / "scripts" / "smoke_test_stage3.py").exists()


def test_tests_cannot_reach_the_network():
    """Garde-fou de conftest : toute connexion TCP échoue immédiatement."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(RuntimeError):
        sock.connect(("127.0.0.1", 9))


# --- Schémas JSON stricts : identiques à la référence --------------------------------------------------

def test_agent_schemas_match_the_snapshot():
    """Schémas envoyés aux agents et empreintes (schema_sha256 : clés de cache, restaurations, imports de
    l'étape 6) strictement inchangés par le retrait du SDK."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert set(snapshot) == {spec.name for spec in ALL_SPECS}
    for spec in ALL_SPECS:
        assert spec.output_schema == snapshot[spec.name]["output_schema"], spec.name
        assert spec.schema_sha256 == snapshot[spec.name]["schema_sha256"], spec.name


def test_strict_schema_rules():
    from typing import Literal

    from pydantic import BaseModel, Field

    class Item(BaseModel):
        kind: Literal["a", "b"]
        tags: list[str] = Field(min_length=1, max_length=3, description="Étiquettes")
        note: str | None = None

    schema = strict_json_schema(Item)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["kind", "tags"]
    tags = schema["properties"]["tags"]
    assert tags["minItems"] == 1 and "maxItems" not in tags and "maxItems: 3" in tags["description"]
    assert schema["properties"]["note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


# --- Réponses d'agent (jamais acceptées silencieusement) ----------------------------------------------

def test_valid_response_is_parsed_with_neutral_accounting():
    client = FakeAgents({PRACTICE: text_response(si.GOOD_PRACTICES)})
    result = complete(client)
    assert len(result.data.practices) == len(si.GOOD_PRACTICES["practices"])
    assert result.attempts == 0  # aucun appel API
    assert result.usage == {"input_tokens": None, "output_tokens": None, "cache_creation_input_tokens": None,
                            "cache_read_input_tokens": None}


@pytest.mark.parametrize("text, code", [
    ("ceci n'est pas du JSON", "INVALID_JSON"),
    ('{"practices": [', "INVALID_JSON"),
    (json.dumps({"practices": [{"summary": "incomplet"}], "extraction_notes": None}), "SCHEMA_VALIDATION"),
    (json.dumps({"practices": [], "extraction_notes": None, "extra": 1}), "SCHEMA_VALIDATION"),
    ("", "EMPTY_RESPONSE"),
    ("   ", "EMPTY_RESPONSE"),
])
def test_invalid_responses_are_rejected(text, code):
    with pytest.raises(LLMError) as info:
        parse_json_output(text, PRACTICE_SPEC.output_model)
    assert info.value.code == code
    assert info.value.attempts == 0 and info.value.usage is None


def test_schema_error_message_never_contains_transcript_content():
    secret_text = "PHRASE CONFIDENTIELLE DE L'ENTRETIEN"
    bad = {"practices": [{**si.GOOD_PRACTICES["practices"][0], "use_status": secret_text}], "extraction_notes": None}
    with pytest.raises(LLMError) as info:
        complete(FakeAgents({PRACTICE: text_response(bad)}))
    assert info.value.code == "SCHEMA_VALIDATION"
    assert "use_status" in info.value.user_message and secret_text not in info.value.user_message


def test_schema_error_detail_is_bounded():
    bad = {"practices": [{"summary": "x"}] * 4, "extraction_notes": None}
    with pytest.raises(LLMError) as info:
        parse_json_output(json.dumps(bad), PRACTICE_SPEC.output_model, max_problems=2)
    assert "(+" in info.value.detail


def test_errors_have_safe_messages_and_stable_dict():
    error = LLMError("AWAITING_AGENT", request_id="practice_extractor-0123456789")
    assert error.to_dict() == {"code": "AWAITING_AGENT", "message": error.user_message, "status_code": None,
                               "request_id": "practice_extractor-0123456789"}
    assert "aucun appel API" in error.user_message and "practice_extractor-0123456789" in error.user_message
    assert LLMError("NOPE").user_message == llm_client.USER_MESSAGES["UNEXPECTED_ERROR"]
    assert agent_error().code == "SCHEMA_VALIDATION"


def test_fake_settings_carry_a_test_label():
    assert fake_settings().model == "fake-model"
