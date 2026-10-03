"""LocalAgentRunner : Ollama local uniquement, sortie structurée, corrections locales, erreurs claires, aucun repli.

Aucun modèle réel : faux Ollama (tests/fake_llm.FakeOllama) ou petit serveur HTTP local qui imite l'API d'Ollama
(pour tester le vrai transport HTTP). Aucune connexion hors de la boucle locale.
"""

import asyncio
import http.server
import ipaddress
import json
import socket
import threading

import pytest

import tests.conftest as guards
from agents.practice_extractor import SPEC as PRACTICE_SPEC
from core import llm_client
from core.llm_client import DEFAULT_LOCAL_MODEL, DEFAULT_OLLAMA_URL, LLMError, LLMSettings
from core.local_agent_runner import (CORRECTION_INSTRUCTIONS, LocalAgentRunner, OllamaTransport, check_runtime,
                                     ensure_local_url, require_runtime)
from tests import synthetic_interviews as si
from tests.fake_llm import PRACTICE, FakeLocalAgentRunner, OllamaRaw, fake_settings, text_response

LABEL = "ENTRETIEN_SYNTHETIQUE/practice_extractor"


def complete(runner, label=LABEL, user_content="<transcript>…</transcript>"):
    return asyncio.run(runner.complete_json(system_prompt=PRACTICE_SPEC.system_prompt, user_content=user_content,
                                            output_schema=PRACTICE_SPEC.output_schema,
                                            response_model=PRACTICE_SPEC.output_model, label=label))


# --- Paramètres : local, sans clé ---------------------------------------------------------------------------

def test_defaults_are_local_ollama_without_any_key():
    settings = LLMSettings.from_env({})
    assert settings.model == DEFAULT_LOCAL_MODEL == "qwen2.5:7b"
    assert settings.ollama_url == DEFAULT_OLLAMA_URL == "http://localhost:11434"
    assert settings.max_corrections == 2 and settings.temperature == 0.0 and settings.num_ctx == 32768
    assert not any("key" in name.lower() for name in settings.__dataclass_fields__)


def test_model_and_url_come_from_the_environment():
    settings = LLMSettings.from_env({"TRACE_LOCAL_MODEL": "mistral-small3.2", "TRACE_OLLAMA_URL": "http://127.0.0.1:9999/",
                                     "TRACE_LOCAL_MAX_CORRECTIONS": "9", "TRACE_OLLAMA_NUM_CTX": "abc"})
    assert settings.model == "mistral-small3.2" and settings.ollama_url == "http://127.0.0.1:9999"
    assert settings.max_corrections == llm_client.MAX_CORRECTIONS_LIMIT
    assert settings.num_ctx == llm_client.DEFAULT_NUM_CTX and len(settings.problems) == 2


def test_api_keys_in_the_environment_are_ignored():
    env = {"ANTHROPIC_API_KEY": "sk-ant-x", "OPENAI_API_KEY": "sk-x", "GEMINI_API_KEY": "x", "ANTHROPIC_MODEL": "m"}
    assert LLMSettings.from_env(env) == LLMSettings.from_env({})


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434",
                                 "http://127.0.0.2:8080/"])
def test_loopback_addresses_are_accepted(url):
    assert ensure_local_url(url) == url.rstrip("/")


@pytest.mark.parametrize("url", ["http://192.168.1.20:11434", "http://10.0.0.5:11434", "https://api.openai.com",
                                 "https://api.anthropic.com", "http://ollama.example.com:11434", "ftp://localhost",
                                 "localhost:11434"])
def test_non_local_addresses_are_refused_before_any_connection(url):
    with pytest.raises(LLMError) as info:
        OllamaTransport(url)
    assert info.value.code == "NON_LOCAL_URL" and "aucune donnée ne quitte cet ordinateur" in info.value.user_message
    status = check_runtime(LLMSettings(ollama_url=url))
    assert not status["available"] and status["error"]["code"] == "NON_LOCAL_URL"


# --- Requête structurée, réponse valide ----------------------------------------------------------------------

def test_request_uses_the_agent_prompt_and_its_strict_json_schema(tmp_path):
    runner = FakeLocalAgentRunner({PRACTICE: text_response(si.GOOD_PRACTICES)}, journal_dir=tmp_path)
    result = complete(runner)
    [call] = runner.calls
    payload = call["payload"]
    assert payload["model"] == "fake-model" and payload["stream"] is True  # progression en flux
    assert payload["format"] == PRACTICE_SPEC.output_schema  # sortie structurée : le JSON Schema strict de l'agent
    # fenêtre dimensionnée pour CET appel (prompt + réponse réservée + marge), réponse bornée, modèle gardé chargé
    assert payload["options"] == {"temperature": 0.0, "num_ctx": 16384, "num_predict": 8192}
    assert payload["keep_alive"] == "30m"
    assert payload["messages"][0] == {"role": "system", "content": PRACTICE_SPEC.system_prompt}  # prompt du dépôt
    assert payload["messages"][1]["content"] == "<transcript>…</transcript>" and len(payload["messages"]) == 2
    assert len(result.data.practices) == len(si.GOOD_PRACTICES["practices"])
    assert result.attempts == 0  # champ legacy « appels API » : toujours 0
    assert result.usage["input_tokens"] > 0 and result.model_requested == "fake-model"
    journal = json.loads((tmp_path / f"{result.request_id}.json").read_text(encoding="utf-8"))
    assert journal["outcome"] == "accepted" and journal["attempts"] == 1 and journal["execution"] == "locale"
    assert journal["runtime"] == "Ollama" and journal["agent"] == "practice_extractor"


# --- Corrections LOCALES ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad, code", [
    ("ceci n'est pas du JSON", "INVALID_JSON"),
    (json.dumps({"practices": [{"summary": "incomplet"}], "extraction_notes": None}), "SCHEMA_VALIDATION"),
    ("", "EMPTY_RESPONSE"),
])
def test_invalid_response_is_corrected_locally_with_the_error(bad, code, tmp_path):
    runner = FakeLocalAgentRunner({PRACTICE: [bad, text_response(si.GOOD_PRACTICES)]}, journal_dir=tmp_path)
    result = complete(runner)
    first, second = runner.calls
    history = second["params"]["history"]
    assert history[0] == {"role": "assistant", "content": bad}  # sa réponse…
    assert history[1]["role"] == "user" and code in history[1]["content"]  # … et l'erreur, pour corriger
    assert CORRECTION_INSTRUCTIONS in history[1]["content"]
    assert len(result.data.practices) == len(si.GOOD_PRACTICES["practices"])
    journal = json.loads((tmp_path / f"{result.request_id}.json").read_text(encoding="utf-8"))
    assert [a["status"] for a in journal["attempt_log"]] == ["invalid", "accepted"]


def test_corrections_are_bounded_then_the_error_is_reported(tmp_path):
    runner = FakeLocalAgentRunner({PRACTICE: lambda p: "toujours pas du JSON"}, journal_dir=tmp_path)
    with pytest.raises(LLMError) as info:
        complete(runner)
    assert info.value.code == "INVALID_JSON" and len(runner.calls) == 1 + runner.settings.max_corrections
    journal = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert journal["outcome"] == "failed" and journal["error"]["code"] == "INVALID_JSON"


def test_no_correction_when_disabled():
    runner = FakeLocalAgentRunner({PRACTICE: lambda p: "x"}, settings=fake_settings(max_corrections=0))
    with pytest.raises(LLMError):
        complete(runner)
    assert len(runner.calls) == 1


def test_methodological_errors_are_sent_back_and_the_validator_is_never_bypassed():
    blocking = ["QUOTE_NOT_FOUND — P001 : citation absente du tour cité"]
    runner = FakeLocalAgentRunner({PRACTICE: text_response(si.GOOD_PRACTICES)},
                                  checker=lambda spec, label, content, output: list(blocking))
    result = complete(runner)  # la réponse reste conforme au schéma : rendue telle quelle, anomalies signalées
    assert len(runner.calls) == 1 + runner.settings.max_method_corrections  # chaque correction régénère tout
    assert "QUOTE_NOT_FOUND" in runner.calls[1]["params"]["history"][1]["content"]
    [record] = runner.records.values()
    assert record["outcome"] == "accepted_with_issues" and record["remaining_issues"] == blocking
    assert result.data is not None


def test_truncated_response_is_never_accepted():
    runner = FakeLocalAgentRunner({PRACTICE: lambda p: OllamaRaw('{"practices": [', done_reason="length")})
    with pytest.raises(LLMError) as info:
        complete(runner)
    assert info.value.code == "TRUNCATED" and "TRACE_OLLAMA_NUM_CTX" in info.value.user_message


# --- Runtime absent : erreur claire, jamais de repli ----------------------------------------------------------

def test_ollama_unavailable_is_a_clear_error_without_retry_nor_fallback():
    runner = FakeLocalAgentRunner({PRACTICE: text_response(si.GOOD_PRACTICES)}, available=False)
    with pytest.raises(LLMError) as info:
        complete(runner)
    assert info.value.code == "OLLAMA_UNAVAILABLE" and "ollama serve" in info.value.user_message
    assert "Aucun autre fournisseur" in info.value.user_message and runner.calls == []
    status = runner.check()
    assert status["available"] is False and status["ready"] is False and status["start_command"] == "ollama serve"
    with pytest.raises(LLMError) as info:
        runner.require_ready()
    assert info.value.code == "OLLAMA_UNAVAILABLE" and "ollama serve" in info.value.user_message


def test_missing_model_is_a_clear_error_with_the_install_command():
    runner = FakeLocalAgentRunner({PRACTICE: text_response(si.GOOD_PRACTICES)}, models=["llama3.2:3b"],
                                  settings=fake_settings(model="qwen2.5:7b"))
    status = runner.check()
    assert status["available"] and not status["model_installed"] and status["models"] == ["llama3.2:3b"]
    assert status["install_command"] == "ollama pull qwen2.5:7b"
    with pytest.raises(LLMError) as info:
        runner.require_ready()
    assert info.value.code == "MODEL_NOT_FOUND" and "ollama pull qwen2.5:7b" in info.value.user_message
    with pytest.raises(LLMError) as info:
        complete(runner)
    assert info.value.code == "MODEL_NOT_FOUND" and runner.calls == []


def test_ready_runtime_status_says_local_and_no_external_data():
    status = FakeLocalAgentRunner({}).check()
    assert status["ready"] and status["execution"] == "locale" and status["runtime"] == "Ollama"
    assert status["external_data"] == "aucune" and status["model"] == "fake-model"


# --- Vrai transport HTTP, contre un serveur local qui imite Ollama ----------------------------------------------

@pytest.fixture
def loopback_only(monkeypatch):
    """Autorise les connexions vers la boucle locale SEULEMENT (le reste reste interdit)."""
    def connect(self, address, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not ipaddress.ip_address(address[0]).is_loopback:
            raise RuntimeError(f"Connexion réseau interdite pendant les tests : {address!r}")
        return guards._real_connect(self, address, *args, **kwargs)
    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def ollama_stub(loopback_only):
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            return None

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/version":
                self._send(200, {"version": "0.9.9"})
            elif self.path == "/api/tags":
                self._send(200, {"models": [{"name": "qwen2.5:7b"}, {"name": "mistral:latest"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"path": self.path, "body": body})
            if body["model"] not in ("qwen2.5:7b", "mistral:latest"):
                self._send(404, {"error": f'model "{body["model"]}" not found, try pulling it first'})
                return
            content = json.dumps(si.GOOD_PRACTICES)
            final = {"model": body["model"], "done": True, "done_reason": "stop", "prompt_eval_count": 1234,
                     "prompt_eval_duration": 617_000_000, "eval_count": 567, "eval_duration": 14_175_000_000,
                     "load_duration": 2_000_000_000}
            if not body.get("stream"):
                self._send(200, {**final, "message": {"role": "assistant", "content": content}})
                return
            # flux NDJSON, comme Ollama : un fragment par token, puis le bilan
            pieces = [content[i:i + 40] for i in range(0, len(content), 40)]
            lines = [{"model": body["model"], "message": {"role": "assistant", "content": piece}, "done": False}
                     for piece in pieces] + [{**final, "message": {"role": "assistant", "content": ""}}]
            data = "".join(json.dumps(line) + "\n" for line in lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", received
    server.shutdown()
    server.server_close()


def test_real_http_transport_talks_to_a_local_ollama_and_ignores_proxies(ollama_stub, monkeypatch):
    url, received = ollama_stub
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:3128")  # jamais utilisé : boucle locale, sans proxy
    monkeypatch.setenv("http_proxy", "http://10.255.255.1:3128")
    settings = LLMSettings(model="qwen2.5:7b", ollama_url=url)
    status = check_runtime(settings)
    assert status["ready"] and status["version"] == "0.9.9" and "mistral:latest" in status["models"]
    assert check_runtime(LLMSettings(model="mistral", ollama_url=url))["model_installed"]  # « latest » implicite
    require_runtime(settings)
    result = complete(LocalAgentRunner(settings))
    [request] = received
    assert request["path"] == "/api/chat" and request["body"]["format"] == PRACTICE_SPEC.output_schema
    assert request["body"]["stream"] is True and request["body"]["messages"][0]["content"] == PRACTICE_SPEC.system_prompt
    assert result.usage["input_tokens"] == 1234 and result.usage["output_tokens"] == 567
    assert len(result.data.practices) == len(si.GOOD_PRACTICES["practices"])


def test_real_http_transport_reports_a_missing_model(ollama_stub):
    url, _ = ollama_stub
    settings = LLMSettings(model="absent:1b", ollama_url=url)
    assert not check_runtime(settings)["model_installed"]
    with pytest.raises(LLMError) as info:
        complete(LocalAgentRunner(settings))
    assert info.value.code == "MODEL_NOT_FOUND"


def test_real_http_transport_reports_ollama_stopped(loopback_only):
    with socket.socket() as sock:  # port libre : aucun serveur n'y écoute
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = LLMSettings(ollama_url=f"http://127.0.0.1:{port}")
    status = check_runtime(settings)
    assert not status["available"] and status["error"]["code"] == "OLLAMA_UNAVAILABLE"
    with pytest.raises(LLMError) as info:
        complete(LocalAgentRunner(settings))
    assert info.value.code == "OLLAMA_UNAVAILABLE"


def test_tests_cannot_reach_a_real_ollama():
    """Sans faux Ollama, le garde-fou de conftest interdit même la boucle locale : l'état est « indisponible »."""
    status = check_runtime(LLMSettings())
    assert not status["available"] and status["error"]["code"] == "OLLAMA_UNAVAILABLE"
