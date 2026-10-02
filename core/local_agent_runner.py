"""Exécution LOCALE des agents de TRACE : Ollama sur cet ordinateur, rien d'autre.

    orchestrateur d'une étape (core/analysis.py, accountability.py, trajectory.py, cross_interview.py)
      → LocalAgentRunner.complete_json(prompt système de l'agent, matériau exact, JSON Schema, libellé)
      → Ollama local (POST /api/chat, `format` = JSON Schema strict de l'agent)
      → JSON → Pydantic (schéma de l'agent) → validateur méthodologique de l'étape (core/agent_checks.py)
      → réponse rendue à l'orchestrateur, qui applique ensuite ses validateurs habituels.

Règles :
- adresse d'Ollama limitée à la boucle locale (localhost, 127.0.0.1, ::1) : aucune donnée ne quitte l'ordinateur ;
  les proxys HTTP de l'environnement ne sont jamais utilisés ;
- aucun fournisseur externe, aucune clé, aucun repli : Ollama absent, modèle absent ou délai dépassé donnent une
  erreur claire (`OLLAMA_UNAVAILABLE`, `MODEL_NOT_FOUND`, `OLLAMA_TIMEOUT`), jamais un autre modèle ;
- une réponse non conforme (JSON invalide, schéma, anomalie bloquante du validateur de l'étape) donne lieu à au plus
  `max_corrections` nouvelles tentatives LOCALES : le modèle reçoit sa réponse et la liste des erreurs, et renvoie
  l'objet complet corrigé. Le validateur n'est jamais contourné : une réponse conforme au schéma qui garde une
  anomalie après les corrections est rendue telle quelle à l'orchestrateur, dont les validateurs la signalent
  (needs_review, rejet) ; une réponse qui ne respecte toujours pas le schéma est une erreur ;
- chaque appel est journalisé (tentatives, erreurs, réponse finale) dans le dossier du run ou du corpus.

Les prompts du dépôt (prompts/*.md) restent la source de vérité de chaque agent : ils sont envoyés tels quels.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

from agents.base import AgentSpec, canonical_json, sha256_text
from core.analysis_cache import write_json_atomic
from core.llm_client import (RUNTIME_ERROR_CODES, LLMError, LLMResult, LLMSettings, parse_json_output,
                             usage_to_dict)

RUNTIME_NAME = "Ollama"
EXECUTION = "locale"
EXTERNAL_DATA = "aucune"
START_COMMAND = "ollama serve"
JOURNAL_VERSION = "1.0"

CORRECTION_INSTRUCTIONS = (
    "Corrige ta réponse en respectant tes consignes et le schéma JSON : renvoie l'objet JSON COMPLET corrigé, sans "
    "texte autour. Ne modifie pas la transcription : une citation se recopie mot pour mot depuis le tour cité, et un "
    "objet qui n'est pas appuyé par le matériau est retiré. N'utilise que les identifiants présents dans le matériau."
)


def install_command(model: str) -> str:
    return f"ollama pull {model}"


# --- Adresse locale uniquement ---------------------------------------------------------------------

def ensure_local_url(url: str) -> str:
    """Adresse d'Ollama, seulement si elle désigne CET ordinateur (boucle locale). Sinon NON_LOCAL_URL."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    local = host == "localhost"
    if not local and host:
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
    if parsed.scheme not in ("http", "https") or not local:
        raise LLMError("NON_LOCAL_URL", url)
    return url.rstrip("/")


# --- Transport HTTP vers Ollama (bibliothèque standard, sans proxy) -----------------------------------

class OllamaTransport:
    """Appels HTTP à l'API locale d'Ollama : /api/version, /api/tags, /api/chat."""

    def __init__(self, base_url: str, timeout_seconds: float = 1800):
        self.base_url = ensure_local_url(base_url)
        self.timeout_seconds = timeout_seconds
        # Aucun proxy : une requête vers la boucle locale ne doit jamais transiter par un intermédiaire.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=timeout or self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(text).get("error") or text
            except (ValueError, AttributeError):
                message = text
            if exc.code == 404 and "not found" in str(message).lower():
                raise LLMError("MODEL_NOT_FOUND", str(body.get("model")) if body else None) from None
            raise LLMError("OLLAMA_ERROR", f"HTTP {exc.code} : {str(message)[:300]}") from None
        except (TimeoutError, socket.timeout):
            raise LLMError("OLLAMA_TIMEOUT", f"{timeout or self.timeout_seconds} s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMError("OLLAMA_TIMEOUT", f"{timeout or self.timeout_seconds} s") from None
            raise LLMError("OLLAMA_UNAVAILABLE", f"{self.base_url} : {exc.reason}") from None
        except (ConnectionError, OSError) as exc:
            raise LLMError("OLLAMA_UNAVAILABLE", f"{self.base_url} : {type(exc).__name__}") from None
        except ValueError:
            raise LLMError("OLLAMA_ERROR", "réponse illisible") from None

    def version(self) -> str:
        return str(self._request("GET", "/api/version", timeout=5).get("version") or "?")

    def models(self) -> list[str]:
        return [m.get("name") or m.get("model") for m in self._request("GET", "/api/tags", timeout=10).get("models", [])]

    def chat(self, payload: dict) -> dict:
        return self._request("POST", "/api/chat", payload)


def model_installed(model: str, installed: list[str]) -> bool:
    """« qwen2.5:14b » ou « mistral » (étiquette « latest » implicite)."""
    return model in installed or (":" not in model and f"{model}:latest" in installed)


def check_runtime(settings: LLMSettings, transport=None) -> dict:
    """État du runtime local, pour l'interface et la CLI : Ollama joignable ? modèle installé ? Ne lève pas."""
    status = {"execution": EXECUTION, "runtime": RUNTIME_NAME, "url": settings.ollama_url, "model": settings.model,
              "external_data": EXTERNAL_DATA, "available": False, "version": None, "models": [],
              "model_installed": False, "ready": False, "error": None,
              "install_command": install_command(settings.model), "start_command": START_COMMAND}
    try:
        transport = transport or OllamaTransport(settings.ollama_url, settings.timeout_seconds)
        status["version"] = transport.version()
        status["available"] = True
        status["models"] = transport.models()
    except LLMError as error:
        status["error"] = error.to_dict()
        return status
    except Exception as exc:  # noqa: BLE001 — l'état du runtime ne doit jamais faire échouer l'interface
        status["error"] = LLMError("OLLAMA_UNAVAILABLE", f"{settings.ollama_url} : {type(exc).__name__}").to_dict()
        return status
    status["model_installed"] = model_installed(settings.model, status["models"])
    status["ready"] = status["model_installed"]
    if not status["ready"]:
        status["error"] = LLMError("MODEL_NOT_FOUND", f"{settings.model} — installez-le : "
                                                     f"{install_command(settings.model)}").to_dict()
    return status


def require_runtime(settings: LLMSettings, transport=None) -> dict:
    """Comme `check_runtime`, mais lève l'erreur claire si Ollama ou le modèle manquent (aucun repli)."""
    status = check_runtime(settings, transport)
    if not status["available"]:
        code = status["error"]["code"]
        detail = settings.ollama_url + ("" if code == "NON_LOCAL_URL" else f" — démarrage : {START_COMMAND}")
        raise LLMError(code, detail)
    if not status["model_installed"]:
        raise LLMError("MODEL_NOT_FOUND", f"{settings.model} — installez-le : {install_command(settings.model)}")
    return status


# --- Runner ------------------------------------------------------------------------------------------

def call_id_for(label: str, system_prompt: str, user_content: str, output_schema: dict) -> str:
    """Identifiant lisible et stable d'un appel : libellé + empreinte de la requête exacte."""
    digest = sha256_text(canonical_json({
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_content_sha256": sha256_text(user_content),
        "schema_sha256": sha256_text(canonical_json(output_schema)),
    }))
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label.replace("/", "__")).strip("-")
    return f"{slug}__{digest[:10]}"


def correction_message(problems: list[str]) -> str:
    lines = "\n".join(f"- {p}" for p in problems[:30])
    more = f"\n- … (+{len(problems) - 30} autre(s))" if len(problems) > 30 else ""
    return ("Ta réponse précédente ne peut pas être acceptée par TRACE :\n" + lines + more + "\n\n"
            + CORRECTION_INSTRUCTIONS)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class LocalAgentRunner:
    """Client des agents des étapes 3 à 6 (`complete_json`, `aclose`), injecté dans les orchestrateurs.

    `transport` : objet avec `chat(payload) -> dict`, `version()`, `models()` (Ollama local par défaut ; un faux
    transport dans les tests). `checker(spec, label, user_content, output) -> list[str]` : anomalies BLOQUANTES du
    validateur de l'étape (core/agent_checks.MethodChecker), renvoyées au modèle pour correction. `journal_dir` :
    un fichier JSON par appel. `specs` : agents connus (pour le journal et le contrôle).
    """

    def __init__(self, settings: LLMSettings, *, transport=None, checker: Callable | None = None,
                 journal_dir: Path | None = None, specs: tuple[AgentSpec, ...] = ()):
        self.settings = settings
        self.transport = transport or OllamaTransport(settings.ollama_url, settings.timeout_seconds)
        self.checker = checker
        self.journal_dir = Path(journal_dir) if journal_dir else None
        self.specs = {spec.system_prompt: spec for spec in specs}
        self.records: dict[str, dict] = {}
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    def check(self) -> dict:
        return check_runtime(self.settings, self.transport)

    def require_ready(self) -> dict:
        return require_runtime(self.settings, self.transport)

    async def aclose(self) -> None:
        return None

    def _semaphore(self) -> asyncio.Semaphore:
        loop = id(asyncio.get_running_loop())
        if loop not in self._semaphores:
            self._semaphores[loop] = asyncio.Semaphore(self.settings.concurrency)
        return self._semaphores[loop]

    async def _chat(self, messages: list[dict], output_schema: dict) -> dict:
        payload = {"model": self.settings.model, "messages": messages, "format": output_schema, "stream": False,
                   "options": {"temperature": self.settings.temperature, "num_ctx": self.settings.num_ctx}}
        async with self._semaphore():
            return await asyncio.to_thread(self.transport.chat, payload)

    async def complete_json(self, *, system_prompt: str, user_content: str, output_schema: dict,
                            response_model, label: str = "") -> LLMResult:
        spec = self.specs.get(system_prompt)
        call_id = call_id_for(label, system_prompt, user_content, output_schema)
        record = {"call_id": call_id, "label": label, "interview_id": label.split("/", 1)[0],
                  "agent": spec.name if spec else None, "agent_label": spec.label if spec else None,
                  "model": self.settings.model, "attempts": 0, "outcome": None, "error": None,
                  "remaining_issues": []}
        self.records[call_id] = record
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        attempts, started = [], time.monotonic()
        tokens = {"input_tokens": 0, "output_tokens": 0}
        last_valid = last_valid_reply = last_error = None
        reply: dict = {}
        text = last_valid_text = ""
        try:
            for number in range(1, self.settings.max_corrections + 2):
                reply = await self._chat(messages, output_schema)  # erreur du runtime : levée, jamais de repli
                text = (reply.get("message") or {}).get("content") or ""
                tokens["input_tokens"] += int(reply.get("prompt_eval_count") or 0)
                tokens["output_tokens"] += int(reply.get("eval_count") or 0)
                attempt = {"attempt": number, "done_reason": reply.get("done_reason"),
                           "prompt_eval_count": reply.get("prompt_eval_count"), "eval_count": reply.get("eval_count")}
                if reply.get("done_reason") == "length":
                    last_error = LLMError("TRUNCATED")
                    attempts.append({**attempt, "status": "invalid", "problems": [last_error.user_message]})
                else:
                    try:
                        data = parse_json_output(text, response_model, max_problems=20)
                    except LLMError as error:
                        last_error = error
                        attempts.append({**attempt, "status": "invalid",
                                         "problems": [f"{error.code} — {error.user_message}"]})
                    else:
                        problems = (self.checker(spec, label, user_content, data.model_dump(mode="json"))
                                    if self.checker and spec else [])
                        last_valid, last_valid_reply, last_valid_text = data, reply, text
                        attempts.append({**attempt, "status": "blocking" if problems else "accepted",
                                         "problems": problems})
                        if not problems:
                            record.update(outcome="accepted")
                            return self._result(data, record, attempts, started, tokens, reply, text)
                if number <= self.settings.max_corrections:  # nouvelle tentative LOCALE, erreurs à l'appui
                    messages += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": correction_message(attempts[-1]["problems"])}]
            if last_valid is not None:
                # Validateur jamais contourné : la dernière réponse conforme au schéma garde ses anomalies, que les
                # validateurs de l'étape signaleront (needs_review, rejet) ; elles sont notées dans le journal.
                remaining = next(a["problems"] for a in reversed(attempts) if a["status"] == "blocking")
                record.update(outcome="accepted_with_issues", remaining_issues=remaining)
                return self._result(last_valid, record, attempts, started, tokens, last_valid_reply,
                                    last_valid_text)
            raise last_error or LLMError("SCHEMA_VALIDATION")
        except LLMError as error:
            error.request_id = error.request_id or call_id
            error.duration_seconds = round(time.monotonic() - started, 3)
            error.usage = usage_to_dict(tokens) if tokens["input_tokens"] or tokens["output_tokens"] else None
            record.update(outcome="failed", error=error.user_message)
            self._journal(record, attempts, started, error=error.to_dict(), text=text)
            raise
        finally:
            record["attempts"] = len(attempts)

    def _result(self, data, record: dict, attempts: list, started: float, tokens: dict, reply: dict,
                text: str) -> LLMResult:
        record["attempts"] = len(attempts)
        duration = round(time.monotonic() - started, 3)
        self._journal(record, attempts, started, text=text)
        return LLMResult(data=data, usage=usage_to_dict(tokens), attempts=0, duration_seconds=duration,
                         model_requested=self.settings.model, response_model=reply.get("model") or self.settings.model,
                         request_id=record["call_id"], stop_reason=reply.get("done_reason"))

    def _journal(self, record: dict, attempts: list, started: float, *, error: dict | None = None,
                 text: str = "") -> None:
        if self.journal_dir is None:
            return
        write_json_atomic(self.journal_dir / f"{record['call_id']}.json", {
            "journal_version": JOURNAL_VERSION, **record, "execution": EXECUTION, "runtime": RUNTIME_NAME,
            "ollama_url": self.settings.ollama_url, "options": {"temperature": self.settings.temperature,
                                                                "num_ctx": self.settings.num_ctx},
            "max_corrections": self.settings.max_corrections, "finished_at": _now(),
            "duration_seconds": round(time.monotonic() - started, 3), "attempt_log": attempts, "error": error,
            "final_response": text,
        })


def runtime_error_code(error: dict | None) -> bool:
    """Erreur du runtime (Ollama absent, modèle absent…) plutôt que d'une réponse ?"""
    return bool(error) and error.get("code") in RUNTIME_ERROR_CODES
