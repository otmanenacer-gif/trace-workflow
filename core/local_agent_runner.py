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
- étape 3 (Practice Extractor, Interaction Reader, longue distance) : une réponse lisible n'est jamais régénérée en
  entier ; citations corrigées sans modèle quand c'est sûr (core/citation_resolver.py), au plus UNE réparation par le
  modèle et par objet pour une autre anomalie locale (core/stage3_repair.py) ;
- sinon, une réponse non conforme donne lieu à de nouvelles tentatives LOCALES : JSON invalide ou hors schéma, au plus
  `max_corrections` ; anomalie bloquante du validateur de l'étape, au plus `max_method_corrections`. Le modèle reçoit
  sa réponse et la liste des erreurs, et renvoie l'objet complet corrigé. Une réponse tronquée (réponse maximale
  atteinte) est redemandée en entier, avec une réserve doublée, sans être renvoyée au modèle. Le validateur n'est jamais contourné : une réponse conforme au schéma qui garde une
  anomalie après les corrections est rendue telle quelle à l'orchestrateur, dont les validateurs la signalent
  (needs_review, rejet) ; une réponse qui ne respecte toujours pas le schéma est une erreur ;
- chaque appel reçoit la plus petite fenêtre de contexte qui contient sa requête et sa réponse (jamais au-delà de
  TRACE_OLLAMA_NUM_CTX ; une requête plus longue est refusée, jamais tronquée), une réponse bornée (num_predict), et
  le modèle reste chargé entre deux appels (keep_alive) ;
- chaque appel est journalisé (tentatives et leurs mesures, erreurs, réponse finale) dans le dossier du run ou du
  corpus, et résumé en une ligne dans le journal Python « trace.local ».

Les prompts du dépôt (prompts/*.md) restent la source de vérité de chaque agent : ils sont envoyés tels quels.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from agents.base import AgentSpec, canonical_json, sha256_text
from core import stage3_repair
from core.analysis_cache import write_json_atomic
from core.llm_client import (CTX_BUCKETS, RUNTIME_ERROR_CODES, LLMError, LLMResult, LLMSettings, parse_json_output,
                             usage_to_dict)

RUNTIME_NAME = "Ollama"
EXECUTION = "locale"
EXTERNAL_DATA = "aucune"
START_COMMAND = "ollama serve"
JOURNAL_VERSION = "1.1"

logger = logging.getLogger("trace.local")

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
    """Appels HTTP à l'API locale d'Ollama : /api/version, /api/tags, /api/chat (réponse d'un bloc ou en flux)."""

    def __init__(self, base_url: str, timeout_seconds: float = 1800):
        self.base_url = ensure_local_url(base_url)
        self.timeout_seconds = timeout_seconds
        # Aucun proxy : une requête vers la boucle locale ne doit jamais transiter par un intermédiaire.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @contextlib.contextmanager
    def _errors(self, body: dict | None, timeout: float):
        """Erreurs HTTP / réseau d'Ollama → LLMError claire (jamais de repli)."""
        try:
            yield
        except LLMError:
            raise
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
            raise LLMError("OLLAMA_TIMEOUT", f"{timeout} s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMError("OLLAMA_TIMEOUT", f"{timeout} s") from None
            raise LLMError("OLLAMA_UNAVAILABLE", f"{self.base_url} : {exc.reason}") from None
        except (ConnectionError, OSError) as exc:
            raise LLMError("OLLAMA_UNAVAILABLE", f"{self.base_url} : {type(exc).__name__}") from None
        except ValueError:
            raise LLMError("OLLAMA_ERROR", "réponse illisible") from None

    def _open(self, method: str, path: str, body: dict | None, timeout: float):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        return self._opener.open(request, timeout=timeout)

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        timeout = timeout or self.timeout_seconds
        with self._errors(body, timeout), self._open(method, path, body, timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    def version(self) -> str:
        return str(self._request("GET", "/api/version", timeout=5).get("version") or "?")

    def models(self) -> list[str]:
        return [m.get("name") or m.get("model") for m in self._request("GET", "/api/tags", timeout=10).get("models", [])]

    def chat(self, payload: dict, on_progress: Callable[[int], None] | None = None) -> dict:
        """/api/chat. En flux (`stream: true`), `on_progress(tokens générés)` est appelé au fil de la génération
        (au plus une fois par seconde) ; la réponse rendue a la même forme qu'une réponse d'un bloc."""
        if not payload.get("stream"):
            return self._request("POST", "/api/chat", payload)
        deadline = time.monotonic() + self.timeout_seconds
        parts: list[str] = []
        final: dict = {}
        produced, last_report = 0, 0.0
        # délai entre deux fragments : celui du chargement du modèle et de la lecture du prompt (borné par le total)
        with self._errors(payload, self.timeout_seconds), \
                self._open("POST", "/api/chat", payload, self.timeout_seconds) as response:
            for raw in response:
                if not raw.strip():
                    continue
                chunk = json.loads(raw.decode("utf-8"))
                if chunk.get("error"):
                    message = str(chunk["error"])
                    if "not found" in message.lower():
                        raise LLMError("MODEL_NOT_FOUND", str(payload.get("model")))
                    raise LLMError("OLLAMA_ERROR", message[:300])
                parts.append((chunk.get("message") or {}).get("content") or "")
                produced += 1
                if chunk.get("done"):
                    final = chunk
                    break
                now = time.monotonic()
                if now > deadline:
                    raise LLMError("OLLAMA_TIMEOUT", f"{self.timeout_seconds} s")
                if on_progress and now - last_report >= 1.0:
                    last_report = now
                    on_progress(produced)
        if not final:
            raise LLMError("OLLAMA_ERROR", "flux interrompu avant la fin de la réponse")
        return {**final, "message": {"role": "assistant", "content": "".join(parts)}}


def model_installed(model: str, installed: list[str]) -> bool:
    """« qwen2.5:7b » ou « mistral » (étiquette « latest » implicite)."""
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


# --- Fenêtre de contexte dimensionnée par appel --------------------------------------------------------

# Estimation PRUDENTE des tokens d'un prompt (français, identifiants de tours, JSON) : 3 caractères par token, soit
# plus que l'estimation des blocs (3,5) ; elle sert à ne JAMAIS envoyer un prompt plus long que la fenêtre (Ollama le
# tronquerait sans erreur). Corrigée à la hausse si Ollama compte davantage de tokens qu'estimé.
CTX_CHARS_PER_TOKEN = 3.0
CTX_MESSAGE_OVERHEAD = 8      # balises du gabarit de conversation, par message
CTX_MARGIN = 512              # marge de sécurité de la fenêtre
MIN_OUTPUT_TOKENS = 1024      # réponse minimale : en deçà, la requête est refusée (PROMPT_TOO_LONG)
# Réponse maximale réservée par agent (num_predict) : borne une génération qui s'emballe et dimensionne la fenêtre.
# Une réponse qui l'atteint (done_reason « length ») est redemandée avec une réserve doublée (sans être renvoyée
# au modèle), jamais acceptée tronquée.
# Ordres de grandeur mesurés sur les réponses de référence : une pratique ≈ 1 000 caractères JSON (23 champs requis,
# ≈ 330 tokens), un signal ≈ 460 caractères (10 champs, ≈ 150 tokens).
OUTPUT_RESERVE = {"speaker_attribution_auditor": 2048, "practice_extractor": 8192,
                  "interaction_signal_reader": 6144, "interaction_signal_reader_long_distance": 3072,
                  "accountability_episode_builder": 6144, "episode_grounding_checker": 3072,
                  "trajectory_mapper": 8192,
                  "cross_interview_comparator": 8192, "theory_block_analyst": 4096, "theory_synthesizer": 6144,
                  "report_section_writer": 4096, "report_validator": 3072}
DEFAULT_OUTPUT_RESERVE = 4096


def estimate_prompt_tokens(messages: list[dict], chars_per_token: float = CTX_CHARS_PER_TOKEN) -> int:
    return sum(math.ceil(len(m["content"]) / chars_per_token) + CTX_MESSAGE_OVERHEAD for m in messages)


def context_plan(prompt_tokens: int, reserve: int, maximum: int, floor: int = 0) -> tuple[int, int]:
    """(num_ctx, num_predict) : la plus petite fenêtre de CTX_BUCKETS qui contient le prompt, la réponse réservée et
    la marge (au moins `floor`, pour ne pas recharger le modèle à chaque appel), jamais au-delà de `maximum`.
    PROMPT_TOO_LONG si le prompt et une réponse minimale ne tiennent pas dans `maximum` : rien n'est tronqué."""
    if prompt_tokens + MIN_OUTPUT_TOKENS + CTX_MARGIN > maximum:
        raise LLMError("PROMPT_TOO_LONG", f"~{prompt_tokens} tokens estimés, fenêtre maximale {maximum}")
    need = max(prompt_tokens + reserve + CTX_MARGIN, min(floor, maximum))
    num_ctx = next((b for b in CTX_BUCKETS if need <= b <= maximum), maximum)
    return num_ctx, min(reserve, num_ctx - prompt_tokens - CTX_MARGIN)


def _seconds(nanoseconds) -> float | None:
    return round(nanoseconds / 1e9, 2) if nanoseconds else None


def _rate(tokens, nanoseconds) -> float | None:
    return round(tokens / (nanoseconds / 1e9), 1) if tokens and nanoseconds else None


def _thousands(value) -> str:
    return f"{value:,}".replace(",", "\u202f") if isinstance(value, int) else "?"


def call_summary_line(record: dict) -> str:
    """« Practice Extractor — OTMANE/bloc2 — 2 900 tokens entrée — 620 sortie — 41 s — 15 tok/s — … »."""
    label = record.get("agent_label") or record.get("agent") or "agent"
    where = record["label"].split("/", 1)[0] + (f"/{record['label'].rsplit('/', 1)[1]}"
                                                 if record["label"].count("/") >= 2 else "")
    reasons = ", ".join(record.get("correction_reasons") or [])
    return (f"{label} — {where} — {_thousands(record.get('input_tokens'))} tokens entrée — "
            f"{_thousands(record.get('output_tokens'))} sortie — {record.get('duration_seconds')} s — "
            f"{record.get('tokens_per_second') or '?'} tok/s — contexte {record.get('num_ctx')} — "
            f"{record.get('attempts')} tentative(s)" + (f" (corrections : {reasons})" if reasons else "")
            + f" — {record.get('outcome')}")


class PrefixAwareSlots:
    """Accès au modèle local (au plus `size` appels simultanés). Quand un appel se termine, la place libérée va en
    priorité à un appel en attente du MÊME agent (même prompt système) : Ollama réutilise alors le préfixe déjà lu
    (prompt système de 4 000 à 5 500 tokens) au lieu de le relire à chaque alternance Practice / Interaction. L'ordre
    d'exécution ne change rien aux résultats (chaque appel est indépendant ; fusion déterministe par l'orchestrateur)."""

    def __init__(self, size: int):
        self.size = size
        self.busy = 0
        self.last_key: str | None = None
        self.waiters: list[tuple[str, asyncio.Future]] = []

    async def acquire(self, key: str) -> None:
        if self.busy < self.size and not self.waiters:
            self.busy += 1
            self.last_key = key
            return
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((key, future))
        await future

    def release(self) -> None:
        self.busy -= 1
        while self.waiters and self.busy < self.size:
            index = next((i for i, (k, f) in enumerate(self.waiters) if k == self.last_key and not f.cancelled()), 0)
            key, future = self.waiters.pop(index)
            if future.cancelled():
                continue
            self.busy += 1
            self.last_key = key
            future.set_result(None)

    @contextlib.asynccontextmanager
    async def slot(self, key: str):
        await self.acquire(key)
        try:
            yield
        finally:
            self.release()


class LocalAgentRunner:
    """Client des agents des étapes 3 à 6 (`complete_json`, `aclose`), injecté dans les orchestrateurs.

    `transport` : objet avec `chat(payload[, on_progress]) -> dict`, `version()`, `models()` (Ollama local par
    défaut ; un faux transport dans les tests). `checker(spec, label, user_content, output) -> list[str]` : anomalies
    BLOQUANTES du validateur de l'étape (core/agent_checks.MethodChecker), renvoyées au modèle pour correction.
    `journal_dir` : un fichier JSON par appel. `specs` : agents connus (pour le journal et le contrôle).
    `on_event(event)` : progression (début d'appel, tokens générés, fin d'appel), pour l'exécution en arrière-plan.

    Mesures de chaque tentative (journal, `records`, journal Python « trace.local ») : taille du prompt (caractères,
    tokens estimés), fenêtre de contexte et réponse maximale demandées, début, fin, durée, tokens lus et générés
    (comptés par Ollama), vitesses, chargement du modèle, taille de la réponse, motif de la correction.
    """

    item_repair = True  # étape 3 : réparation ciblée des seuls objets fautifs (core/stage3_repair.py)

    def __init__(self, settings: LLMSettings, *, transport=None, checker: Callable | None = None,
                 journal_dir: Path | None = None, specs: tuple[AgentSpec, ...] = (),
                 on_event: Callable[[dict], None] | None = None):
        self.settings = settings
        self.transport = transport or OllamaTransport(settings.ollama_url, settings.timeout_seconds)
        self.checker = checker
        self.journal_dir = Path(journal_dir) if journal_dir else None
        self.specs = {spec.system_prompt: spec for spec in specs}
        self.on_event = on_event
        self.records: dict[str, dict] = {}
        self._semaphores: dict[int, PrefixAwareSlots] = {}
        self._ctx_floor = 0                       # fenêtre déjà chargée : jamais réduite (pas de rechargement)
        self._chars_per_token = CTX_CHARS_PER_TOKEN

    def check(self) -> dict:
        return check_runtime(self.settings, self.transport)

    def require_ready(self) -> dict:
        return require_runtime(self.settings, self.transport)

    async def aclose(self) -> None:
        return None

    def _semaphore(self) -> PrefixAwareSlots:
        loop = id(asyncio.get_running_loop())
        if loop not in self._semaphores:
            self._semaphores[loop] = PrefixAwareSlots(self.settings.concurrency)
        return self._semaphores[loop]

    def _emit(self, event: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001 — la progression ne doit jamais interrompre un appel
                logger.exception("Progression : événement non transmis")

    async def _chat(self, messages: list[dict], output_schema: dict, prepare: Callable[[], tuple[int, int]],
                    on_progress: Callable[[int], None] | None = None) -> dict:
        """Un appel à Ollama, une fois l'accès au modèle obtenu. `prepare()` (appelé à ce moment-là : fenêtre
        choisie d'après la fenêtre déjà chargée, chronomètre et événement de début) rend (num_ctx, num_predict)."""
        async with self._semaphore().slot(sha256_text(messages[0]["content"])):
            num_ctx, num_predict = prepare()
            payload = {"model": self.settings.model, "messages": messages, "format": output_schema, "stream": True,
                       "keep_alive": self.settings.keep_alive,
                       "options": {"temperature": self.settings.temperature, "num_ctx": num_ctx,
                                   "num_predict": num_predict}}
            return await asyncio.to_thread(self.transport.chat, payload, on_progress)

    async def complete_json(self, *, system_prompt: str, user_content: str, output_schema: dict,
                            response_model, label: str = "") -> LLMResult:
        spec = self.specs.get(system_prompt)
        agent = spec.name if spec else None
        call_id = call_id_for(label, system_prompt, user_content, output_schema)
        base_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        record = {"call_id": call_id, "label": label, "interview_id": label.split("/", 1)[0],
                  "agent": agent, "agent_label": spec.label if spec else None,
                  "model": self.settings.model, "attempts": 0, "outcome": None, "error": None,
                  "remaining_issues": [], "started_at": None, "finished_at": None,
                  "prompt_chars": sum(len(m["content"]) for m in base_messages),
                  "schema_chars": len(canonical_json(output_schema)),
                  "estimated_input_tokens": estimate_prompt_tokens(base_messages, self._chars_per_token),
                  "num_ctx": None, "num_predict": None, "input_tokens": 0, "output_tokens": 0,
                  "duration_seconds": None, "generation_seconds": 0.0, "tokens_per_second": None,
                  "load_seconds": 0.0, "response_chars": None, "correction_reasons": []}
        self.records[call_id] = record
        messages = list(base_messages)
        reserve = OUTPUT_RESERVE.get(agent, DEFAULT_OUTPUT_RESERVE)
        attempts, started = [], time.monotonic()
        tokens = {"input_tokens": 0, "output_tokens": 0}
        last_valid = last_valid_reply = last_error = None
        reply: dict = {}
        text = last_valid_text = ""
        budget = {"format": self.settings.max_corrections, "method": self.settings.max_method_corrections}
        repair = (self.item_repair and spec is not None and hasattr(self.checker, "repairable")
                  and self.checker.repairable(spec, label))
        state_file = stage3_repair.state_path(self.journal_dir, call_id) if repair else None
        try:
            state = stage3_repair.load_state(state_file, call_id, self.settings.model) if repair else None
            if state is not None:  # reprise : génération initiale et réparations terminées conservées
                logger.info("%s — %s : reprise des réparations (%s objet(s) à traiter)", record["agent_label"], label,
                            sum(i["status"] == stage3_repair.PENDING for i in state["items"]))
                return await self._repair_items(spec, label, system_prompt, user_content, response_model, None,
                                                record, attempts, started, tokens, {}, state_file, state=state)
            number = 0
            while True:
                number += 1
                reply, text, attempt = await self._generate(messages, output_schema, reserve, record, tokens,
                                                            number=number)
                kind = None
                if reply.get("done_reason") == "length":
                    last_error = LLMError("TRUNCATED")
                    kind, problems = "truncated", [last_error.user_message]
                    attempts.append({**attempt, "status": "invalid", "problems": problems})
                elif repair and (split := stage3_repair.split_response(text, spec, response_model)) is not None:
                    # réponse lisible : seuls les objets fautifs seront redemandés (jamais toute la réponse)
                    attempts.append({**attempt, "status": "split", "problems": [], "phase": "initial"})
                    self._log_attempt(record, attempts[-1], None)
                    return await self._repair_items(spec, label, system_prompt, user_content, response_model, split,
                                                    record, attempts, started, tokens, reply, state_file)
                else:
                    try:
                        data = parse_json_output(text, response_model, max_problems=20)
                    except LLMError as error:
                        last_error = error
                        kind, problems = "format", [f"{error.code} — {error.user_message}"]
                        attempts.append({**attempt, "status": "invalid", "problems": problems})
                    else:
                        problems = (self.checker(spec, label, user_content, data.model_dump(mode="json"))
                                    if self.checker and spec else [])
                        last_valid, last_valid_reply, last_valid_text = data, reply, text
                        attempts.append({**attempt, "status": "blocking" if problems else "accepted",
                                         "problems": problems})
                        if not problems:
                            record.update(outcome="accepted")
                            return self._result(data, record, attempts, started, tokens, reply, text)
                        kind = "method"
                self._log_attempt(record, attempts[-1], kind)
                budget_key = "method" if kind == "method" else "format"
                if budget[budget_key] <= 0:
                    break
                budget[budget_key] -= 1
                reason = {"truncated": "réponse tronquée", "format": attempts[-1]["problems"][0].split(" — ")[0],
                          "method": "validateur méthodologique"}[kind]
                attempts[-1]["correction_reason"] = reason
                record["correction_reasons"].append(reason)
                if kind == "truncated":
                    # Réponse coupée : redemandée en entier avec une réserve doublée, sans la renvoyer au modèle.
                    reserve = min(reserve * 2, self.settings.num_ctx)
                    messages = list(base_messages)
                else:  # nouvelle tentative LOCALE, erreurs à l'appui ; le modèle renvoie l'objet complet corrigé
                    messages = messages + [{"role": "assistant", "content": text},
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
            error.duration_seconds = round(sum(a.get("duration_seconds") or 0 for a in attempts), 3)
            error.usage = usage_to_dict(tokens) if tokens["input_tokens"] or tokens["output_tokens"] else None
            record.update(outcome="failed", error=error.user_message)
            self._finish(record, attempts, started, tokens, text)
            self._journal(record, attempts, started, error=error.to_dict(), text=text)
            raise
        finally:
            record["attempts"] = len(attempts)

    async def _generate(self, messages: list[dict], output_schema: dict, reserve: int, record: dict, tokens: dict, *,
                        number: int, phase: str = "initial", item_index: int | None = None) -> tuple[dict, str, dict]:
        """UNE génération (réponse complète ou réparation d'un objet), mesurée. → (réponse d'Ollama, texte, mesures)."""
        label, agent = record["label"], record["agent"]
        attempt: dict = {"attempt": number, "phase": phase}
        if item_index is not None:
            attempt["item_index"] = item_index

        def prepare() -> tuple[int, int]:
            # au moment où l'appel obtient le modèle : la fenêtre tient compte de celle déjà chargée
            prompt_tokens = estimate_prompt_tokens(messages, self._chars_per_token)
            num_ctx, num_predict = context_plan(prompt_tokens, reserve, self.settings.num_ctx, self._ctx_floor)
            self._ctx_floor = max(self._ctx_floor, num_ctx)
            attempt.update(started_at=_now(), prompt_chars=sum(len(m["content"]) for m in messages),
                           estimated_input_tokens=prompt_tokens, num_ctx=num_ctx, num_predict=num_predict,
                           _clock=time.monotonic())
            record.update(num_ctx=num_ctx, started_at=record["started_at"] or attempt["started_at"])
            if phase != "repair":
                record["num_predict"] = num_predict
            self._emit({"type": "call_started", "label": label, "agent": agent, "agent_label": record["agent_label"],
                        "attempt": number, "phase": phase, "item_index": item_index, "num_ctx": num_ctx,
                        "estimated_input_tokens": prompt_tokens, "at": time.time()})
            return num_ctx, num_predict

        def progress(produced: int) -> None:
            self._emit({"type": "tokens", "label": label, "attempt": number, "output_tokens": produced,
                        "at": time.time()})

        reply = await self._chat(messages, output_schema, prepare, progress)
        attempt_started = attempt.pop("_clock")
        text = (reply.get("message") or {}).get("content") or ""
        prompt_eval, evals = int(reply.get("prompt_eval_count") or 0), int(reply.get("eval_count") or 0)
        tokens["input_tokens"] += prompt_eval
        tokens["output_tokens"] += evals
        attempt.update(finished_at=_now(), duration_seconds=round(time.monotonic() - attempt_started, 2),
                       done_reason=reply.get("done_reason"), prompt_eval_count=reply.get("prompt_eval_count"),
                       prompt_eval_seconds=_seconds(reply.get("prompt_eval_duration")),
                       prompt_tokens_per_second=_rate(prompt_eval, reply.get("prompt_eval_duration")),
                       eval_count=reply.get("eval_count"), eval_seconds=_seconds(reply.get("eval_duration")),
                       tokens_per_second=_rate(evals, reply.get("eval_duration")),
                       load_seconds=_seconds(reply.get("load_duration")), response_chars=len(text))
        record["generation_seconds"] = round(record["generation_seconds"] + (attempt["eval_seconds"] or 0), 2)
        record["load_seconds"] = round(record["load_seconds"] + (attempt["load_seconds"] or 0), 2)
        if prompt_eval > attempt["estimated_input_tokens"]:  # Ollama compte plus qu'estimé : estimation durcie
            self._chars_per_token = min(self._chars_per_token, attempt["prompt_chars"] / prompt_eval * 0.95)
        return reply, text, attempt

    async def _repair_items(self, spec: AgentSpec, label: str, system_prompt: str, user_content: str,
                            response_model, split: dict | None, record: dict, attempts: list, started: float,
                            tokens: dict, reply: dict, state_file, state: dict | None = None) -> LLMResult:
        """Étape 3 (core/stage3_repair.py) : objets valides conservés ; citations invalides corrigées de façon
        DÉTERMINISTE quand c'est sûr (core/citation_resolver.py) ; objet qui ne garde que des anomalies de citation :
        conservé tel quel et signalé par le validateur (aucun appel au modèle) ; autre anomalie locale : UNE
        réparation par le modèle au plus ; fusion dans l'ordre d'origine, puis validation complète habituelle."""
        if state is None:
            initial = [a for a in attempts if a.get("phase", "initial") == "initial"]
            state = {"state_version": stage3_repair.STATE_VERSION, "call_id": record["call_id"],
                     "model": self.settings.model, "agent": spec.name, "label": label, "root": split["root"],
                     "initial": {"generations": len(initial),
                                 "output_tokens": sum(int(a.get("eval_count") or 0) for a in initial),
                                 "input_tokens": sum(int(a.get("prompt_eval_count") or 0) for a in initial),
                                 "seconds": round(sum(a.get("duration_seconds") or 0 for a in initial), 2)},
                     "items": []}
            for index, item in enumerate(split["items"]):
                state["items"].append(self._triage_item(spec, label, user_content, index, item,
                                                        split["schema_problems"].get(index)))
            stage3_repair.save_state(state_file, state)  # objets validés et corrections déterministes enregistrés
        else:
            reply = {"model": self.settings.model}
        total = len(state["items"])
        reserve = stage3_repair.REPAIR_RESERVE.get(spec.name, 1024)
        schema = stage3_repair.repair_schema(spec)
        number = len(attempts)
        for entry in state["items"]:
            if entry["status"] != stage3_repair.PENDING:
                continue
            if entry["repairs"] >= stage3_repair.MAX_LLM_REPAIRS_PER_OBJECT:  # reprise après une réparation faite
                self._settle(entry)
                stage3_repair.save_state(state_file, state)
                continue
            number += 1
            turns_json = self.checker.repair_turns(spec, label, user_content, entry["object"])
            message = stage3_repair.repair_message(spec, entry["index"], total, entry["object"], entry["problems"],
                                                   turns_json)
            _, text, attempt = await self._generate(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}], schema,
                reserve, record, tokens, number=number, phase="repair", item_index=entry["index"])
            entry["repairs"] += 1  # réparation terminée : jamais une 2e pour le même objet, même après une reprise
            parsed = stage3_repair.parse_repair(text, spec, state["root"], response_model)
            if attempt.get("done_reason") == "length":
                parsed = {"decision": "invalid", "object": None, "reason": None,
                          "problems": ["TRUNCATED — réparation tronquée"]}
            problems = parsed["problems"]
            if parsed["decision"] == "withdrawn":
                entry.update(status=stage3_repair.WITHDRAWN, reason=parsed["reason"])
                status = "withdrawn"
            elif parsed["decision"] == "corrected":
                fixed, log = self.checker.resolve_citations(spec, label, user_content, parsed["object"])
                checked = self.checker.item_check(spec, label, fixed)
                problems = checked["problems"]
                entry["repair_citation_log"] = log
                if not problems:
                    entry.update(status=stage3_repair.REPAIRED, object=fixed, problems=[],
                                 invalid_citations=0, citations=checked["citations"])
                status = "repaired" if not problems else "blocking"
            else:
                status = "invalid"
            if entry["status"] == stage3_repair.PENDING:
                entry["repair_problems"] = problems
                self._settle(entry)
            attempts.append({**attempt, "status": status, "problems": problems})
            self._log_attempt(record, attempts[-1], "repair")
            stage3_repair.save_state(state_file, state)  # chaque réparation terminée est conservée
        output = stage3_repair.merged_output(state, spec)
        try:
            data = response_model.model_validate(output)
        except ValidationError:
            raise LLMError("SCHEMA_VALIDATION", "fusion des objets") from None
        final = self.checker(spec, label, user_content, data.model_dump(mode="json"))  # validation complète
        statuses = [i["status"] for i in state["items"]]
        repairs = [a for a in attempts if a.get("phase") == "repair"]
        quality = stage3_repair.citation_summary(state)
        record.update(
            outcome="accepted" if not final else "accepted_with_issues", remaining_issues=final,
            full_generations=state["initial"]["generations"], repairs=len(repairs),
            initial_output_tokens=state["initial"]["output_tokens"],
            repair_output_tokens=sum(int(a.get("eval_count") or 0) for a in repairs),
            repair_input_tokens=sum(int(a.get("prompt_eval_count") or 0) for a in repairs),
            repair_seconds=round(sum(a.get("duration_seconds") or 0 for a in repairs), 2),
            objects_total=total, objects_valid_initial=statuses.count(stage3_repair.VALID),
            objects_resolved=statuses.count(stage3_repair.RESOLVED),
            objects_repaired=statuses.count(stage3_repair.REPAIRED),
            objects_unresolved=statuses.count(stage3_repair.UNRESOLVED),
            objects_withdrawn=statuses.count(stage3_repair.WITHDRAWN),
            citation_fixes=[{"index": i["index"], **c} for i in state["items"] for c in i.get("citation_log") or []],
            rejected_objects=[{"index": i["index"], "status": i["status"], "reason": i["reason"],
                               "problems": i["problems"]} for i in state["items"]
                              if i["status"] in (stage3_repair.WITHDRAWN, stage3_repair.DROPPED)],
            **quality)
        if quality["citations_fixed_deterministic"]:
            record["correction_reasons"].append(f"citations corrigées sans modèle ({quality['citations_fixed_deterministic']})")
        if repairs:
            record["correction_reasons"].append(f"réparation ciblée ({len(repairs)})")
        text = json.dumps(output, ensure_ascii=False)
        result = self._result(data, record, attempts, started, tokens, reply, text)
        stage3_repair.drop_state(state_file)
        return result

    def _triage_item(self, spec: AgentSpec, label: str, user_content: str, index: int, item,
                     schema_problems: list[str] | None) -> dict:
        """Un objet de la réponse : valide, corrigé sans modèle, conservé avec ses anomalies de citation (signalées
        par le validateur), ou à réparer UNE fois par le modèle (autre anomalie locale)."""
        entry = {"index": index, "original": item, "object": item, "resolved": item, "schema_ok": not schema_problems,
                 "problems": list(schema_problems or []), "invalid_citations_initial": 0, "invalid_citations": 0,
                 "citations": 0, "citation_log": [], "repairs": 0, "reason": None, "status": stage3_repair.PENDING}
        if schema_problems:
            if isinstance(item, dict) and isinstance(item.get("evidence"), list):
                try:  # citations d'un objet hors schéma, pour le bilan
                    entry["invalid_citations_initial"] = self.checker.item_check(spec, label, item)["invalid_citations"]
                except Exception:  # noqa: BLE001 — objet trop malformé pour être lu : aucune citation comptée
                    pass
            return entry
        checked = self.checker.item_check(spec, label, item)
        entry.update(invalid_citations_initial=checked["invalid_citations"], citations=checked["citations"])
        if not checked["problems"]:
            entry["status"] = stage3_repair.VALID
            return entry
        fixed, log = self.checker.resolve_citations(spec, label, user_content, item)
        if log:
            checked = self.checker.item_check(spec, label, fixed)
        entry.update(object=fixed, resolved=fixed, citation_log=log, problems=checked["problems"],
                     invalid_citations=checked["invalid_citations"], citations=checked["citations"])
        if not checked["problems"]:
            entry["status"] = stage3_repair.RESOLVED
        elif checked["citation_only"]:
            # preuve introuvable ou ambiguë : aucun appel au modèle ; le validateur signale l'objet et les étapes
            # suivantes l'écartent s'il n'a aucune citation valide (règle existante)
            entry["status"] = stage3_repair.UNRESOLVED
        return entry

    def _settle(self, entry: dict) -> None:
        """Réparation par le modèle échouée ou déjà tentée : version déterministe conservée (conforme au schéma) et
        signalée par le validateur, ou objet écarté s'il est hors schéma. Jamais une réparation ratée."""
        if entry["schema_ok"]:
            entry.update(status=stage3_repair.UNRESOLVED, object=entry["resolved"])
        else:
            entry["status"] = stage3_repair.DROPPED

    def _finish(self, record: dict, attempts: list, started: float, tokens: dict, text: str) -> None:
        # durée passée sur le modèle (toutes tentatives), sans l'attente d'accès au modèle
        record.update(attempts=len(attempts), finished_at=_now(),
                      duration_seconds=round(sum(a.get("duration_seconds") or 0 for a in attempts), 2),
                      input_tokens=tokens["input_tokens"], output_tokens=tokens["output_tokens"],
                      response_chars=len(text),
                      tokens_per_second=(round(tokens["output_tokens"] / record["generation_seconds"], 1)
                                         if record["generation_seconds"] else None))
        logger.info("%s", call_summary_line(record))
        self._emit({"type": "call_finished", "label": record["label"], "record": dict(record), "at": time.time()})

    def _log_attempt(self, record: dict, attempt: dict, kind: str | None) -> None:
        reason = {"truncated": "réponse tronquée", "format": "JSON ou schéma non conforme",
                  "method": f"{len(attempt['problems'])} anomalie(s) bloquante(s) du validateur",
                  "repair": f"réparation ciblée de l'objet n° {(attempt.get('item_index') or 0) + 1}"}.get(kind)
        logger.info("%s — %s — tentative %s : %s (%s tokens lus, %s générés en %s s, contexte %s)%s",
                    record.get("agent_label") or record.get("agent"), record["label"], attempt["attempt"],
                    attempt["status"], attempt.get("prompt_eval_count"), attempt.get("eval_count"),
                    attempt.get("duration_seconds"), attempt.get("num_ctx"), f" — {reason}" if reason else "")

    def _result(self, data, record: dict, attempts: list, started: float, tokens: dict, reply: dict,
                text: str) -> LLMResult:
        self._log_attempt(record, attempts[-1], None)
        self._finish(record, attempts, started, tokens, text)
        self._journal(record, attempts, started, text=text)
        return LLMResult(data=data, usage=usage_to_dict(tokens), attempts=0, duration_seconds=record["duration_seconds"],
                         model_requested=self.settings.model, response_model=reply.get("model") or self.settings.model,
                         request_id=record["call_id"], stop_reason=reply.get("done_reason"))

    def _journal(self, record: dict, attempts: list, started: float, *, error: dict | None = None,
                 text: str = "") -> None:
        if self.journal_dir is None:
            return
        write_json_atomic(self.journal_dir / f"{record['call_id']}.json", {
            "journal_version": JOURNAL_VERSION, **record, "execution": EXECUTION, "runtime": RUNTIME_NAME,
            "ollama_url": self.settings.ollama_url,
            "options": {"temperature": self.settings.temperature, "num_ctx": record.get("num_ctx"),
                        "num_predict": record.get("num_predict"), "num_ctx_max": self.settings.num_ctx,
                        "keep_alive": self.settings.keep_alive},
            "max_corrections": self.settings.max_corrections,
            "max_method_corrections": self.settings.max_method_corrections,
            "duration_seconds": round(time.monotonic() - started, 3), "attempt_log": attempts, "error": error,
            "final_response": text,
        })


def runtime_error_code(error: dict | None) -> bool:
    """Erreur du runtime (Ollama absent, modèle absent…) plutôt que d'une réponse ?"""
    return bool(error) and error.get("code") in RUNTIME_ERROR_CODES
