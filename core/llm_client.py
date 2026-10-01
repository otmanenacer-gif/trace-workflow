"""Client LLM minimal de TRACE, au-dessus du SDK Python officiel Anthropic.

Règles :
- la clé est lue UNIQUEMENT dans la variable d'environnement ANTHROPIC_API_KEY
  et transmise explicitement au SDK (aucune autre source d'identifiants) ;
- le modèle est lu UNIQUEMENT dans ANTHROPIC_MODEL (aucun identifiant de
  modèle n'est écrit dans le code) ;
- sans clé ou sans modèle, l'analyse IA est désactivée, l'ingestion continue ;
- sortie structurée officielle (`output_config.format` + JSON schema), puis
  validation immédiate de la réponse : une réponse invalide est une erreur ;
- réessais limités (429, 408/409, 5xx, délai dépassé, réseau) avec backoff
  exponentiel borné, jamais de boucle infinie ;
- les messages d'erreur ne contiennent ni la clé, ni le contenu des entretiens.

Le transport (appel HTTP réel) est injectable : les tests utilisent un faux
transport et ne consomment jamais de tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_MODEL = "ANTHROPIC_MODEL"
ENV_MAX_CONCURRENCY = "TRACE_MAX_CONCURRENCY"
ENV_TIMEOUT = "TRACE_LLM_TIMEOUT_SECONDS"
ENV_MAX_RETRIES = "TRACE_LLM_MAX_RETRIES"
ENV_MAX_TOKENS = "TRACE_LLM_MAX_TOKENS"
ENV_EFFORT = "TRACE_LLM_EFFORT"
ENV_TEMPERATURE = "TRACE_LLM_TEMPERATURE"
ENV_INTERACTION_CHUNK_TOKENS = "TRACE_INTERACTION_CHUNK_TOKENS"

DEFAULT_MAX_CONCURRENCY = 2
MAX_CONCURRENCY_LIMIT = 8
DEFAULT_TIMEOUT_SECONDS = 600.0
CONNECT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RETRIES = 2          # soit 3 tentatives au plus
MAX_RETRIES_LIMIT = 5
DEFAULT_MAX_TOKENS = 32000
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 30.0
RETRY_AFTER_MAX_SECONDS = 60.0
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Interaction Signal Reader : taille cible (tokens estimés) d'un bloc d'entretien long (voir core/interaction_chunking.py)
DEFAULT_INTERACTION_CHUNK_TOKENS = 5000
INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX = 500, 30000

RETRYABLE_STATUS = {408, 409, 429}

Transport = Callable[[dict], Awaitable[Any]]


# --- Configuration ----------------------------------------------------------------------

def _int_env(env, name, default, low, high, problems):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        problems.append(f"{name} invalide (« {raw} ») : valeur par défaut {default} utilisée.")
        return default
    clamped = min(max(value, low), high)
    if clamped != value:
        problems.append(f"{name}={value} hors limites : {clamped} utilisé.")
    return clamped


def _float_env(env, name, default, problems):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        problems.append(f"{name} invalide (« {raw} ») : ignoré.")
        return default
    return value


@dataclass(frozen=True)
class LLMSettings:
    """Paramètres de l'analyse IA, lus dans l'environnement."""

    api_key: str | None = field(default=None, repr=False)
    model: str | None = None
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = None
    temperature: float | None = None
    interaction_chunk_tokens: int = DEFAULT_INTERACTION_CHUNK_TOKENS
    problems: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: dict | None = None) -> "LLMSettings":
        env = os.environ if env is None else env
        problems: list[str] = []
        effort = (env.get(ENV_EFFORT) or "").strip().lower() or None
        if effort and effort not in EFFORT_LEVELS:
            problems.append(f"{ENV_EFFORT} invalide (« {effort} ») : ignoré.")
            effort = None
        timeout = _float_env(env, ENV_TIMEOUT, DEFAULT_TIMEOUT_SECONDS, problems)
        if timeout <= 0:
            problems.append(f"{ENV_TIMEOUT} doit être positif : {DEFAULT_TIMEOUT_SECONDS:g} utilisé.")
            timeout = DEFAULT_TIMEOUT_SECONDS
        return cls(
            api_key=(env.get(ENV_API_KEY) or "").strip() or None,
            model=(env.get(ENV_MODEL) or "").strip() or None,
            max_concurrency=_int_env(env, ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY, 1, MAX_CONCURRENCY_LIMIT, problems),
            timeout_seconds=timeout,
            max_retries=_int_env(env, ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES, 0, MAX_RETRIES_LIMIT, problems),
            max_tokens=_int_env(env, ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS, 1024, 128000, problems),
            effort=effort,
            temperature=_float_env(env, ENV_TEMPERATURE, None, problems),
            interaction_chunk_tokens=_int_env(env, ENV_INTERACTION_CHUNK_TOKENS, DEFAULT_INTERACTION_CHUNK_TOKENS,
                                              INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX, problems),
            problems=tuple(problems),
        )

    @property
    def missing(self) -> list[str]:
        """Variables obligatoires absentes (noms seulement)."""
        return [name for name, value in ((ENV_API_KEY, self.api_key), (ENV_MODEL, self.model)) if not value]

    @property
    def enabled(self) -> bool:
        return not self.missing

    def disabled_reason(self) -> str | None:
        """Message explicite si l'analyse IA est désactivée, sinon None."""
        if self.enabled:
            return None
        return (
            "Analyse IA désactivée : variable(s) d'environnement manquante(s) : "
            + ", ".join(self.missing)
            + ". L'ingestion reste disponible. Renseignez ces variables (voir .env.example) puis relancez l'application."
        )

    def request_params(self) -> dict:
        """Paramètres de génération qui peuvent influencer le résultat (manifest, cache)."""
        return {"effort": self.effort, "temperature": self.temperature}


# --- Erreurs ----------------------------------------------------------------------------

USER_MESSAGES = {
    "NOT_CONFIGURED": "Analyse IA non configurée (clé API ou modèle absent).",
    "AUTHENTICATION_ERROR": "Clé API refusée par Anthropic : vérifiez ANTHROPIC_API_KEY.",
    "PERMISSION_DENIED": "Cette clé API n'a pas accès à ce modèle ou à cette ressource.",
    "MODEL_NOT_FOUND": "Modèle introuvable : vérifiez la valeur de ANTHROPIC_MODEL.",
    "BAD_REQUEST": "Requête refusée par l'API (paramètre non pris en charge par ce modèle ?).",
    "REQUEST_TOO_LARGE": "Entretien trop long pour une seule requête.",
    "RATE_LIMITED": "Limite de débit de l'API atteinte (429) malgré les réessais : réessayez plus tard ou baissez TRACE_MAX_CONCURRENCY.",
    "SERVER_ERROR": "Erreur temporaire côté Anthropic (5xx) malgré les réessais : réessayez plus tard.",
    "TIMEOUT": "Délai dépassé malgré les réessais : réessayez plus tard.",
    "CONNECTION_ERROR": "Connexion à l'API impossible malgré les réessais : vérifiez le réseau.",
    "API_ERROR": "Erreur de l'API Anthropic.",
    "REFUSAL": "Le modèle a refusé de traiter la demande (stop_reason=refusal).",
    "TRUNCATED": "Réponse tronquée (limite max_tokens atteinte) : augmentez TRACE_LLM_MAX_TOKENS.",
    "EMPTY_RESPONSE": "Réponse vide du modèle.",
    "INVALID_JSON": "Réponse du modèle non conforme : JSON invalide.",
    "SCHEMA_VALIDATION": "Réponse du modèle non conforme au schéma attendu.",
    "UNEXPECTED_ERROR": "Erreur inattendue pendant l'appel au modèle.",
}


class LLMError(Exception):
    """Erreur d'appel au modèle, avec un message sûr pour l'utilisateur."""

    def __init__(self, code: str, detail: str | None = None, *, retryable: bool = False,
                 status_code: int | None = None, request_id: str | None = None,
                 retry_after: float | None = None):
        self.code = code
        self.detail = detail
        self.retryable = retryable
        self.status_code = status_code
        self.request_id = request_id
        self.retry_after = retry_after
        # Renseignés par le client : coût d'un échec (tokens déjà consommés)
        self.usage: dict | None = None
        self.attempts = 0
        self.duration_seconds = 0.0
        super().__init__(self.user_message)

    @property
    def user_message(self) -> str:
        message = USER_MESSAGES.get(self.code, USER_MESSAGES["UNEXPECTED_ERROR"])
        extras = []
        if self.status_code:
            extras.append(f"HTTP {self.status_code}")
        if self.detail:
            extras.append(self.detail)
        if self.request_id:
            extras.append(f"request_id {self.request_id}")
        return message + (f" ({' ; '.join(extras)})" if extras else "")

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.user_message, "status_code": self.status_code,
                "request_id": self.request_id}


_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def redact(text: str, secret: str | None = None) -> str:
    """Retire toute clé API d'un texte destiné à l'utilisateur ou aux journaux."""
    if secret:
        text = text.replace(secret, "[clé masquée]")
    return _KEY_PATTERN.sub("[clé masquée]", text)


def _short(text: str | None, limit: int = 300) -> str | None:
    if not text:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    try:
        value = float(exc.response.headers.get("retry-after", ""))
    except (TypeError, ValueError, AttributeError):
        return None
    return max(0.0, value)


def classify_exception(exc: BaseException, secret: str | None = None) -> LLMError:
    """Traduit une exception du SDK en LLMError (du plus spécifique au plus général)."""
    request_id = getattr(exc, "request_id", None)
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, anthropic.APITimeoutError)):
        return LLMError("TIMEOUT", retryable=True)
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError("CONNECTION_ERROR", retryable=True)
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        # Détail de l'API utile au diagnostic (jamais le contenu de l'entretien, clé masquée)
        detail = _short(redact(str(getattr(exc, "message", "") or ""), secret))
        common = {"status_code": status, "request_id": request_id}
        if isinstance(exc, anthropic.AuthenticationError):
            return LLMError("AUTHENTICATION_ERROR", **common)
        if isinstance(exc, anthropic.PermissionDeniedError):
            return LLMError("PERMISSION_DENIED", **common)
        if isinstance(exc, anthropic.NotFoundError):
            return LLMError("MODEL_NOT_FOUND", **common)
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError("RATE_LIMITED", retryable=True, retry_after=_retry_after(exc), **common)
        if status == 413:
            return LLMError("REQUEST_TOO_LARGE", **common)
        if isinstance(exc, anthropic.BadRequestError):
            return LLMError("BAD_REQUEST", detail, **common)
        if status >= 500:
            return LLMError("SERVER_ERROR", retryable=True, retry_after=_retry_after(exc), **common)
        if status in RETRYABLE_STATUS:
            return LLMError("API_ERROR", retryable=True, retry_after=_retry_after(exc), **common)
        return LLMError("API_ERROR", detail, **common)
    if isinstance(exc, anthropic.APIError):  # erreur survenue pendant le flux de réponse
        return LLMError("API_ERROR", _short(redact(str(getattr(exc, "message", "") or ""), secret)), retryable=True)
    return LLMError("UNEXPECTED_ERROR", type(exc).__name__)


def backoff_delay(attempt: int, retry_after: float | None = None, rng: random.Random | None = None) -> float:
    """Délai avant la tentative suivante (attempt = numéro de la tentative échouée, 1-indexé)."""
    if retry_after is not None:
        return min(retry_after, RETRY_AFTER_MAX_SECONDS)
    base = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
    jitter = (rng or random).uniform(0.75, 1.25)
    return min(base * jitter, BACKOFF_MAX_SECONDS)


# --- Résultat ---------------------------------------------------------------------------

def usage_to_dict(usage: Any) -> dict:
    """Tokens rapportés par l'API (None si l'information n'est pas fournie)."""
    def get(name):
        value = getattr(usage, name, None) if usage is not None else None
        return value if isinstance(value, int) else None

    return {
        "input_tokens": get("input_tokens"),
        "output_tokens": get("output_tokens"),
        "cache_creation_input_tokens": get("cache_creation_input_tokens"),
        "cache_read_input_tokens": get("cache_read_input_tokens"),
    }


@dataclass
class LLMResult:
    data: BaseModel
    usage: dict
    attempts: int
    duration_seconds: float
    model_requested: str
    response_model: str | None
    request_id: str | None
    stop_reason: str | None


# --- Transport réel ---------------------------------------------------------------------

class AnthropicTransport:
    """Appel réel à l'API Messages (flux, pour ne pas subir les délais HTTP des longues réponses).

    Les réessais du SDK sont désactivés (max_retries=0) : ils sont gérés,
    bornés et journalisés par LLMClient.
    """

    def __init__(self, settings: LLMSettings, http_client: Any = None):
        options = {"http_client": http_client} if http_client is not None else {}  # client HTTP de test
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.api_key,  # clé explicite : aucune autre source d'identifiants
            timeout=anthropic.Timeout(settings.timeout_seconds, connect=CONNECT_TIMEOUT_SECONDS),
            max_retries=0,
            **options,
        )

    async def __call__(self, params: dict) -> Any:
        async with self._client.messages.stream(**params) as stream:
            message = await stream.get_final_message()
            if getattr(message, "_request_id", None) is None:
                message._request_id = stream.request_id  # le message final d'un flux ne le porte pas
            return message

    async def aclose(self) -> None:
        await self._client.close()


# --- Client -----------------------------------------------------------------------------

class LLMClient:
    """Envoie une requête à sortie JSON structurée et valide la réponse.

    Un sémaphore limite le nombre d'appels simultanés (TRACE_MAX_CONCURRENCY).
    """

    def __init__(self, settings: LLMSettings, transport: Transport | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        if not settings.enabled:
            raise LLMError("NOT_CONFIGURED", ", ".join(settings.missing))
        self.settings = settings
        self._transport = transport if transport is not None else AnthropicTransport(settings)
        self._sleep = sleep
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:  # créé dans la boucle asyncio en cours
            self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        return self._semaphore

    async def aclose(self) -> None:
        close = getattr(self._transport, "aclose", None)
        if close is not None:
            await close()

    def build_params(self, system_prompt: str, user_content: str, output_schema: dict) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": output_schema}}
        if self.settings.effort:
            output_config["effort"] = self.settings.effort
        params = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            # Consignes stables en tête (réutilisables par le cache de prompt de l'API),
            # transcription (variable) dans le message utilisateur.
            "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_content}],
            "output_config": output_config,
        }
        if self.settings.temperature is not None:
            # Paramètre refusé par les modèles récents : envoyé seulement s'il est configuré.
            params["extra_body"] = {"temperature": self.settings.temperature}
        return params

    async def complete_json(self, *, system_prompt: str, user_content: str, output_schema: dict,
                            response_model: type[BaseModel], label: str) -> LLMResult:
        """Un appel (avec réessais bornés) → objet validé par `response_model`.

        `label` sert uniquement aux journaux (ex. « ELOISE/practice_extractor ») :
        aucun contenu d'entretien n'est journalisé.
        """
        params = self.build_params(system_prompt, user_content, output_schema)
        started = time.monotonic()
        attempts = 0
        async with self.semaphore:
            while True:
                attempts += 1
                try:
                    response = await asyncio.wait_for(self._transport(params), self.settings.timeout_seconds)
                    break
                except Exception as exc:  # noqa: BLE001 — classé ci-dessous
                    error = classify_exception(exc, self.settings.api_key)
                    if not error.retryable or attempts > self.settings.max_retries:
                        error.attempts = attempts
                        error.duration_seconds = round(time.monotonic() - started, 3)
                        logger.warning("Appel LLM %s en échec : %s (tentative %d)", label, error.code, attempts)
                        raise error from None
                    delay = backoff_delay(attempts, error.retry_after)
                    logger.info("Appel LLM %s : %s, nouvel essai dans %.1f s (tentative %d/%d)",
                                label, error.code, delay, attempts, self.settings.max_retries + 1)
                    await self._sleep(delay)
        duration = round(time.monotonic() - started, 3)
        usage = usage_to_dict(getattr(response, "usage", None))
        request_id = getattr(response, "_request_id", None)
        try:
            data = self.parse_response(response, response_model)
        except LLMError as error:
            error.usage, error.attempts, error.duration_seconds = usage, attempts, duration
            error.request_id = error.request_id or request_id
            logger.warning("Réponse LLM %s rejetée : %s", label, error.code)
            raise
        logger.info("Appel LLM %s réussi : %s tokens entrée, %s tokens sortie, %.1f s, %d tentative(s)",
                    label, usage["input_tokens"], usage["output_tokens"], duration, attempts)
        return LLMResult(
            data=data, usage=usage, attempts=attempts, duration_seconds=duration,
            model_requested=self.settings.model, response_model=getattr(response, "model", None),
            request_id=request_id, stop_reason=getattr(response, "stop_reason", None),
        )

    @staticmethod
    def parse_response(response: Any, response_model: type[BaseModel]) -> BaseModel:
        """Vérifie l'arrêt, extrait le texte, parse le JSON et valide le schéma. Jamais silencieux."""
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise LLMError("REFUSAL")
        if stop_reason == "max_tokens":
            raise LLMError("TRUNCATED")
        text = "".join(getattr(block, "text", "") for block in (getattr(response, "content", None) or [])
                       if getattr(block, "type", None) == "text")
        if not text.strip():
            raise LLMError("EMPTY_RESPONSE")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("INVALID_JSON", f"position {exc.pos}") from None
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            # Emplacements et types d'erreur seulement : jamais les valeurs (extraits d'entretien)
            problems = [f"{'.'.join(str(p) for p in e['loc']) or '(racine)'}: {e['type']}"
                        for e in exc.errors(include_url=False, include_context=False, include_input=False)]
            detail = "; ".join(problems[:5]) + (f" (+{len(problems) - 5})" if len(problems) > 5 else "")
            raise LLMError("SCHEMA_VALIDATION", detail) from None
