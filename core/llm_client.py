"""Échange avec les agents de TRACE : paramètres du modèle local, erreurs, résultat, validation par schéma.

Les agents sémantiques des étapes 3 à 6 sont exécutés par un modèle LOCAL servi par Ollama
(core/local_agent_runner.py) : aucune API externe, aucune clé, aucun SDK de modèle. Ce module contient ce qui est
commun à toutes les étapes :

- `LLMSettings` : paramètres lus dans l'environnement — modèle Ollama (`TRACE_LOCAL_MODEL`), adresse locale
  d'Ollama (`TRACE_OLLAMA_URL`, boucle locale seulement), fenêtre de contexte, délai, nombre de corrections
  locales, tailles des blocs de lecture des entretiens longs (étape 3) ;
- `parse_json_output` : la réponse d'un agent (texte JSON) validée par le modèle Pydantic de l'agent ; une
  réponse invalide est une erreur (`INVALID_JSON`, `SCHEMA_VALIDATION`, `EMPTY_RESPONSE`), jamais corrigée en
  silence ;
- `LLMError` : erreur d'un agent ou du runtime local, avec un message sûr et une action claire (jamais de contenu
  d'entretien) ;
- `LLMResult` : réponse validée rendue aux orchestrateurs des étapes 3 à 6.

Champs « legacy » des manifests (`api_calls`, `billed_this_run`) : conservés pour la compatibilité des sorties et
des imports ; ils valent toujours 0 / false (aucun appel API, aucun coût). `usage` reçoit les comptes de tokens
rapportés par Ollama (information locale), `request_id` l'identifiant de l'appel local (journal de l'appel).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

# --- Paramètres du runtime local -----------------------------------------------------------------

ENV_LOCAL_MODEL = "TRACE_LOCAL_MODEL"
ENV_OLLAMA_URL = "TRACE_OLLAMA_URL"
ENV_NUM_CTX = "TRACE_OLLAMA_NUM_CTX"
ENV_TIMEOUT = "TRACE_OLLAMA_TIMEOUT"
ENV_MAX_CORRECTIONS = "TRACE_LOCAL_MAX_CORRECTIONS"
ENV_MAX_METHOD_CORRECTIONS = "TRACE_LOCAL_MAX_METHOD_CORRECTIONS"
ENV_KEEP_ALIVE = "TRACE_OLLAMA_KEEP_ALIVE"
ENV_TEMPERATURE = "TRACE_LOCAL_TEMPERATURE"
ENV_CONCURRENCY = "TRACE_LOCAL_CONCURRENCY"
ENV_INTERACTION_CHUNK_TOKENS = "TRACE_INTERACTION_CHUNK_TOKENS"
ENV_PRACTICE_CHUNK_TOKENS = "TRACE_PRACTICE_CHUNK_TOKENS"
ENV_VARS = (ENV_LOCAL_MODEL, ENV_OLLAMA_URL, ENV_NUM_CTX, ENV_TIMEOUT, ENV_MAX_CORRECTIONS, ENV_MAX_METHOD_CORRECTIONS,
            ENV_KEEP_ALIVE, ENV_TEMPERATURE, ENV_CONCURRENCY, ENV_INTERACTION_CHUNK_TOKENS, ENV_PRACTICE_CHUNK_TOKENS)

# Modèle par défaut : bon en français, sorties JSON structurées fiables, contexte long, et tient entièrement dans
# une carte graphique de 8 Go avec le contexte dimensionné par appel. Remplaçable par TRACE_LOCAL_MODEL.
DEFAULT_LOCAL_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Fenêtre de contexte : choisie PAR APPEL (la plus petite de CTX_BUCKETS qui contient le prompt et une marge pour la
# réponse), jamais au-delà de ce maximum. Le défaut d'Ollama (2 048 à 4 096) tronquerait le matériau ; un contexte
# systématique de 32 768 alloue un cache inutile qui peut déborder de la carte graphique (calcul partiel sur CPU).
DEFAULT_NUM_CTX = 32768
NUM_CTX_MIN, NUM_CTX_MAX = 4096, 131072
CTX_BUCKETS = (4096, 8192, 12288, 16384, 20480, 24576, 32768, 49152, 65536, 98304, 131072)
DEFAULT_KEEP_ALIVE = "30m"              # garder le modèle chargé entre deux appels (pas de rechargement)
DEFAULT_TIMEOUT_SECONDS = 1800          # une génération locale peut être longue sur un ordinateur portable
TIMEOUT_MIN, TIMEOUT_MAX = 30, 7200
DEFAULT_MAX_CORRECTIONS = 2             # nouvelles tentatives LOCALES après un JSON invalide ou hors schéma
DEFAULT_MAX_METHOD_CORRECTIONS = 1      # … après une anomalie BLOQUANTE du validateur méthodologique (réponse conforme
                                        # au schéma) : chaque correction régénère toute la réponse
MAX_CORRECTIONS_LIMIT = 5
DEFAULT_TEMPERATURE = 0.0               # reproductibilité : génération déterministe autant que possible
DEFAULT_CONCURRENCY = 1                 # un seul appel à la fois : un modèle local occupe toute la machine
CONCURRENCY_MAX = 4

# Interaction Signal Reader : taille cible (tokens estimés) d'un bloc d'entretien long (voir core/interaction_chunking.py)
DEFAULT_INTERACTION_CHUNK_TOKENS = 5000
INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX = 500, 30000
# Practice Extractor : taille cible d'un bloc (voir core/practice_chunking.py). Plus petite que celle de
# l'Interaction Reader : une pratique décrite est plus longue qu'un signal.
DEFAULT_PRACTICE_CHUNK_TOKENS = 4000


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


def _float_env(env, name, default, low, high, problems):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        problems.append(f"{name} invalide (« {raw} ») : valeur par défaut {default} utilisée.")
        return default
    clamped = min(max(value, low), high)
    if clamped != value:
        problems.append(f"{name}={value} hors limites : {clamped} utilisé.")
    return clamped


@dataclass(frozen=True)
class LLMSettings:
    """Paramètres des agents, lus dans l'environnement. Aucune clé, aucun fournisseur externe."""

    model: str = DEFAULT_LOCAL_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    num_ctx: int = DEFAULT_NUM_CTX
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_corrections: int = DEFAULT_MAX_CORRECTIONS
    max_method_corrections: int = DEFAULT_MAX_METHOD_CORRECTIONS
    keep_alive: str = DEFAULT_KEEP_ALIVE
    temperature: float = DEFAULT_TEMPERATURE
    concurrency: int = DEFAULT_CONCURRENCY
    interaction_chunk_tokens: int = DEFAULT_INTERACTION_CHUNK_TOKENS
    practice_chunk_tokens: int = DEFAULT_PRACTICE_CHUNK_TOKENS
    problems: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: dict | None = None) -> "LLMSettings":
        env = os.environ if env is None else env
        problems: list[str] = []
        return cls(
            model=(env.get(ENV_LOCAL_MODEL) or "").strip() or DEFAULT_LOCAL_MODEL,
            ollama_url=((env.get(ENV_OLLAMA_URL) or "").strip() or DEFAULT_OLLAMA_URL).rstrip("/"),
            num_ctx=_int_env(env, ENV_NUM_CTX, DEFAULT_NUM_CTX, NUM_CTX_MIN, NUM_CTX_MAX, problems),
            timeout_seconds=_int_env(env, ENV_TIMEOUT, DEFAULT_TIMEOUT_SECONDS, TIMEOUT_MIN, TIMEOUT_MAX, problems),
            max_corrections=_int_env(env, ENV_MAX_CORRECTIONS, DEFAULT_MAX_CORRECTIONS, 0, MAX_CORRECTIONS_LIMIT,
                                     problems),
            max_method_corrections=_int_env(env, ENV_MAX_METHOD_CORRECTIONS, DEFAULT_MAX_METHOD_CORRECTIONS, 0,
                                            MAX_CORRECTIONS_LIMIT, problems),
            keep_alive=(env.get(ENV_KEEP_ALIVE) or "").strip() or DEFAULT_KEEP_ALIVE,
            temperature=_float_env(env, ENV_TEMPERATURE, DEFAULT_TEMPERATURE, 0.0, 1.0, problems),
            concurrency=_int_env(env, ENV_CONCURRENCY, DEFAULT_CONCURRENCY, 1, CONCURRENCY_MAX, problems),
            interaction_chunk_tokens=_int_env(env, ENV_INTERACTION_CHUNK_TOKENS, DEFAULT_INTERACTION_CHUNK_TOKENS,
                                              INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX, problems),
            practice_chunk_tokens=_int_env(env, ENV_PRACTICE_CHUNK_TOKENS, DEFAULT_PRACTICE_CHUNK_TOKENS,
                                           INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX, problems),
            problems=tuple(problems),
        )

    def request_params(self) -> dict:
        """Paramètres de génération qui influencent la réponse : écrits dans les manifests et dans les clés de
        cache (`effort` : champ legacy, toujours null)."""
        return {"effort": None, "temperature": self.temperature}


# --- Erreurs ------------------------------------------------------------------------------------

USER_MESSAGES = {
    "EMPTY_RESPONSE": "Réponse vide de l'agent.",
    "INVALID_JSON": "Réponse de l'agent non conforme : JSON invalide.",
    "SCHEMA_VALIDATION": "Réponse de l'agent non conforme au schéma attendu.",
    "TRUNCATED": "Réponse du modèle local tronquée (fenêtre de contexte atteinte) : augmentez TRACE_OLLAMA_NUM_CTX "
                 "ou choisissez un modèle à contexte plus long.",
    "OLLAMA_UNAVAILABLE": "Ollama ne répond pas : démarrez-le sur cet ordinateur (commande « ollama serve » ou "
                          "application Ollama), puis relancez. Aucun autre fournisseur n'est utilisé.",
    "MODEL_NOT_FOUND": "Le modèle local demandé n'est pas installé dans Ollama.",
    "NON_LOCAL_URL": "Adresse d'Ollama refusée : TRACE n'appelle qu'un Ollama local (localhost, 127.0.0.1 ou ::1), "
                     "aucune donnée ne quitte cet ordinateur.",
    "OLLAMA_TIMEOUT": "Le modèle local n'a pas répondu dans le délai imparti (TRACE_OLLAMA_TIMEOUT).",
    "PROMPT_TOO_LONG": "Requête plus longue que la fenêtre de contexte maximale (TRACE_OLLAMA_NUM_CTX) : elle serait "
                       "tronquée par Ollama ; augmentez TRACE_OLLAMA_NUM_CTX (rien n'est envoyé tronqué).",
    "OLLAMA_ERROR": "Erreur renvoyée par Ollama.",
    "UNEXPECTED_ERROR": "Erreur inattendue pendant le traitement de la réponse de l'agent.",
}
# Erreurs du runtime (et non de la réponse) : jamais de nouvelle tentative, jamais de repli.
RUNTIME_ERROR_CODES = ("OLLAMA_UNAVAILABLE", "MODEL_NOT_FOUND", "NON_LOCAL_URL", "OLLAMA_TIMEOUT", "OLLAMA_ERROR",
                       "PROMPT_TOO_LONG")


class LLMError(Exception):
    """Erreur d'un agent ou du runtime local, avec un message sûr pour l'utilisateur (jamais de contenu
    d'entretien)."""

    def __init__(self, code: str, detail: str | None = None, *, request_id: str | None = None):
        self.code = code
        self.detail = detail
        self.request_id = request_id
        # Champs lus par les orchestrateurs (manifests) : `attempts` = appels API (legacy), toujours 0.
        self.status_code: int | None = None
        self.usage: dict | None = None
        self.attempts = 0
        self.duration_seconds = 0.0
        super().__init__(self.user_message)

    @property
    def user_message(self) -> str:
        message = USER_MESSAGES.get(self.code, USER_MESSAGES["UNEXPECTED_ERROR"])
        extras = [x for x in (self.detail, f"appel {self.request_id}" if self.request_id else None) if x]
        return message + (f" ({' ; '.join(extras)})" if extras else "")

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.user_message, "status_code": self.status_code,
                "request_id": self.request_id}


# --- Résultat -----------------------------------------------------------------------------------

def usage_to_dict(usage: Any = None) -> dict:
    """Champ `usage` des manifests : tokens rapportés par le modèle local (objet ou dict), sinon null."""
    def get(name):
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return value if isinstance(value, int) else None

    return {"input_tokens": get("input_tokens"), "output_tokens": get("output_tokens"),
            "cache_creation_input_tokens": get("cache_creation_input_tokens"),
            "cache_read_input_tokens": get("cache_read_input_tokens")}


@dataclass
class LLMResult:
    """Réponse validée d'un agent. `attempts` alimente le champ legacy `api_calls` : toujours 0 (aucune API)."""

    data: BaseModel
    usage: dict
    attempts: int
    duration_seconds: float
    model_requested: str
    response_model: str | None
    request_id: str | None
    stop_reason: str | None


# --- Validation d'une réponse -------------------------------------------------------------------

def parse_json_output(text: str, response_model: type[BaseModel], max_problems: int = 5) -> BaseModel:
    """Parse le JSON produit par un agent et le valide avec son schéma Pydantic. Jamais silencieux."""
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
        detail = "; ".join(problems[:max_problems]) + (
            f" (+{len(problems) - max_problems})" if len(problems) > max_problems else "")
        raise LLMError("SCHEMA_VALIDATION", detail) from None
