"""Échange avec les agents de TRACE : réponse JSON, validation par schéma, erreurs, paramètres.

Les agents sémantiques sont joués par Claude Code (workflow, voir TRACE_WORKFLOW.md et
core/claude_code_workflow.py) : TRACE ne fait AUCUN appel réseau et n'utilise aucun SDK de modèle ni aucune
clé. Ce module contient seulement ce qui est commun à toutes les étapes :

- `parse_json_output` : la réponse d'un agent (texte JSON) validée par le modèle Pydantic de l'agent ;
  une réponse invalide est une erreur (`INVALID_JSON`, `SCHEMA_VALIDATION`, `EMPTY_RESPONSE`), jamais
  corrigée en silence ;
- `LLMError` : erreur d'un agent avec un message sûr (jamais de contenu d'entretien), dont
  `AWAITING_AGENT` (réponse pas encore écrite par l'agent) ;
- `LLMResult` : réponse validée rendue aux orchestrateurs des étapes 3 à 6 ;
- `LLMSettings` : paramètres lus dans l'environnement — tailles des blocs de lecture des entretiens longs
  (étape 3) et étiquette du moteur écrite dans les manifests.

Champs « legacy » des manifests (`api_calls`, `billed_this_run`, `usage`, `request_id`, `stop_reason`) :
conservés pour la compatibilité des sorties et des imports ; le workflow les renseigne toujours avec des
valeurs neutres (0, false, null ; `request_id` = identifiant de la tâche du workflow).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

# Étiquette du moteur, écrite dans les manifests et documents (champ « model ») : pas un identifiant de modèle.
AGENT_RUNNER_MODEL = "workflow_claude_code"

ENV_INTERACTION_CHUNK_TOKENS = "TRACE_INTERACTION_CHUNK_TOKENS"
ENV_PRACTICE_CHUNK_TOKENS = "TRACE_PRACTICE_CHUNK_TOKENS"
# Interaction Signal Reader : taille cible (tokens estimés) d'un bloc d'entretien long (voir core/interaction_chunking.py)
DEFAULT_INTERACTION_CHUNK_TOKENS = 5000
INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX = 500, 30000
# Practice Extractor : taille cible d'un bloc (voir core/practice_chunking.py). Plus petite que celle de
# l'Interaction Reader : une pratique décrite est plus longue qu'un signal.
DEFAULT_PRACTICE_CHUNK_TOKENS = 4000


# --- Paramètres ---------------------------------------------------------------------------------

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


@dataclass(frozen=True)
class LLMSettings:
    """Paramètres des agents, lus dans l'environnement (aucune clé, aucun modèle d'API)."""

    model: str = AGENT_RUNNER_MODEL
    interaction_chunk_tokens: int = DEFAULT_INTERACTION_CHUNK_TOKENS
    practice_chunk_tokens: int = DEFAULT_PRACTICE_CHUNK_TOKENS
    problems: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: dict | None = None) -> "LLMSettings":
        env = os.environ if env is None else env
        problems: list[str] = []
        return cls(
            interaction_chunk_tokens=_int_env(env, ENV_INTERACTION_CHUNK_TOKENS, DEFAULT_INTERACTION_CHUNK_TOKENS,
                                              INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX, problems),
            practice_chunk_tokens=_int_env(env, ENV_PRACTICE_CHUNK_TOKENS, DEFAULT_PRACTICE_CHUNK_TOKENS,
                                           INTERACTION_CHUNK_TOKENS_MIN, INTERACTION_CHUNK_TOKENS_MAX, problems),
            problems=tuple(problems),
        )

    def request_params(self) -> dict:
        """Champ « legacy » des manifests et des clés de cache (paramètres de génération) : toujours neutre."""
        return {"effort": None, "temperature": None}


# --- Erreurs ------------------------------------------------------------------------------------

USER_MESSAGES = {
    "AWAITING_AGENT": "En attente de la réponse de l'agent (workflow Claude Code, aucun appel API).",
    "EMPTY_RESPONSE": "Réponse vide de l'agent.",
    "INVALID_JSON": "Réponse de l'agent non conforme : JSON invalide.",
    "SCHEMA_VALIDATION": "Réponse de l'agent non conforme au schéma attendu.",
    "UNEXPECTED_ERROR": "Erreur inattendue pendant le traitement de la réponse de l'agent.",
}


class LLMError(Exception):
    """Erreur d'un agent, avec un message sûr pour l'utilisateur (jamais de contenu d'entretien)."""

    def __init__(self, code: str, detail: str | None = None, *, request_id: str | None = None):
        self.code = code
        self.detail = detail
        self.request_id = request_id
        # Champs « legacy » lus par les orchestrateurs (manifests) : toujours neutres.
        self.status_code: int | None = None
        self.usage: dict | None = None
        self.attempts = 0
        self.duration_seconds = 0.0
        super().__init__(self.user_message)

    @property
    def user_message(self) -> str:
        message = USER_MESSAGES.get(self.code, USER_MESSAGES["UNEXPECTED_ERROR"])
        extras = [x for x in (self.detail, f"tâche {self.request_id}" if self.request_id else None) if x]
        return message + (f" ({' ; '.join(extras)})" if extras else "")

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.user_message, "status_code": self.status_code,
                "request_id": self.request_id}


# --- Résultat -----------------------------------------------------------------------------------

def usage_to_dict(usage: Any = None) -> dict:
    """Champ « legacy » `usage` des manifests (tokens) : le workflow n'en consomme pas, toujours neutre."""
    def get(name):
        value = getattr(usage, name, None) if usage is not None else None
        return value if isinstance(value, int) else None

    return {"input_tokens": get("input_tokens"), "output_tokens": get("output_tokens"),
            "cache_creation_input_tokens": get("cache_creation_input_tokens"),
            "cache_read_input_tokens": get("cache_read_input_tokens")}


@dataclass
class LLMResult:
    """Réponse validée d'un agent. `attempts` (nombre d'appels API) vaut toujours 0 dans le workflow."""

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
