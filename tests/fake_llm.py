"""Agents simulés pour les tests : aucun modèle réel, aucune connexion réseau, aucun SDK, aucune clé.

- `FakeOllama` : faux Ollama (transport du LocalAgentRunner) ; chaque requête /api/chat est servie par un
  « répondeur » du test, choisi d'après l'agent (déduit du schéma demandé). Il peut aussi simuler un Ollama arrêté
  ou un modèle absent.
- `FakeLocalAgentRunner` : le VRAI LocalAgentRunner (core/local_agent_runner.py) branché sur ce faux Ollama :
  mêmes requêtes, même parsing, même validation Pydantic, mêmes corrections locales, même contrôle méthodologique.
- `use_fake_runtime` : remplace le runner du pipeline local (core/local_pipeline.make_runner) dans un test, pour
  exécuter les étapes 3 à 6 (ou l'interface Streamlit) de bout en bout sans modèle réel.
- `FakeAgents` : double minimal du client des agents, branché directement sur les orchestrateurs (tests
  d'orchestration) : une réponse par appel, sans correction.

Répondeur : une réponse (`text_response(...)`, dict, texte), une exception, une liste (consommée dans l'ordre) ou
une fonction (params) -> réponse (éventuellement async). `params` : `system`, `messages`, `output_schema`, `label`
(et, pour FakeOllama, `attempt` et `history` : la conversation de correction).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass

from core.llm_client import (DEFAULT_LOCAL_MODEL, LLMError, LLMResult, LLMSettings, parse_json_output,
                             usage_to_dict)
from core.local_agent_runner import LocalAgentRunner, model_installed

PRACTICE = "practice_extractor"
INTERACTION = "interaction_signal_reader"
AUDITOR = "speaker_attribution_auditor"
LONG_DISTANCE = "interaction_signal_reader_long_distance"  # lecture à longue distance (entretien long)
ACCOUNTABILITY = "accountability_episode_builder"  # étape 4 : épisodes d'accountability
TRAJECTORY = "trajectory_mapper"  # étape 5 : configuration et trajectoire intra-entretien
COMPARATOR = "cross_interview_comparator"  # étape 6 : comparaison inter-entretiens

REPAIR = "repair:"  # préfixe des réparations ciblées de l'étape 3 : "repair:practice_extractor", …

FAKE_MODEL = "fake-model"  # étiquette du moteur dans les manifests des tests (champ « model »)


@dataclass
class AgentReply:
    """Réponse d'un agent simulé : le texte JSON que le modèle local renverrait."""

    text: str


def text_response(payload) -> AgentReply:
    return AgentReply(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))


def agent_error(code: str = "SCHEMA_VALIDATION", detail: str | None = None) -> LLMError:
    """Échec d'un agent (réponse non conforme) : à renvoyer ou lever par un répondeur."""
    return LLMError(code, detail)


def agent_of(params: dict) -> str:
    properties = params["output_schema"]["properties"]
    if "decision" in properties and "object" in properties:  # réparation ciblée d'UN objet (core/stage3_repair.py)
        item = properties["object"]["anyOf"][0]
        if "$ref" in item:
            item = params["output_schema"]["$defs"][item["$ref"].rsplit("/", 1)[1]]
        if "use_status" in item["properties"]:
            return REPAIR + PRACTICE
        long_distance = params["system"][0]["text"].startswith("# Interaction Signal Reader — lecture à longue distance")
        return REPAIR + (LONG_DISTANCE if long_distance else INTERACTION)
    if "assessments" in properties:
        return AUDITOR
    if "cross_case_claims" in properties:
        return COMPARATOR
    if "student_role_criteria" in properties:
        return TRAJECTORY
    if "episodes" in properties:
        return ACCOUNTABILITY
    if params["system"][0]["text"].startswith("# Interaction Signal Reader — lecture à longue distance"):
        return LONG_DISTANCE
    return PRACTICE if "practices" in properties else INTERACTION


class FakeAgents:
    """Répond selon l'agent (déduit du schéma demandé) : PRACTICE, INTERACTION, AUDITOR, LONG_DISTANCE,
    ACCOUNTABILITY (étape 4), TRAJECTORY (étape 5) ou COMPARATOR (étape 6).

    Un appel à un agent sans réponse prévue échoue (KeyError) : un test qui ne prévoit
    pas d'audit des locuteurs vérifie donc aussi qu'aucun appel d'audit n'a lieu.

    `responders[agent]` peut être : une réponse, une exception, une liste
    (consommée dans l'ordre) ou une fonction (params) -> réponse (éventuellement async).
    `params` décrit la tâche de l'agent : `system` (consignes), `messages` (message exact), `output_schema`, `label`.
    """

    def __init__(self, responders: dict):
        self.responders = {k: (list(v) if isinstance(v, list) else v) for k, v in responders.items()}
        self.calls: list[dict] = []
        self.closed = False

    async def _respond(self, params: dict):
        agent = agent_of(params)
        self.calls.append({"agent": agent, "params": params})
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

    async def complete_json(self, *, system_prompt: str, user_content: str, output_schema: dict,
                            response_model, label: str = "") -> LLMResult:
        params = {"system": [{"type": "text", "text": system_prompt}],
                  "messages": [{"role": "user", "content": user_content}],
                  "output_schema": output_schema, "label": label}
        reply = await self._respond(params)
        text = reply.text if isinstance(reply, AgentReply) else reply
        data = parse_json_output(text, response_model)
        return LLMResult(data=data, usage=usage_to_dict(None), attempts=0, duration_seconds=0.0,
                         model_requested=FAKE_MODEL, response_model="fake_agent", request_id=None,
                         stop_reason=None)

    async def aclose(self):
        self.closed = True

    def calls_for(self, agent: str) -> list[dict]:
        return [c for c in self.calls if c["agent"] == agent]


def repair_object(params: dict) -> dict:
    """Objet à réparer, tel que TRACE l'envoie dans la tâche de réparation."""
    content = params["messages"][0]["content"]
    return json.loads(content.split("<objet>\n", 1)[1].split("\n</objet>", 1)[0])


def repair_turns(params: dict) -> list[dict]:
    """Tours de l'entretien fournis pour la réparation (texte exact)."""
    content = params["messages"][0]["content"]
    return json.loads(content.split("<tours>\n", 1)[1].split("\n</tours>", 1)[0])["turns"]


def repaired(obj: dict) -> AgentReply:
    return text_response({"decision": "corrected", "object": obj, "reason": None})


def withdrawn(reason: str = "aucun passage ne soutient l'objet") -> AgentReply:
    return text_response({"decision": "withdrawn", "object": None, "reason": reason})


def unchanged_repair(params: dict) -> AgentReply:
    """Réparation par défaut du faux modèle : il renvoie l'objet sans l'avoir corrigé (le validateur le signale)."""
    return repaired(repair_object(params))


def fake_settings(**overrides) -> LLMSettings:
    values = {"model": FAKE_MODEL}
    values.update(overrides)
    return LLMSettings(**values)


# --- Faux Ollama et runner local simulé ----------------------------------------------------------------

@dataclass
class OllamaRaw:
    """Réponse brute d'Ollama imposée par un répondeur (ex. `done_reason="length"` : réponse tronquée)."""

    text: str
    done_reason: str = "stop"


def _resolve(handler, params):
    result = handler(params) if callable(handler) and not isinstance(handler, BaseException) else handler
    if inspect.isawaitable(result):
        result = asyncio.run(result)  # appelé dans un fil d'exécution sans boucle (asyncio.to_thread)
    return result


class FakeOllama:
    """Faux Ollama local : `version()`, `models()`, `chat(payload)` (même forme de réponse qu'Ollama)."""

    def __init__(self, responders: dict, *, models=(FAKE_MODEL, DEFAULT_LOCAL_MODEL), available: bool = True):
        self.responders = {k: (list(v) if isinstance(v, list) else v) for k, v in responders.items()}
        self.installed = list(models)
        self.available = available
        self.calls: list[dict] = []

    def _check(self) -> None:
        if not self.available:
            raise LLMError("OLLAMA_UNAVAILABLE", "http://localhost:11434 : faux Ollama arrêté")

    def version(self) -> str:
        self._check()
        return "0.0-test"

    def models(self) -> list[str]:
        self._check()
        return list(self.installed)

    def chat(self, payload: dict, on_progress=None) -> dict:
        self._check()
        if not model_installed(payload["model"], self.installed):
            raise LLMError("MODEL_NOT_FOUND", payload["model"])
        messages = payload["messages"]
        params = {"system": [{"type": "text", "text": messages[0]["content"]}],
                  "messages": [{"role": "user", "content": messages[1]["content"]}],
                  "output_schema": payload["format"], "attempt": (len(messages) - 2) // 2 + 1,
                  "history": messages[2:], "options": payload.get("options")}
        agent = agent_of(params)
        self.calls.append({"agent": agent, "params": params, "payload": payload})
        handler = self.responders.get(agent, unchanged_repair) if agent.startswith(REPAIR) else self.responders[agent]
        if isinstance(handler, list):
            handler = handler.pop(0)
        result = _resolve(handler, params)
        if isinstance(result, BaseException):
            raise result
        done_reason = "stop"
        if isinstance(result, OllamaRaw):
            text, done_reason = result.text, result.done_reason
        elif isinstance(result, AgentReply):
            text = result.text
        else:
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        eval_count = len(text) // 4
        if on_progress is not None:
            on_progress(eval_count)
        # mêmes métriques qu'Ollama (durées en nanosecondes) : 1 000 tokens/s en lecture, 50 tokens/s en génération
        prompt_eval_count = len(messages[1]["content"]) // 4
        return {"model": payload["model"], "message": {"role": "assistant", "content": text}, "done": True,
                "done_reason": done_reason, "prompt_eval_count": prompt_eval_count,
                "prompt_eval_duration": prompt_eval_count * 1_000_000, "eval_count": eval_count,
                "eval_duration": eval_count * 20_000_000, "load_duration": 1_000_000,
                "total_duration": prompt_eval_count * 1_000_000 + eval_count * 20_000_000}

    def calls_for(self, agent: str) -> list[dict]:
        return [c for c in self.calls if c["agent"] == agent]


class FakeLocalAgentRunner(LocalAgentRunner):
    """Le vrai LocalAgentRunner, sur un faux Ollama (`ollama`) : aucun modèle réel, aucun réseau."""

    def __init__(self, responders: dict | None = None, settings: LLMSettings | None = None, *,
                 ollama: FakeOllama | None = None, checker=None, journal_dir=None, **ollama_options):
        from core.agent_checks import AGENT_SPECS
        self.ollama = ollama or FakeOllama(responders or {}, **ollama_options)
        super().__init__(settings or fake_settings(), transport=self.ollama, checker=checker,
                         journal_dir=journal_dir, specs=AGENT_SPECS)

    @property
    def calls(self) -> list[dict]:
        return self.ollama.calls

    def calls_for(self, agent: str) -> list[dict]:
        return self.ollama.calls_for(agent)


def use_fake_runtime(monkeypatch, responders: dict | None = None, *, ollama: FakeOllama | None = None,
                     **ollama_options) -> FakeOllama:
    """Le pipeline local (et donc Streamlit) exécute les agents sur un faux Ollama partagé. Renvoie ce faux
    Ollama (ses `calls` cumulent toutes les étapes)."""
    from core import local_pipeline
    ollama = ollama or FakeOllama(responders or {}, **ollama_options)

    def make_runner(settings, *, journal_dir=None, checker=None):
        return FakeLocalAgentRunner(settings=settings, ollama=ollama, checker=checker, journal_dir=journal_dir)

    monkeypatch.setattr(local_pipeline, "make_runner", make_runner)
    return ollama
