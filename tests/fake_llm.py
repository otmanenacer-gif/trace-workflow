"""Agents simulés pour les tests : aucune connexion réseau, aucun SDK, aucune clé.

`FakeAgents` remplace, dans les orchestrateurs des étapes 3 à 6, le client des agents du workflow Claude Code
(core.claude_code_workflow.WorkflowClient) : même interface (`complete_json`, `aclose`), même validation de la
réponse (core.llm_client.parse_json_output : INVALID_JSON, SCHEMA_VALIDATION…) et même comptabilité (0 appel API,
usage nul), mais la réponse de chaque agent est fournie par le test au lieu d'être écrite par Claude Code.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass

from core.llm_client import LLMError, LLMResult, LLMSettings, parse_json_output, usage_to_dict

PRACTICE = "practice_extractor"
INTERACTION = "interaction_signal_reader"
AUDITOR = "speaker_attribution_auditor"
LONG_DISTANCE = "interaction_signal_reader_long_distance"  # lecture à longue distance (entretien long)
ACCOUNTABILITY = "accountability_episode_builder"  # étape 4 : épisodes d'accountability
TRAJECTORY = "trajectory_mapper"  # étape 5 : configuration et trajectoire intra-entretien
COMPARATOR = "cross_interview_comparator"  # étape 6 : comparaison inter-entretiens

FAKE_MODEL = "fake-model"  # étiquette du moteur dans les manifests des tests (champ « model »)


@dataclass
class AgentReply:
    """Réponse d'un agent simulé : le texte que l'agent écrirait dans response.json."""

    text: str


def text_response(payload) -> AgentReply:
    return AgentReply(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))


def agent_error(code: str = "SCHEMA_VALIDATION", detail: str | None = None) -> LLMError:
    """Échec d'un agent (réponse non conforme) : à renvoyer ou lever par un répondeur."""
    return LLMError(code, detail)


def agent_of(params: dict) -> str:
    properties = params["output_schema"]["properties"]
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


def fake_settings(**overrides) -> LLMSettings:
    values = {"model": FAKE_MODEL}
    values.update(overrides)
    return LLMSettings(**values)


def answering_workflow(agents: FakeAgents):
    """`WorkflowClient.complete_json` où chaque tâche sans réponse est jouée aussitôt par `agents` : la réponse est
    écrite dans response.json (là où un sous-agent Claude Code l'écrirait), puis lue et validée par le vrai
    workflow. Un répondeur qui échoue (LLMError) écrit une réponse non conforme : la tâche reste « à corriger ».

    Usage : monkeypatch.setattr(WorkflowClient, "complete_json", answering_workflow(agents)).
    """
    from core import claude_code_workflow as wf

    # le vrai `complete_json`, même si un remplacement précédent est en place (relance du script Streamlit)
    original = getattr(wf.WorkflowClient.complete_json, "workflow_original", wf.WorkflowClient.complete_json)

    async def complete_json(self, *, system_prompt, user_content, output_schema, response_model, label):
        task_id = wf.task_id_for(label, system_prompt, user_content, output_schema)
        response = self.tasks_root / task_id / wf.RESPONSE_FILENAME
        if not response.is_file():
            params = {"system": [{"type": "text", "text": system_prompt}],
                      "messages": [{"role": "user", "content": user_content}],
                      "output_schema": output_schema, "label": label}
            try:
                reply = await agents._respond(params)
                text = reply.text if isinstance(reply, AgentReply) else reply
            except LLMError:
                text = "{}"  # réponse non conforme au schéma : SCHEMA_VALIDATION
            response.parent.mkdir(parents=True, exist_ok=True)
            response.write_text(text, encoding="utf-8")
        return await original(self, system_prompt=system_prompt, user_content=user_content,
                              output_schema=output_schema, response_model=response_model, label=label)

    complete_json.workflow_original = original
    return complete_json
