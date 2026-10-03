"""Réparation CIBLÉE des réponses de l'étape 3 : seuls les objets fautifs sont redemandés au modèle.

Agents concernés : Practice Extractor, Interaction Signal Reader et sa lecture à longue distance.

    réponse du modèle (JSON lisible)
      → objets conformes au schéma ET sans anomalie bloquante du validateur : conservés tels quels, jamais redemandés
      → chaque objet fautif : UNE tâche de réparation minimale (prompt système de l'agent, inchangé ; l'objet ; les
        erreurs exactes du validateur ; les seuls tours de l'entretien utiles ; son numéro ; le schéma de l'objet)
      → le modèle rend l'objet corrigé, ou le retire si aucun passage ne le soutient (règle de correction existante :
        « un objet qui n'est pas appuyé par le matériau est retiré »)
      → fusion dans l'ordre d'origine → validation complète habituelle.

Classes d'erreurs :
- A, locale réparable : citation absente ou non littérale, tour inconnu, intervalle invalide, aucune preuve valide,
  champ non conforme au schéma dans un objet → réparation de cet objet seulement ;
- B, locale non réparable : le modèle retire l'objet faute de passage qui le soutienne, ou l'objet reste fautif après
  les tentatives de réparation → l'objet D'ORIGINE (jamais une réparation ratée) est conservé s'il est conforme au
  schéma, et le validateur habituel le signale (citation invalide, needs_review) ; un objet d'origine non conforme
  au schéma est écarté (il ne peut pas figurer dans la sortie) ;
- C, globale : JSON illisible, racine non conforme, réponse tronquée → nouvelle génération complète (mécanisme existant).

Rien n'est inventé : les tours fournis sont ceux de l'entretien (et du bloc envoyé), le texte est exact, et la
réponse réparée passe par les mêmes validateurs. Les prompts, schémas, validateurs et le découpage ne changent pas.

État persistant (reprise) : après la génération initiale puis après chaque réparation, l'état des objets (validés,
réparés, retirés, à traiter) est écrit dans <journal>/partial/<appel>.json ; une reprise ne refait ni la génération
initiale ni une réparation terminée. Le fichier est supprimé quand l'appel aboutit.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from agents.base import AgentSpec
from core.analysis_cache import write_json_atomic

PRACTICE_AGENT = "practice_extractor"
INTERACTION_AGENT = "interaction_signal_reader"
LONG_DISTANCE_AGENT = "interaction_signal_reader_long_distance"
REPAIRABLE_AGENTS = (PRACTICE_AGENT, INTERACTION_AGENT, LONG_DISTANCE_AGENT)

# Réponse maximale d'une réparation : un objet (une pratique ≈ 330 tokens, un signal ≈ 150), avec marge.
REPAIR_RESERVE = {PRACTICE_AGENT: 1536, INTERACTION_AGENT: 1024, LONG_DISTANCE_AGENT: 1024}
REPAIR_MAX_TURNS = 14
REPAIR_MAX_CHARS = 9000
STATE_VERSION = "1"

# États d'un objet
VALID, PENDING, REPAIRED, WITHDRAWN, UNRESOLVED, DROPPED = (
    "valid", "pending", "repaired", "withdrawn", "unresolved", "dropped")
KEPT = (VALID, REPAIRED, UNRESOLVED)
DECISIONS = ("corrected", "withdrawn")

REPAIR_INSTRUCTIONS = """Consignes de la réparation :
- Renvoie UNIQUEMENT cet objet, corrigé, dans "object", avec "decision": "corrected", en respectant tes consignes et
  le schéma de l'objet. Ne renvoie aucun autre objet : les autres objets de ta réponse sont déjà validés et conservés.
- Corrige ce que les erreurs signalent (citation, identifiant de tour, intervalle, champ) et garde le reste de l'objet
  tel quel, sauf ce que la correction exige (par exemple surface_form, turn_ids ou turn_start / turn_end alignés sur
  la citation corrigée).
- Une citation se recopie MOT POUR MOT, caractère pour caractère (apostrophes, ponctuation et casse comprises), depuis
  le texte d'UN tour ci-dessus, et son turn_id est celui de ce tour.
- N'utilise que les turn_id des tours ci-dessus. N'invente aucun tour, aucune citation, aucun contenu.
- Si aucun passage de ces tours ne soutient réellement l'objet, ne force pas de citation : renvoie
  "decision": "withdrawn", "object": null, et la raison dans "reason"."""


def repairable(spec: AgentSpec | None) -> bool:
    return spec is not None and spec.name in REPAIRABLE_AGENTS


def repair_schema(spec: AgentSpec) -> dict:
    """Schéma de la réponse d'une réparation : la décision et l'objet attendu (schéma de l'objet de l'agent)."""
    root = spec.output_schema
    item = root["properties"][spec.items_key]["items"]
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"decision": {"type": "string", "enum": list(DECISIONS)},
                             "object": {"anyOf": [item, {"type": "null"}]},
                             "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
              "required": ["decision", "object", "reason"]}
    if "$defs" in root:
        schema["$defs"] = root["$defs"]
    return schema


def _schema_lines(exc: ValidationError, prefix_len: int) -> list[str]:
    # emplacements et types d'erreur seulement : jamais les valeurs (extraits d'entretien)
    return [f"SCHEMA_VALIDATION — {'.'.join(str(p) for p in e['loc'][prefix_len:]) or '(objet)'} : {e['type']}"
            for e in exc.errors(include_url=False, include_context=False, include_input=False)][:20]


def item_schema_problems(item, root: dict, spec: AgentSpec, response_model) -> list[str]:
    """Anomalies de schéma d'UN objet (validé seul, avec les champs racine de la réponse)."""
    if not isinstance(item, dict):
        return ["SCHEMA_VALIDATION — (objet) : object_type"]
    try:
        response_model.model_validate({**root, spec.items_key: [item]})
    except ValidationError as exc:
        return _schema_lines(exc, 2)
    return []


def split_response(text: str, spec: AgentSpec, response_model) -> dict | None:
    """Réponse lisible → {"root", "items", "schema_problems" (index → anomalies)} ; None si l'erreur est globale
    (JSON illisible, racine absente ou non conforme) : seule une nouvelle génération complète y répond."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    key = spec.items_key
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        return None
    root = {k: v for k, v in payload.items() if k != key}
    try:
        response_model.model_validate({**root, key: []})
    except ValidationError:
        return None
    items = payload[key]
    problems = {}
    for index, item in enumerate(items):
        found = item_schema_problems(item, root, spec, response_model)
        if found:
            problems[index] = found
    return {"root": root, "items": items, "schema_problems": problems}


def repair_message(spec: AgentSpec, index: int, total: int, item, problems: list[str], turns_json: str) -> str:
    lines = "\n".join(f"- {p}" for p in problems[:20])
    return (f"RÉPARATION CIBLÉE — {spec.label}, objet n° {index + 1} sur {total} de ta réponse précédente.\n"
            "Cet objet ne peut pas être accepté par TRACE. Erreurs du validateur :\n"
            f"{lines}\n\n<objet>\n{json.dumps(item, ensure_ascii=False)}\n</objet>\n\n"
            "Tours de l'entretien disponibles pour cette réparation (texte exact) :\n"
            f"<tours>\n{turns_json}\n</tours>\n\n{REPAIR_INSTRUCTIONS}")


def parse_repair(text: str, spec: AgentSpec, root: dict, response_model) -> dict:
    """{"decision": corrected | withdrawn | invalid, "object", "reason", "problems"}."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"decision": "invalid", "object": None, "reason": None,
                "problems": ["INVALID_JSON — réponse de réparation illisible"]}
    if not isinstance(payload, dict) or payload.get("decision") not in DECISIONS:
        return {"decision": "invalid", "object": None, "reason": None,
                "problems": ["SCHEMA_VALIDATION — decision : corrected ou withdrawn attendu"]}
    if payload["decision"] == "withdrawn":
        return {"decision": "withdrawn", "object": None, "reason": payload.get("reason"), "problems": []}
    obj = payload.get("object")
    problems = item_schema_problems(obj, root, spec, response_model)
    return {"decision": "corrected" if not problems else "invalid", "object": obj,
            "reason": payload.get("reason"), "problems": problems}


# --- État persistant d'un appel ----------------------------------------------------------------------

def state_path(journal_dir: Path | None, call_id: str) -> Path | None:
    return Path(journal_dir) / "partial" / f"{call_id}.json" if journal_dir else None


def load_state(path: Path | None, call_id: str, model: str) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if state.get("state_version") != STATE_VERSION or state.get("call_id") != call_id or state.get("model") != model:
        return None
    return state


def save_state(path: Path | None, state: dict) -> None:
    if path is not None:
        write_json_atomic(path, state)


def drop_state(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def merged_output(state: dict, spec: AgentSpec) -> dict:
    """Objets conservés (validés, réparés, ou fautifs conformes au schéma), dans l'ordre d'origine."""
    return {**state["root"], spec.items_key: [i["object"] for i in state["items"] if i["status"] in KEPT]}


def citation_summary(state: dict) -> dict:
    """Citations invalides avant réparation, réparées (objet validé), retirées faute de preuve, encore invalides."""
    out = {"citations_invalid_initial": 0, "citations_repaired": 0, "citations_withdrawn": 0,
           "citations_invalid_after_repair": 0}
    for item in state["items"]:
        initial = item.get("invalid_citations_initial") or 0
        out["citations_invalid_initial"] += initial
        if item["status"] == REPAIRED:
            out["citations_repaired"] += initial
        elif item["status"] in (WITHDRAWN, DROPPED):
            out["citations_withdrawn"] += initial
        elif item["status"] == UNRESOLVED:
            out["citations_invalid_after_repair"] += item.get("invalid_citations") or 0
    return out
