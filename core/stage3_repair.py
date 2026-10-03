"""Réparation des réponses de l'étape 3 : jamais de régénération complète d'une réponse lisible, des corrections
DÉTERMINISTES d'abord, au plus UNE réparation par le modèle et par objet, et seulement quand le déterminisme ne peut
rien.

Agents concernés : Practice Extractor, Interaction Signal Reader et sa lecture à longue distance.

    réponse du modèle (JSON lisible)
      → objets conformes au schéma ET sans anomalie bloquante : conservés tels quels, jamais redemandés
      → citations invalides : CitationResolver (core/citation_resolver.py), sans modèle — passage littéral restauré
        quand une correspondance textuelle sûre et unique existe dans le matériau envoyé ; sinon la citation reste
        telle quelle (ambiguë ou introuvable)
      → objet dont il ne reste QUE des anomalies de citation : conservé tel quel ; le validateur habituel le signale
        (citation invalide, needs_review) et les étapes suivantes l'écartent faute de preuve valide (règle existante,
        NO_VALID_EVIDENCE) — aucun appel au modèle, aucun objet sauvé artificiellement
      → autre anomalie locale (intervalle, champ hors schéma, turn_ids inexistant…) : UNE réparation par le modèle
        (prompt système de l'agent inchangé, l'objet, les erreurs exactes, les seuls tours utiles, le schéma de l'objet) ;
        si elle échoue, l'objet garde sa version déterministe (jamais une réparation ratée) ou, hors schéma, est écarté
      → fusion dans l'ordre d'origine → validation complète habituelle (formats de sortie inchangés).

Seule une erreur GLOBALE (JSON illisible, racine non conforme, réponse tronquée) donne lieu à une nouvelle génération
complète. Corrections, décisions et réparations sont journalisées et enregistrées au fur et à mesure dans
<journal>/partial/<appel>.json : une reprise ne les recalcule pas. Le fichier est supprimé quand l'appel aboutit.
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
STATE_VERSION = "2"
MAX_LLM_REPAIRS_PER_OBJECT = 1   # jamais une 2e réparation du même objet

# Anomalies bloquantes qui portent sur une citation (et non sur un champ ou un intervalle)
CITATION_CODES = {"UNKNOWN_TURN_ID", "FOREIGN_INTERVIEW_TURN", "EMPTY_QUOTE", "QUOTE_NOT_FOUND", "QUOTE_NOT_EXACT",
                  "NO_VALID_EVIDENCE"}

# États d'un objet
VALID, RESOLVED, PENDING, REPAIRED, WITHDRAWN, UNRESOLVED, DROPPED = (
    "valid", "resolved", "pending", "repaired", "withdrawn", "unresolved", "dropped")
KEPT = (VALID, RESOLVED, REPAIRED, UNRESOLVED)
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
    """Bilan qualité d'un appel : citations invalides initiales, corrigées de façon déterministe, ambiguës,
    introuvables ; objets rejetés faute de preuve ; réparations par le modèle ; citations invalides finales."""
    from core import citation_resolver as cr
    out = {"citations_invalid_initial": 0, "citations_fixed_deterministic": 0, "citations_ambiguous": 0,
           "citations_not_found": 0, "objects_rejected_no_evidence": 0, "llm_repairs_attempted": 0,
           "llm_repairs_succeeded": 0, "citations_invalid_final": 0, "objects_dropped_schema": 0}
    for item in state["items"]:
        out["citations_invalid_initial"] += item.get("invalid_citations_initial") or 0
        for entry in item.get("citation_log") or []:
            decision = entry["decision"]
            if decision in (cr.FIXED_IN_TURN, cr.FIXED_OTHER_TURN):
                out["citations_fixed_deterministic"] += 1
            elif decision == cr.AMBIGUOUS:
                out["citations_ambiguous"] += 1
            else:
                out["citations_not_found"] += 1
        out["llm_repairs_attempted"] += item.get("repairs") or 0
        out["llm_repairs_succeeded"] += item["status"] == REPAIRED
        if item["status"] in KEPT:
            out["citations_invalid_final"] += item.get("invalid_citations") or 0
            # aucune citation valide : l'objet reste signalé et les étapes suivantes l'écartent (NO_VALID_EVIDENCE)
            out["objects_rejected_no_evidence"] += bool(item.get("citations")) and \
                (item.get("invalid_citations") or 0) >= item["citations"]
        elif item["status"] == WITHDRAWN:
            out["objects_rejected_no_evidence"] += 1
        elif item["status"] == DROPPED:
            out["objects_dropped_schema"] += 1
    return out
