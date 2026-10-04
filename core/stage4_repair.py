"""Étape 4 — réparation CIBLÉE des épisodes : jamais de régénération d'un bloc dont les autres candidats sont valides.

    réponses des blocs (cache TRACE : chaque bloc validé reste tel quel)
      → 1. identifiants étrangers (déterministe) : un épisode ne garde que les pratiques et signaux de SES candidats
           (`practice_ids` ⊆ pratiques des `candidate_ids`, `signal_ids` ⊆ leurs signaux). Un identifiant d'un autre
           candidat dont l'épisode ne cite aucun tour propre est retiré, et ce retrait est noté sur l'épisode
           (`removed_foreign_ids`, information FOREIGN_ID_REMOVED) ; s'il appuie l'épisode (citation ou opération sur
           un tour qui n'appartient qu'à lui), il n'est pas retiré : l'épisode est réparé (étape 2)
      → 2. réparations par le modèle, au plus UNE par objet, chacune sur les SEULS candidats concernés :
           - épisode avec une anomalie bloquante du validateur (ex. NO_ACCOUNTING_MOVES, citation absente,
             identifiant étranger utile) : l'épisode, les erreurs exactes et la consigne de reclasser sans rien
             inventer (une opération réellement appuyée, sinon `ordinary_practice` ou `uncertain`) ;
           - candidat sans aucune disposition (CANDIDATE_NOT_ADDRESSED) : ce seul candidat, à décider ;
         Message : le gabarit habituel de l'agent (prompt système et schéma inchangés) sur la représentation
         normalisée des seuls candidats concernés, suivi des consignes de réparation. Chaque réponse validée est mise
         en cache (clé : ce message exact) : une reprise ne la redemande pas.
      → 3. une réparation n'est retenue que si elle traite chaque candidat concerné exactement une fois, sans
           anomalie bloquante ; sinon l'épisode d'origine est conservé tel quel (le validateur le signale, rejet) et
           un candidat sans disposition reste une anomalie bloquante (étape 4 PARTIAL).

Les avertissements et informations (ex. SPEAKER_WARNING_PROPAGATED) ne provoquent jamais d'appel.
"""

from __future__ import annotations

import asyncio
import json

from core import accountability_candidates as candidates_mod
from core import accountability_episode_validator as episode_validator

REPAIR_VERSION = "1.0"
MAX_REPAIRS_PER_OBJECT = 1   # jamais une 2e réparation du même épisode ou candidat
KIND_EPISODE, KIND_MISSING = "episode", "missing_candidate"
REPAIRED, UNRESOLVED, FAILED = "repaired", "unresolved", "failed"

REPAIR_NOTE = (" RÉPARATION CIBLÉE : ce message ne porte que sur ce(s) candidat(s) ; les épisodes des autres candidats "
               "de l'entretien sont déjà validés et conservés.")

COMMON_INSTRUCTIONS = """Consignes de la réparation :
- Réponds avec l'objet JSON demandé, qui ne contient QUE les épisodes de ce(s) candidat(s) : chaque candidat figure dans exactement un épisode, avec l'une des dispositions prévues (`accountability_episode`, `ordinary_practice` ou `uncertain`).
- N'utilise que les identifiants de pratiques, de signaux et de tours de ce(s) candidat(s), ci-dessus : jamais ceux d'un autre candidat.
- Un `accountability_episode` exige au moins une opération (`accounting_moves`) réellement appuyée sur une citation d'un tour de l'enquêté·e. Si le matériau n'en montre aucune, n'en invente pas : choisis `ordinary_practice` (aucune opération) ou, si le matériau est ambigu, `uncertain` (`confidence: "low"`, `needs_review: true`).
- Recopie chaque citation MOT POUR MOT depuis le texte d'un tour ci-dessus, avec son `turn_id`."""


# --- 1. Identifiants étrangers (déterministe) -----------------------------------------------------------

def _material(prepared, candidate_ids: list[str]) -> tuple[set, set, set]:
    by_id = {c["candidate_id"]: c for c in prepared.built["candidates"]}
    cands = [by_id[c] for c in candidate_ids if c in by_id]
    return ({p for c in cands for p in c["practice_ids"]}, {s for c in cands for s in c["signal_ids"]},
            {t for c in cands for t in c["turn_ids"]})


def _object_turns(prepared, object_id: str) -> set:
    for practice in prepared.practices:
        if practice["practice_id"] == object_id:
            return {e.get("turn_id") for e in practice.get("evidence", [])}
    for signal in prepared.signals:
        if signal["signal_id"] == object_id:
            return set(signal.get("turn_ids", [])) | {e.get("turn_id") for e in signal.get("evidence", [])}
    return set()


def remove_foreign_ids(prepared, episode: dict) -> tuple[dict, list[str]]:
    """(épisode, identifiants étrangers UTILES). Retire les pratiques / signaux existants qui n'appartiennent à aucun
    candidat de l'épisode et dont l'épisode ne cite aucun tour propre ; les autres restent (à réparer)."""
    allowed_p, allowed_s, cand_turns = _material(prepared, episode.get("candidate_ids", []))
    if not allowed_p and not allowed_s:
        return episode, []  # candidats inconnus : le validateur le signale (UNKNOWN_CANDIDATE_ID)
    known = {p["practice_id"] for p in prepared.practices} | {s["signal_id"] for s in prepared.signals}
    cited = ({e.get("turn_id") for e in episode.get("evidence", [])}
             | {t for m in episode.get("accounting_moves", []) for t in m.get("evidence_turn_ids", [])})
    own_turns = cand_turns | set().union(*(_object_turns(prepared, i) for i in allowed_p | allowed_s))
    removed, needed = [], []
    for key, allowed in (("practice_ids", allowed_p), ("signal_ids", allowed_s)):
        for object_id in episode.get(key, []):
            if object_id in allowed or object_id not in known:
                continue
            (needed if (_object_turns(prepared, object_id) - own_turns) & cited else removed).append(object_id)
    if not removed:
        return episode, needed
    cleaned = {**episode, "practice_ids": [p for p in episode.get("practice_ids", []) if p not in removed],
               "signal_ids": [s for s in episode.get("signal_ids", []) if s not in removed]}
    cleaned["removed_foreign_ids"] = [*episode.get("removed_foreign_ids", []), *removed]
    return cleaned, needed


# --- 2. Réparations ciblées ------------------------------------------------------------------------------

def _validate(prepared, episodes: list[dict], candidate_ids: list[str] | None = None) -> dict:
    candidates = [c for c in prepared.built["candidates"]
                  if candidate_ids is None or c["candidate_id"] in candidate_ids]
    return episode_validator.validate_episodes(episodes, prepared.transcript, prepared.practices, prepared.signals,
                                               candidates, prepared.speaker_warnings)


def blocking_issues(prepared, episode: dict) -> list[str]:
    """Anomalies bloquantes (« error ») d'UN épisode, sur ses seuls candidats ; la couverture se juge ailleurs."""
    report = _validate(prepared, [episode], episode.get("candidate_ids", []))["report"]
    return [f"{i['code']} : {i['message']}" + _detail(i) for i in report["issues"]
            if i["severity"] == episode_validator.ERROR and i["code"] != "CANDIDATE_NOT_ADDRESSED"]


def _detail(issue: dict) -> str:
    keys = ("practice_id", "signal_id", "candidate_id", "turn_id", "move_index", "evidence_index")
    found = [f"{k} {issue[k]}" for k in keys if issue.get(k) is not None]
    return f" ({', '.join(found)})" if found else ""


def _short(value, prefix: str):
    if isinstance(value, str):
        return value.removeprefix(prefix)
    if isinstance(value, list):
        return [_short(v, prefix) for v in value]
    if isinstance(value, dict):
        return {k: _short(v, prefix) if k != "quote" else v for k, v in value.items()}
    return value


def repair_message(prepared, task: dict) -> str:
    """Gabarit habituel de l'agent sur les seuls candidats de la tâche, puis la demande de réparation."""
    from core.accountability import SPEC  # import différé : core.accountability importe ce module

    ids = task["candidate_ids"]
    payload = candidates_mod.build_payload(prepared.transcript, prepared.built, prepared.speaker_warnings, ids)
    message = SPEC.user_template.format(
        interview_id=prepared.interview_id, candidate_count=len(ids), turn_count=len(payload["turns_by_id"]),
        payload_json=candidates_mod.serialize_payload(payload), chunk_note=REPAIR_NOTE)
    prefix = candidates_mod.id_prefix(prepared.interview_id)
    if task["kind"] == KIND_MISSING:
        problem = ("Dans ta réponse précédente, ce candidat n'a reçu AUCUNE disposition (CANDIDATE_NOT_ADDRESSED). "
                   "Décide pour lui, selon tes consignes.")
    else:
        previous = [{k: v for k, v in e.items() if k != "removed_foreign_ids"} for e in task["episodes"]]
        previous_json = json.dumps({"episodes": _short(previous, prefix)}, ensure_ascii=False, indent=1)
        problem = ("Ta réponse précédente pour ce(s) candidat(s) :\n<previous_episodes>\n" + previous_json
                   + "\n</previous_episodes>\nLe validateur de TRACE y relève des anomalies bloquantes :\n"
                   + "\n".join(f"- {_short(p, prefix)}" for p in task["problems"])
                   + "\nCorrige-les et garde le reste de l'épisode tel quel quand c'est possible.")
    return f"{message}\n\n{problem}\n\n{COMMON_INSTRUCTIONS}"


def plan_repairs(prepared, episodes: list[dict], skip_candidate_ids: set) -> list[dict]:
    """Tâches de réparation : un épisode bloquant (ou dont un identifiant étranger est utile), un candidat sans
    disposition. `skip_candidate_ids` : candidats d'un bloc en échec (rejoué en entier à la reprise)."""
    tasks, taken = [], set()
    for episode in episodes:
        cands = [c for c in episode.get("candidate_ids", []) if c not in taken]
        if not cands or set(cands) & skip_candidate_ids:
            continue
        problems = blocking_issues(prepared, episode)  # dont un identifiant étranger utile (…_NOT_IN_CANDIDATES)
        if problems:
            tasks.append({"kind": KIND_EPISODE, "candidate_ids": cands, "episodes": [episode], "problems": problems})
            taken.update(cands)
    addressed = {c for e in episodes for c in e.get("candidate_ids", [])}
    for candidate in prepared.built["candidates"]:
        cid = candidate["candidate_id"]
        if cid not in addressed and cid not in skip_candidate_ids and cid not in taken:
            tasks.append({"kind": KIND_MISSING, "candidate_ids": [cid], "episodes": [],
                          "problems": ["CANDIDATE_NOT_ADDRESSED"]})
            taken.add(cid)
    return tasks


def accept(prepared, task: dict, output: dict) -> tuple[list[dict] | None, str]:
    """Épisodes de la réparation retenus, ou None (avec la raison) : chaque candidat de la tâche exactement une fois,
    aucun autre candidat, aucune anomalie bloquante après le retrait déterministe des identifiants étrangers."""
    wanted = set(task["candidate_ids"])
    episodes = candidates_mod.expand_ids(output, prepared.interview_id)["episodes"]
    if not episodes:
        return None, "aucun épisode dans la réparation"
    cleaned = []
    for episode in episodes:
        cands = episode.get("candidate_ids", [])
        if not cands or not set(cands) <= wanted:
            return None, f"épisode hors des candidats demandés ({', '.join(cands) or 'aucun candidat'})"
        episode, needed = remove_foreign_ids(prepared, episode)
        if needed:
            return None, f"identifiant(s) étranger(s) toujours utilisé(s) : {', '.join(needed)}"
        problems = blocking_issues(prepared, episode)
        if problems:
            return None, problems[0]
        cleaned.append(episode)
    covered = [c for e in cleaned for c in e["candidate_ids"]]
    if sorted(covered) != sorted(wanted):
        return None, "chaque candidat demandé n'est pas traité exactement une fois"
    return cleaned, ""


async def repair_episodes(prepared, episodes: list[dict], call, skip_candidate_ids: set | None = None) -> dict:
    """Applique 1 (déterministe) puis 2-3 (au plus une réparation par objet). `call(request)` : un message au modèle
    via le cache TRACE (core/accountability._call). Renvoie {"episodes", "repairs"} ; l'ordre est celui d'origine, les
    épisodes réparés remplaçant les épisodes fautifs à leur place (un candidat manquant : à la fin)."""
    skip = set(skip_candidate_ids or ())
    episodes = [remove_foreign_ids(prepared, e)[0] for e in episodes]
    tasks = plan_repairs(prepared, episodes, skip)
    requests = []
    for number, task in enumerate(tasks, start=1):
        message = repair_message(prepared, task)
        requests.append({"chunk": f"reparation{number}", "candidate_ids": task["candidate_ids"],
                         "user_message": message, "estimated_tokens": candidates_mod.payload_estimate(message)})
    records = list(await asyncio.gather(*(call(r) for r in requests)))
    replaced: dict[int, list[dict]] = {}
    appended: list[dict] = []
    repairs = []
    for task, request, record in zip(tasks, requests, records):
        row = {"kind": task["kind"], "candidate_ids": task["candidate_ids"], "problems": task["problems"],
               "status": record["status"], "cache_hit": record["cache_hit"], "request_id": record["request_id"],
               "estimated_input_tokens": request["estimated_tokens"], "usage": record["usage"],
               "duration_seconds": record["duration_seconds"], "error": record["error"]}
        if record["output"] is None:
            row.update(outcome=FAILED, reason=(record["error"] or {}).get("message"))
        else:
            kept, reason = accept(prepared, task, record["output"])
            if kept is None:
                row.update(outcome=UNRESOLVED, reason=reason)
            else:
                row.update(outcome=REPAIRED, reason=None)
                annotation = {"kind": task["kind"], "problems": task["problems"], "repair_version": REPAIR_VERSION}
                kept = [{**e, "trace_repair": annotation} for e in kept]
                if task["kind"] == KIND_EPISODE:
                    replaced[next(i for i, e in enumerate(episodes) if e is task["episodes"][0])] = kept
                else:
                    appended.extend(kept)
        repairs.append(row)
    final = []
    for index, episode in enumerate(episodes):
        final.extend(replaced.get(index, [episode]))
    return {"episodes": final + appended, "repairs": repairs}
