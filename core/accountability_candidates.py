"""Étape 4 — Candidate Episode Builder DÉTERMINISTE (aucun LLM).

Consomme les sorties STABLES de l'étape 3 (pratiques validées, signaux retenus, avertissements
de l'audit des locuteurs) et propose des PAQUETS de matériaux susceptibles de former un
épisode d'accountability. Il ne décide de rien : l'Accountability Episode Builder (LLM) dit
ensuite s'il s'agit d'un épisode, d'une pratique ordinaire ou d'un cas incertain.

Appuis (« ancrages ») : seules les citations VALIDES comptent (vérification exacte refaite ici,
avec core.evidence_validator). Une pratique ou un signal dont toutes les citations valides
proviennent de tours de l'enquêteur (sans `speaker_warning`) n'est jamais un appui : une
question ne fait pas une position de l'enquêté·e.

Signaux :
- DÉCLENCHEURS (`TRIGGER_SIGNAL_TYPES`) : peuvent créer un candidat s'ils sont proches d'une
  pratique (affect nommé, jugement d'un·e enseignant·e ou des pairs, règle, préférence,
  évaluation métadiscursive, formulation normative, autocorrection, reformulation, restriction,
  exception, contraste, mise à distance, contradiction entre tours) ;
- D'APPUI (`SUPPORTING_SIGNAL_TYPES`) : rattachés à un candidat existant, jamais seuls ;
- IGNORÉS : hésitation, intensification, rire, silence, changement de pronom, et tout
  micro-marqueur nu (« euh », « ben », « juste »… : core.signal_selectivity).

Règles de création d'un candidat :
A. proximité : un signal déclencheur est rattaché à la ou aux pratiques les PLUS PROCHES, si
   leurs tours se chevauchent, sont distants d'au plus `PROXIMITY_MAX_GAP` positions, ou
   appartiennent au même échange question/réponse ;
B. contradiction entre tours : `cross_turn_contradiction` est rattachée aux pratiques ancrées
   sur les tours QU'ELLE CITE, même éloignés — jamais aux pratiques situées entre eux ;
C. polarité : un non-usage / refus et l'usage le plus proche portant sur la même tâche
   (`academic_task` normalisée, même domaine) ;
D. même passage : un usage et un non-usage / refus ancrés sur un même tour forment un groupe ;
E. frontière explicite : une citation de la pratique formule une limite (« mais pas »,
   « à ma place », « moi-même », « normalement », « sauf », « seulement »…).
Jamais de relation entre deux passages au seul motif qu'ils parlent du même outil.

Fusion déterministe : deux candidats aux mêmes pratiques sont fusionnés ; un candidat dont les
pratiques sont incluses dans celles d'UN SEUL autre candidat y est rattaché. Rien d'autre.

Les pratiques sans candidat sont « sans marqueur » (`unmarked_practice_ids`) : racontées sans
aucun des indices ci-dessus, elles ne sont pas envoyées au modèle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from core import evidence_validator, interpretation_guard, signal_selectivity
from core.interaction_chunking import estimate_tokens
from core.schemas import SPEAKER_INTERVIEWER

CANDIDATE_BUILDER_VERSION = "1.0"

PROXIMITY_MAX_GAP = 2                  # positions : un tour de question entre deux réponses
MAX_CONTEXT_TURNS_PER_CANDIDATE = 8
MAX_TURN_CHARS = 1200                  # au-delà, le tour est abrégé autour des citations
QUOTE_WINDOW_CHARS = 250
# Seuils d'un appel unique (documentés dans docs/stage4_accountability_episodes.md) : au-delà, la
# requête part quand même en UN appel, mais le manifest le signale (découpage par candidats à prévoir).
SINGLE_CALL_MAX_INPUT_TOKENS = 24_000
SINGLE_CALL_MAX_CANDIDATES = 60

TRIGGER_SIGNAL_TYPES = frozenset({
    "explicit_emotion", "reference_to_teacher_judgment", "reference_to_peer_judgment", "reference_to_rule",
    "preference_statement", "metadiscursive_self_evaluation", "normative_formulation", "self_correction",
    "self_reformulation", "restriction", "exception", "contrast", "cross_turn_contradiction",
    "distancing_from_own_practice",
})
SUPPORTING_SIGNAL_TYPES = frozenset({
    "minimization", "modalization", "generalization", "attribution_to_others", "significant_repetition",
    "vocabulary_shift", "other",
})

USE_LIKE = frozenset({"use", "past_use", "hypothetical"})
NON_USE_LIKE = frozenset({"non_use", "refusal"})

# Frontière formulée dans une citation de la pratique (texte replié : minuscules, sans accents).
BOUNDARY_PATTERNS = {
    "mais pas / mais jamais": r"\bmais (?:pas|jamais|plus)\b",
    "pas pour / jamais pour": r"\b(?:pas|jamais) pour\b",
    "à ma place": r"\ba ma place\b",
    "moi-même": r"\bmoi[- ]meme\b",
    "je (ne) veux pas que": r"\bje (?:ne )?veux pas qu",
    "normalement": r"\bnormalement\b",
    "sauf": r"\bsauf\b",
    "seulement / uniquement / que pour": r"\b(?:seulement|uniquement|que pour|juste pour)\b",
}
_BOUNDARY_RES = {label: re.compile(pattern) for label, pattern in BOUNDARY_PATTERNS.items()}

REASON_NO_VALID_EVIDENCE = "NO_VALID_EVIDENCE"
REASON_INTERVIEWER_ONLY = "INTERVIEWER_ONLY_EVIDENCE"
REASON_MICRO_MARKER = "MICRO_MARKER"
REASON_NOT_A_TRIGGER = "NOT_A_TRIGGER_TYPE"
REASON_NO_NEARBY_PRACTICE = "NO_NEARBY_PRACTICE"

_ARTICLES = frozenset("le la les l un une des de du d mes ses tes mon ma ton ta son sa leurs leur nos vos".split())


@dataclass
class _Candidate:
    practices: set[str]
    signals: set[str] = field(default_factory=set)
    triggers: list[dict] = field(default_factory=list)


def normalize_task(task: str | None) -> str | None:
    """Tâche comparable : repliée, sans articles ni pluriel final (« les plans » ~ « plan »)."""
    if not task:
        return None
    words = re.findall(r"[a-z0-9]+", interpretation_guard.fold(task).replace("'", " "))
    words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _ARTICLES]
    return " ".join(words) or None


def _valid_evidence(item: dict, turns: dict, interview_id: str) -> list[dict]:
    evidence = item.get("evidence", [])
    results = evidence_validator.validate_evidence(evidence, turns, interview_id)
    return [e for e, r in zip(evidence, results) if r["valid"]]


def _interviewee_voice(turn_id: str, turns: dict, warnings: dict) -> bool:
    return turns[turn_id]["speaker"] != SPEAKER_INTERVIEWER or turn_id in warnings


def _exchanges(transcript: dict) -> list[int]:
    """Numéro d'échange question/réponse de chaque tour : un nouvel échange commence à chaque
    tour de l'enquêteur qui suit un tour d'un autre locuteur."""
    numbers, current, previous = [], 0, None
    for turn in transcript["turns"]:
        if turn["speaker"] == SPEAKER_INTERVIEWER and previous != SPEAKER_INTERVIEWER:
            current += 1
        numbers.append(current)
        previous = turn["speaker"]
    return numbers


def _distance(a: set[int], b: set[int]) -> int:
    return min(abs(x - y) for x in a for y in b)


def boundary_matches(quotes: list[str]) -> list[str]:
    folded = interpretation_guard.fold(" ".join(quotes)).replace("'", " ")
    return [label for label, regex in _BOUNDARY_RES.items() if regex.search(folded)]


def excerpt(text: str, quotes: list[str], limit: int = MAX_TURN_CHARS, window: int = QUOTE_WINDOW_CHARS) -> str:
    """Texte du tour, abrégé autour des citations s'il dépasse `limit` (les citations restent exactes)."""
    if len(text) <= limit:
        return text
    spans = []
    for quote in quotes:
        start = text.find(quote)
        if start >= 0:
            spans.append([max(0, start - window), min(len(text), start + len(quote) + window)])
    if not spans:
        return text[:limit] + " […]"
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    parts = [text[s:e] for s, e in merged]
    out = " […] ".join(parts)
    if merged[0][0] > 0:
        out = "[…] " + out
    if merged[-1][1] < len(text):
        out += " […]"
    return out


def build_candidates(transcript: dict, practices: list[dict], signals: list[dict],
                     speaker_warnings: dict | None = None) -> dict:
    """Candidats d'épisodes (ordre de l'entretien) et bilan déterministe. Sans LLM."""
    warnings = speaker_warnings or {}
    interview_id = transcript["interview_id"]
    turns = evidence_validator.index_turns(transcript)
    position = {turn_id: info["position"] for turn_id, info in turns.items()}
    ids = [t["turn_id"] for t in transcript["turns"]]
    exchange = _exchanges(transcript)

    # --- Pratiques : appuis valides, voix de l'enquêté·e ---------------------------------
    practice_by_id: dict[str, dict] = {}
    anchors: dict[str, set[int]] = {}
    valid_quotes: dict[str, list[dict]] = {}
    excluded: list[dict] = []
    for p in practices:
        pid = p["practice_id"]
        practice_by_id[pid] = p
        valid = _valid_evidence(p, turns, interview_id)
        if not valid:
            excluded.append({"practice_id": pid, "reason": REASON_NO_VALID_EVIDENCE})
            continue
        voiced = {position[e["turn_id"]] for e in valid if _interviewee_voice(e["turn_id"], turns, warnings)}
        if not voiced:
            excluded.append({"practice_id": pid, "reason": REASON_INTERVIEWER_ONLY})
            continue
        anchors[pid] = voiced
        valid_quotes[pid] = valid
    eligible = [pid for pid in practice_by_id if pid in anchors]

    # --- Signaux : déclencheurs, d'appui, ignorés ----------------------------------------
    signal_by_id: dict[str, dict] = {}
    signal_anchors: dict[str, set[int]] = {}
    signal_quotes: dict[str, list[dict]] = {}
    ignored: list[dict] = []
    for s in signals:
        sid = s["signal_id"]
        signal_by_id[sid] = s
        stype = s.get("signal_type")
        if stype not in TRIGGER_SIGNAL_TYPES and stype not in SUPPORTING_SIGNAL_TYPES:
            ignored.append({"signal_id": sid, "reason": REASON_NOT_A_TRIGGER})
            continue
        if signal_selectivity.is_bare_micro_marker(s):
            ignored.append({"signal_id": sid, "reason": REASON_MICRO_MARKER})
            continue
        valid = _valid_evidence(s, turns, interview_id)
        if not valid:
            ignored.append({"signal_id": sid, "reason": REASON_NO_VALID_EVIDENCE})
            continue
        voiced = {position[e["turn_id"]] for e in valid if _interviewee_voice(e["turn_id"], turns, warnings)}
        if not voiced:
            ignored.append({"signal_id": sid, "reason": REASON_INTERVIEWER_ONLY})
            continue
        signal_anchors[sid] = voiced
        signal_quotes[sid] = valid

    def nearest(points: set[int], max_gap: int, same_exchange: bool) -> list[str]:
        scored = []
        for pid in eligible:
            d = _distance(points, anchors[pid])
            close = d <= max_gap or (same_exchange and {exchange[i] for i in points} & {exchange[i] for i in anchors[pid]})
            if close:
                scored.append((d, pid))
        if not scored:
            return []
        best = min(d for d, _ in scored)
        return [pid for d, pid in scored if d == best]

    # --- D. groupes d'un même passage : usage et non-usage ancrés sur un même tour -------
    parent = {pid: pid for pid in eligible}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(eligible):
        for b in eligible[i + 1:]:
            pa, pb = practice_by_id[a].get("use_status"), practice_by_id[b].get("use_status")
            opposite = (pa in USE_LIKE and pb in NON_USE_LIKE) or (pa in NON_USE_LIKE and pb in USE_LIKE)
            if opposite and anchors[a] & anchors[b]:
                parent[find(a)] = find(b)
    groups: dict[str, set[str]] = {}
    for pid in eligible:
        groups.setdefault(find(pid), set()).add(pid)

    raw: list[_Candidate] = []
    attached: dict[str, list[str]] = {}  # signal → pratiques
    unattached: list[dict] = []

    # --- A. proximité ; B. contradiction à distance --------------------------------------
    for sid, points in signal_anchors.items():
        signal = signal_by_id[sid]
        if signal.get("signal_type") == "cross_turn_contradiction":
            cited = {position[t] for t in [*signal.get("turn_ids", []), *(e["turn_id"] for e in signal_quotes[sid])]
                     if t in position}
            ends = set()
            for point in cited:
                ends.update(nearest({point}, 1, False))
            if ends:
                raw.append(_Candidate(practices=ends, signals={sid}, triggers=[{
                    "type": "cross_turn_contradiction", "signal_id": sid,
                    "turn_ids": [ids[i] for i in sorted(cited)]}]))
                attached[sid] = sorted(ends)
            else:
                unattached.append({"signal_id": sid, "reason": REASON_NO_NEARBY_PRACTICE})
            continue
        near = nearest(points, PROXIMITY_MAX_GAP, True)
        if near:
            attached[sid] = near
        elif signal.get("signal_type") in TRIGGER_SIGNAL_TYPES:
            unattached.append({"signal_id": sid, "reason": REASON_NO_NEARBY_PRACTICE})
        else:
            ignored.append({"signal_id": sid, "reason": REASON_NOT_A_TRIGGER})

    for members in groups.values():
        trigger_signals = sorted(sid for sid, pids in attached.items() if set(pids) & members
                                 and signal_by_id[sid].get("signal_type") in TRIGGER_SIGNAL_TYPES
                                 and signal_by_id[sid].get("signal_type") != "cross_turn_contradiction")
        triggers = []
        if trigger_signals:
            triggers.append({"type": "signal_proximity", "signal_ids": trigger_signals})
        statuses = {practice_by_id[p].get("use_status") for p in members}
        if len(members) > 1 and statuses & USE_LIKE and statuses & NON_USE_LIKE:
            triggers.append({"type": "same_passage_use_non_use", "practice_ids": sorted(members)})
        for pid in sorted(members):
            labels = boundary_matches([e["quote"] for e in valid_quotes[pid]])
            if labels:
                triggers.append({"type": "boundary_formulation", "practice_id": pid, "patterns": labels})
        if triggers:
            raw.append(_Candidate(practices=set(members), signals=set(trigger_signals), triggers=triggers))

    # --- C. usage / non-usage sur la même tâche ------------------------------------------
    for pid in eligible:
        practice = practice_by_id[pid]
        key = normalize_task(practice.get("academic_task"))
        if practice.get("use_status") not in NON_USE_LIKE or key is None:
            continue
        counterparts = [(_distance(anchors[pid], anchors[o]), o) for o in eligible
                        if o != pid and practice_by_id[o].get("use_status") in USE_LIKE
                        and normalize_task(practice_by_id[o].get("academic_task")) == key
                        and practice_by_id[o].get("practice_domain") == practice.get("practice_domain")
                        and find(o) != find(pid)]
        if counterparts:
            _, other = min(counterparts)
            raw.append(_Candidate(practices={pid, other}, triggers=[{
                "type": "polarity_contrast", "practice_ids": sorted({pid, other}), "task": key}]))

    # --- Fusion déterministe ---------------------------------------------------------------
    merged: dict[frozenset, _Candidate] = {}
    for cand in raw:
        key = frozenset(cand.practices)
        if key in merged:
            merged[key].signals |= cand.signals
            merged[key].triggers.extend(t for t in cand.triggers if t not in merged[key].triggers)
        else:
            merged[key] = _Candidate(set(cand.practices), set(cand.signals), list(cand.triggers))
    changed = True
    while changed:
        changed = False
        for key in sorted(merged, key=len):
            supersets = [k for k in merged if key < k]
            if len(supersets) == 1:
                target = merged[supersets[0]]
                target.signals |= merged[key].signals
                target.triggers.extend(t for t in merged[key].triggers if t not in target.triggers)
                del merged[key]
                changed = True
                break

    # Signaux d'appui (et déclencheurs rattachés) : ajoutés aux candidats de leurs pratiques
    for sid, pids in attached.items():
        for cand in merged.values():
            if set(pids) & cand.practices:
                cand.signals.add(sid)
    used_signals = {sid for cand in merged.values() for sid in cand.signals}
    for sid, pids in attached.items():
        if sid not in used_signals:
            ignored.append({"signal_id": sid, "reason": REASON_NOT_A_TRIGGER})

    # --- Mise en forme ---------------------------------------------------------------------
    def first_point(cand: _Candidate) -> int:
        return min(min(anchors[p]) for p in cand.practices)

    candidates = []
    in_candidate: set[str] = set()
    for number, cand in enumerate(sorted(merged.values(), key=lambda c: (first_point(c), sorted(c.practices))), 1):
        practice_ids = sorted(cand.practices, key=lambda p: (min(anchors[p]), p))
        signal_ids = sorted(cand.signals, key=lambda s: (min(signal_anchors[s]), s))
        in_candidate.update(practice_ids)
        evidence_points = sorted({position[e["turn_id"]] for p in practice_ids for e in valid_quotes[p]}
                                 | {position[e["turn_id"]] for s in signal_ids for e in signal_quotes[s]})
        context = list(evidence_points)
        for point in evidence_points:
            if point > 0 and transcript["turns"][point - 1]["speaker"] == SPEAKER_INTERVIEWER:
                context.append(point - 1)
        context = sorted(set(context))
        if len(context) > MAX_CONTEXT_TURNS_PER_CANDIDATE:
            questions = [p for p in context if p not in evidence_points]
            keep = set(evidence_points) | set(questions[:max(0, MAX_CONTEXT_TURNS_PER_CANDIDATE - len(evidence_points))])
            context = sorted(keep)
        turn_ids = [ids[i] for i in context]
        candidates.append({
            "candidate_id": f"{interview_id}_C{number:03d}",
            "triggers": cand.triggers,
            "trigger_types": sorted({t["type"] for t in cand.triggers}),
            "practice_ids": practice_ids,
            "signal_ids": signal_ids,
            "turn_ids": turn_ids,
            "first_turn_id": ids[min(evidence_points)],
            "last_turn_id": ids[max(evidence_points)],
            "speaker_warning_turn_ids": [t for t in turn_ids if t in warnings],
        })

    unmarked = [pid for pid in eligible if pid not in in_candidate]
    ignored = sorted({(i["signal_id"], i["reason"]) for i in ignored if i["signal_id"] not in used_signals})
    return {
        "builder_version": CANDIDATE_BUILDER_VERSION,
        "candidates": candidates,
        "unmarked_practice_ids": unmarked,
        "excluded_practices": excluded,
        "ignored_signals": [{"signal_id": s, "reason": r} for s, r in ignored],
        "unattached_signals": unattached,
        "summary": {
            "practice_count": len(practices),
            "signal_count": len(signals),
            "candidate_count": len(candidates),
            "practices_in_candidates": len(in_candidate),
            "unmarked_practice_count": len(unmarked),
            "excluded_practice_count": len(excluded),
            "signals_in_candidates": len(used_signals),
            "ignored_signal_count": len(ignored),
            "unattached_signal_count": len(unattached),
            "candidates_by_trigger": {t: sum(t in c["trigger_types"] for c in candidates) for t in sorted(
                {t for c in candidates for t in c["trigger_types"]})},
        },
        # usage interne (payload) : non sérialisé dans les sorties
        "_index": {"practices": practice_by_id, "signals": signal_by_id, "practice_quotes": valid_quotes,
                   "signal_quotes": signal_quotes},
    }


# --- Représentation compacte envoyée au modèle --------------------------------------------

def build_payload(transcript: dict, built: dict, speaker_warnings: dict | None = None) -> dict:
    """Candidats, pratiques, signaux et tours strictement nécessaires (chacun une seule fois)."""
    warnings = speaker_warnings or {}
    index = built["_index"]
    turns_by_id = {t["turn_id"]: t for t in transcript["turns"]}
    order = {t["turn_id"]: i for i, t in enumerate(transcript["turns"])}
    practice_ids = list(dict.fromkeys(p for c in built["candidates"] for p in c["practice_ids"]))
    signal_ids = list(dict.fromkeys(s for c in built["candidates"] for s in c["signal_ids"]))
    turn_ids = sorted({t for c in built["candidates"] for t in c["turn_ids"]}, key=order.get)

    def quotes(items: list[dict]) -> list[dict]:
        return [{"turn_id": e["turn_id"], "quote": e["quote"]} for e in items]

    practices = []
    for pid in practice_ids:
        p = index["practices"][pid]
        practices.append({
            "practice_id": pid, "use_status": p.get("use_status"), "non_use_reason": p.get("non_use_reason"),
            "practice_domain": p.get("practice_domain"), "academic_task": p.get("academic_task"),
            "summary": p.get("summary"), "turn_start": p.get("turn_start"), "turn_end": p.get("turn_end"),
            "stated_reason": p.get("stated_reason") or [], "explicit_constraints": p.get("explicit_constraints") or [],
            "stated_frequency": p.get("stated_frequency"), "quotes": quotes(index["practice_quotes"][pid]),
        })
    signals = []
    for sid in signal_ids:
        s = index["signals"][sid]
        signals.append({
            "signal_id": sid, "signal_type": s.get("signal_type"), "surface_form": s.get("surface_form"),
            "description": s.get("description"), "turn_ids": s.get("turn_ids", []),
            "explicit_affect": s.get("explicit_affect"), "quotes": quotes(index["signal_quotes"][sid]),
        })
    cited: dict[str, list[str]] = {}
    for item in [*(index["practice_quotes"][p] for p in practice_ids), *(index["signal_quotes"][s] for s in signal_ids)]:
        for e in item:
            cited.setdefault(e["turn_id"], []).append(e["quote"])
    turns = []
    for tid in turn_ids:
        turn = turns_by_id[tid]
        item = {"turn_id": tid, "speaker": turn["speaker"], "text": excerpt(turn["text"], cited.get(tid, []))}
        if tid in warnings:
            item["speaker_warning"] = {"suggested_speaker": warnings[tid].get("suggested_speaker"),
                                       "confidence": warnings[tid].get("confidence")}
        turns.append(item)
    candidates = [{"candidate_id": c["candidate_id"], "triggers": c["trigger_types"], "practice_ids": c["practice_ids"],
                   "signal_ids": c["signal_ids"], "turn_ids": c["turn_ids"]} for c in built["candidates"]]
    return {"interview_id": transcript["interview_id"], "candidates": candidates, "practices": practices,
            "signals": signals, "turns": turns}


def serialize_payload(payload: dict) -> str:
    """JSON déterministe, un objet par ligne ; « </ » échappé (le texte ne peut pas fermer la balise)."""
    parts = ['{"interview_id":' + json.dumps(payload["interview_id"], ensure_ascii=False)]
    for key in ("candidates", "practices", "signals", "turns"):
        lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in payload[key]]
        parts.append(f'"{key}":[\n' + ",\n".join(lines) + "\n]")
    return (",\n".join(parts) + "}").replace("</", "<\\/")


def public(built: dict) -> dict:
    """Bilan du constructeur de candidats, sans l'index interne."""
    return {k: v for k, v in built.items() if not k.startswith("_")}


def payload_estimate(payload_json: str) -> int:
    return estimate_tokens(payload_json)
