"""Étape 5 — préparation DÉTERMINISTE de la configuration intra-entretien (aucun LLM).

Consomme les sorties STABLES de l'étape 4 (accountability_episodes.json) et, pour vérifier un
ancrage temporel ou contextuel, le minimum de l'étape 3 (tâche, domaine et statut d'usage des
pratiques, citations validées, type des signaux) et du transcript (texte des tours cités). Il ne
conclut rien : il prépare une représentation compacte d'UN entretien pour le Trajectory Mapper
(agents/trajectory_mapper.py), puis sert d'index au validateur (core/trajectory_validator.py).

Éléments retenus (« items ») :
- les épisodes `usable_for_next_stages: true` (jamais un épisode rejeté) — statut, pratiques,
  opérations (type + tours), frontières, références au rôle d'étudiant·e et à autrui, intervalle,
  confiance, `needs_review` et `review_reasons`, citations valides, avertissements de locuteur ;
- les pratiques sans marqueur de l'étape 4 (`unmarked_practices`) avec leurs citations valides de
  l'enquêté·e : elles ne sont jamais passées par le modèle de l'étape 4 et forment souvent les
  zones « ordinaires ».

Ancrages temporels (`TEMPORAL_PATTERNS`) : expressions explicites (« au lycée », « l'année
dernière », « maintenant », « je ne … plus », « depuis »…) repérées dans le texte des tours de
l'ENQUÊTÉ·E cités par un item (jamais dans une question de l'enquêteur), à au plus
`ANCHOR_MAX_GAP_CHARS` caractères d'une citation de cet item. Le texte retenu est une copie exacte
du tour. L'ordre des tours n'est JAMAIS un ancrage.

Régularités (`regularities`, sans conclusion sociologique) : opérations répétées, frontières
répétées, mêmes tâches, matériau « règle + cas », contrastes d'usage / non-usage sur une même tâche,
contradictions déjà signalées par l'étape 3, pratiques ordinaires.

Représentation envoyée (`build_payload`) : NORMALISÉE comme à l'étape 4 — chaque citation, pratique
et ancrage une seule fois (`quotes_by_id`, `practices_by_id`, `anchors_by_id`), préfixe commun des
identifiants omis (`E003`, `P012`, `T0040`) ; `expand_ids` le rétablit dans la réponse.
"""

from __future__ import annotations

import json
import re

from core import evidence_validator, interpretation_guard
from core.accountability_candidates import NON_USE_LIKE, USE_LIKE, normalize_task
from core.interaction_chunking import estimate_tokens
from core.schemas import SPEAKER_INTERVIEWEE, SPEAKER_INTERVIEWER

PREPROCESSOR_VERSION = "1.0"
PAYLOAD_FORMAT_VERSION = "1"

ANCHOR_MAX_GAP_CHARS = 200     # même ordre de grandeur que la proximité intra-tour de l'étape 4.1
SENTENCE_MAX_CHARS = 240
# Seuil d'un appel unique (un seul appel par entretien) : au-delà, l'appel a lieu quand même, signalé.
SINGLE_CALL_MAX_INPUT_TOKENS = 24_000
MIN_ITEMS_FOR_LLM = 2          # en dessous, aucune configuration ne peut être décrite : aucun appel

# --- Ancrages temporels ------------------------------------------------------------------------------
# Texte replié (minuscules, sans accents, apostrophes droites ; une apostrophe des motifs accepte aussi une
# espace, sauf \x27 dans une classe de caractères). kind : period (période nommée, avec un rang
# biographique), past, present, transition (le changement est dit : « je ne … plus », « j'ai arrêté »).
_AP = "[' ]"
PERIOD_PATTERNS = {
    r"au college": 1,
    r"au lycee|en (?:seconde|2nde)|en premiere(?! annee)|en terminale|avant le bac": 2,
    r"en (?:classe )?prepa|en (?:premiere|1re|1ere) annee|en l1": 3,
    r"en (?:deuxieme|2e|2eme) annee|en l2": 4,
    r"en (?:troisieme|3e|3eme) annee|en l3": 5,
    r"en master|en m1|en m2": 6,
}
TEMPORAL_PATTERNS = {
    "past": (r"avant(?=\s*[,;.]|\s+(?:je|j'|on|ca|c'|il|elle|quand|que)\b)|a l'epoque|a cette epoque|auparavant|"
             r"autrefois|l'annee (?:derniere|passee)|l'an dernier|(?:le|au) semestre dernier|"
             r"il y a (?:\w+ )?(?:ans?|annees?|mois)|quand j'etais|plus jeune|au debut\b|a ce moment[- ]la"),
    "present": (r"maintenant|aujourd'hui|desormais|dorenavant|cette annee|ce semestre|actuellement|"
                r"en ce moment|a present|ces temps[- ]ci|depuis"),
    "transition": (r"n(?:e |')(?:(?!pas\b)[\w\x27]+ ){0,2}?plus\b|plus jamais|j'ai (?:arrete|cesse)|"
                   r"j'ai commence a|je me suis mise? a|de plus en plus|de moins en moins|au fur et a mesure|"
                   r"petit a petit|(?:ca|cela|tout) a (?:change|evolue)|j'ai (?:change|evolue)"),
}
_TEMPORAL_RES = ([("period", rank, re.compile(r"\b(?:" + p.replace("'", _AP) + r")")) for p, rank in
                  PERIOD_PATTERNS.items()]
                 + [(kind, None, re.compile(r"\b(?:" + p.replace("'", _AP) + r")"))
                    for kind, p in TEMPORAL_PATTERNS.items()])

# Règle / cas (texte replié des citations de l'enquêté·e)
RULE_MARKERS = re.compile(r"\b(?:normalement|jamais|toujours|d[' ]habitude|en general|je m[' ]interdis|pas question|"
                          r"je refuse|je (?:ne )?veux pas|seulement|uniquement|que pour|juste pour)\b")
CASE_MARKERS = re.compile(r"\b(?:une fois|une seule fois|pour une fois|sauf|exceptionnellement|cas particulier|"
                          r"il m[' ]est arrive|ca m[' ]est arrive|ca m[' ]arrive)\b")
CASE_FREQUENCY = re.compile(r"\b(?:une fois|une seule fois|exceptionnel\w*)\b")
RULE_MOVES = frozenset({"general_rule", "refusal", "restriction", "preference"})
CASE_MOVES = frozenset({"exception"})
CONTRADICTION_SIGNAL = "cross_turn_contradiction"


def fold_with_map(text: str) -> tuple[str, list[int]]:
    """Texte replié (interpretation_guard.fold) et, pour chaque caractère replié, sa position d'origine."""
    folded, origin = [], []
    for index, char in enumerate(text):
        for out in interpretation_guard.fold(char):
            folded.append(out)
            origin.append(index)
    return "".join(folded), origin


def find_temporal(text: str) -> list[dict]:
    """Expressions temporelles d'un texte : {text (copie exacte), start, end, kind, rank}."""
    folded, origin = fold_with_map(text)
    found = []
    for kind, rank, regex in _TEMPORAL_RES:
        for match in regex.finditer(folded):
            if match.end() <= match.start():
                continue
            start, end = origin[match.start()], origin[match.end() - 1] + 1
            if any(f["start"] < end and start < f["end"] for f in found):
                continue  # chevauchement : la première forme (la plus spécifique) l'emporte
            found.append({"text": text[start:end], "start": start, "end": end, "kind": kind, "rank": rank})
    return sorted(found, key=lambda f: f["start"])


def anchor_kind(anchor_text: str) -> tuple[str, int | None] | None:
    """Type d'une expression citée comme ancrage (None si elle ne contient aucune expression temporelle)."""
    found = find_temporal(anchor_text)
    if not found:
        return None
    best = next((f for f in found if f["kind"] == "transition"), found[0])
    return best["kind"], best["rank"]


def ordering_basis(anchors: list[tuple[str, int | None]]) -> str | None:
    """Les ancrages permettent-ils d'ORDONNER deux états ? (None sinon)."""
    kinds = {k for k, _ in anchors}
    ranks = {r for k, r in anchors if k == "period" and r is not None}
    if "transition" in kinds:
        return "transition"
    if "present" in kinds and kinds & {"past", "period"}:
        return "past_present"
    if len(ranks) >= 2:
        return "ordered_periods"
    return None


def _sentence(text: str, start: int, end: int) -> str:
    """Phrase (copie exacte) contenant l'expression, bornée à SENTENCE_MAX_CHARS autour d'elle."""
    left = max((m.end() for m in re.finditer(r"[.!?…]\s+|\n", text[:start])), default=0)
    right_match = re.search(r"[.!?…](?=\s|$)|\n", text[end:])
    right = end + right_match.end() if right_match else len(text)
    if right - left > SENTENCE_MAX_CHARS:
        half = SENTENCE_MAX_CHARS // 2
        left, right = max(left, start - half), min(right, end + half)
        while 0 < left < start and not text[left - 1].isspace():
            left += 1
        while end < right < len(text) and not text[right].isspace():
            right -= 1
    return text[left:right].strip()


def _quote_spans(text: str, quotes: list[str]) -> list[tuple[int, int]]:
    spans = []
    for quote in quotes:
        start = text.find(quote)
        if start >= 0:
            spans.append((start, start + len(quote)))
    return spans


def _near(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(max(0, max(s[0], span[0]) - min(s[1], span[1])) <= ANCHOR_MAX_GAP_CHARS for s in spans)


def _fold_boundary(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", interpretation_guard.fold(text)))


# --- Construction des items ------------------------------------------------------------------------

def build_material(transcript: dict, episodes_doc: dict, practices: list[dict], signals: list[dict],
                   speaker_warnings: dict | None = None) -> dict:
    """Représentation complète (identifiants entiers) d'UN entretien pour l'étape 5. Aucun LLM."""
    warnings = speaker_warnings or {}
    interview_id = transcript["interview_id"]
    turns = evidence_validator.index_turns(transcript)
    practice_by_id = {p["practice_id"]: p for p in practices}
    signal_type = {s["signal_id"]: s.get("signal_type") for s in signals}

    def voiced(turn_id: str | None) -> bool:
        return bool(turn_id) and turn_id in turns and (
            turns[turn_id]["speaker"] != SPEAKER_INTERVIEWER or turn_id in warnings)

    def valid_quotes(evidence: list[dict]) -> list[dict]:
        results = evidence_validator.validate_evidence(evidence, turns, interview_id)
        return [{"turn_id": e["turn_id"], "quote": e["quote"]} for e, r in zip(evidence, results) if r["valid"]]

    def practice_facts(pid: str) -> dict:
        p = practice_by_id.get(pid) or {}
        return {"use_status": p.get("use_status"), "academic_task": p.get("academic_task"),
                "task_key": normalize_task(p.get("academic_task")), "practice_domain": p.get("practice_domain"),
                "discipline": p.get("discipline"), "stated_frequency": p.get("stated_frequency"),
                "stated_reason": p.get("stated_reason") or [], "assessment_context": p.get("assessment_context")}

    items: dict[str, dict] = {}
    excluded = []
    for episode in episodes_doc.get("episodes", []):
        if not episode.get("usable_for_next_stages"):
            excluded.append({"episode_id": episode["episode_id"], "validation_status": episode.get("validation_status"),
                             "review_reasons": episode.get("review_reasons", [])})
            continue
        quotes = valid_quotes(episode.get("evidence", []))
        moves = [{"type": m.get("type"), "turn_ids": [t for t in m.get("evidence_turn_ids", []) if t in turns]}
                 for m in episode.get("accounting_moves", [])]
        practice_quotes = [q for pid in episode.get("practice_ids", []) if pid in practice_by_id
                           for q in valid_quotes(practice_by_id[pid].get("evidence", []))]
        items[episode["episode_id"]] = {
            "id": episode["episode_id"], "kind": "episode", "status": episode.get("episode_status"),
            "practice_ids": [p for p in episode.get("practice_ids", []) if p in practice_by_id],
            "signal_types": sorted({signal_type[s] for s in episode.get("signal_ids", []) if s in signal_type}),
            "moves": moves, "boundaries": list(episode.get("boundary_objects") or []),
            "problem": episode.get("accountability_problem"), "summary": episode.get("episode_summary"),
            "role": episode.get("student_role_reference"), "external": episode.get("external_reference"),
            "turn_start": episode.get("turn_start"), "turn_end": episode.get("turn_end"),
            "confidence": episode.get("confidence"), "needs_review": bool(episode.get("needs_review")),
            "review_reasons": list(episode.get("review_reasons", [])),
            "speaker_warning_turns": [w["turn_id"] for w in episode.get("speaker_warnings", [])],
            "quotes": quotes, "practice_quotes": practice_quotes,
        }
    for unmarked in episodes_doc.get("unmarked_practices", []):
        pid = unmarked["practice_id"]
        if pid not in practice_by_id:
            continue
        quotes = [q for q in valid_quotes(practice_by_id[pid].get("evidence", [])) if voiced(q["turn_id"])]
        if not quotes:
            continue  # sans citation valide de l'enquêté·e, une pratique n'appuie rien
        items[pid] = {
            "id": pid, "kind": "unmarked", "status": "ordinary_practice", "practice_ids": [pid], "signal_types": [],
            "moves": [], "boundaries": [], "problem": None, "summary": unmarked.get("summary"), "role": None,
            "external": None, "turn_start": unmarked.get("turn_start"), "turn_end": unmarked.get("turn_end"),
            "confidence": None, "needs_review": False, "review_reasons": [],
            "speaker_warning_turns": [t for t in {q["turn_id"] for q in quotes} if t in warnings],
            "quotes": quotes, "practice_quotes": quotes,
        }

    for item in items.values():
        facts = [practice_facts(pid) for pid in item["practice_ids"]]
        item["facts"] = facts
        item["tasks"] = list(dict.fromkeys(f["academic_task"] for f in facts if f["academic_task"]))
        item["task_keys"] = sorted({f["task_key"] for f in facts if f["task_key"]})
        item["domains"] = sorted({f["practice_domain"] for f in facts if f["practice_domain"]})
        item["use_statuses"] = sorted({f["use_status"] for f in facts if f["use_status"]})
        all_quotes = list({(q["turn_id"], q["quote"]): q for q in item["quotes"] + item["practice_quotes"]}.values())
        item["voiced_quotes"] = [q for q in all_quotes if voiced(q["turn_id"])]
        item["material_turns"] = sorted(
            {q["turn_id"] for q in all_quotes} | {t for m in item["moves"] for t in m["turn_ids"]},
            key=lambda t: turns[t]["position"])
        folded = interpretation_guard.fold(" ".join(q["quote"] for q in item["voiced_quotes"]))
        moves = {m["type"] for m in item["moves"]}
        item["rule"] = bool(moves & RULE_MOVES or set(item["use_statuses"]) & NON_USE_LIKE
                            or RULE_MARKERS.search(folded))
        item["case"] = bool(moves & CASE_MOVES or CASE_MARKERS.search(folded) or any(
            f["stated_frequency"] and CASE_FREQUENCY.search(interpretation_guard.fold(f["stated_frequency"]))
            for f in facts))

    # Ancrages temporels : tours de l'enquêté·e cités par un item, près d'une citation de cet item
    anchors: dict[tuple[str, int, int], dict] = {}
    for item in items.values():
        item["anchor_keys"] = []
        by_turn: dict[str, list[str]] = {}
        for q in item["voiced_quotes"]:
            by_turn.setdefault(q["turn_id"], []).append(q["quote"])
        for turn_id, quotes in by_turn.items():
            text = turns[turn_id]["text"]
            spans = _quote_spans(text, quotes) or [(0, len(text))]
            for found in find_temporal(text):
                if not _near((found["start"], found["end"]), spans):
                    continue
                key = (turn_id, found["start"], found["end"])
                anchors.setdefault(key, {"turn_id": turn_id, "text": found["text"], "kind": found["kind"],
                                         "rank": found["rank"], "sentence": _sentence(text, found["start"], found["end"]),
                                         "item_ids": []})
                anchors[key]["item_ids"].append(item["id"])
                item["anchor_keys"].append(key)
    order = sorted(anchors, key=lambda k: (turns[k[0]]["position"], k[1]))
    anchor_ids = {key: f"A{index:03d}" for index, key in enumerate(order, start=1)}
    anchor_list = [{"anchor_id": anchor_ids[k], **anchors[k]} for k in order]
    for item in items.values():
        item["anchor_ids"] = [anchor_ids[k] for k in dict.fromkeys(item.pop("anchor_keys"))]

    ordered_items = sorted(items.values(), key=lambda i: (turns[i["material_turns"][0]]["position"]
                                                          if i["material_turns"] else len(turns), i["id"]))
    material = {
        "interview_id": interview_id,
        "items": {i["id"]: i for i in ordered_items},
        "excluded_episodes": excluded,
        "temporal_anchors": anchor_list,
        "regularities": regularities(ordered_items),
    }
    material["summary"] = {
        "usable_episode_count": sum(i["kind"] == "episode" for i in ordered_items),
        "excluded_episode_count": len(excluded),
        "unmarked_practice_count": sum(i["kind"] == "unmarked" for i in ordered_items),
        "temporal_anchor_count": len(anchor_list),
        "needs_review_episode_count": sum(i["kind"] == "episode" and i["needs_review"] for i in ordered_items),
        "item_count": len(ordered_items),
    }
    return material


def regularities(items: list[dict]) -> dict:
    """Rapprochements déterministes entre items. Aucune interprétation : des identifiants groupés."""
    episodes = [i for i in items if i["kind"] == "episode"]
    moves: dict[str, list[str]] = {}
    for e in episodes:
        for move_type in dict.fromkeys(m["type"] for m in e["moves"]):
            moves.setdefault(move_type, []).append(e["id"])
    boundaries: dict[str, dict] = {}
    for e in episodes:
        for b in e["boundaries"]:
            key = _fold_boundary(b)
            if key:
                entry = boundaries.setdefault(key, {"boundary": b, "episode_ids": []})
                if e["id"] not in entry["episode_ids"]:
                    entry["episode_ids"].append(e["id"])
    contexts: dict[str, dict] = {}
    for item in items:
        for fact in item["facts"]:
            if fact["task_key"]:
                entry = contexts.setdefault(fact["task_key"], {"context": fact["academic_task"], "item_ids": []})
                if item["id"] not in entry["item_ids"]:
                    entry["item_ids"].append(item["id"])

    rule_case = [{"item_ids": [i["id"]], "context": (i["tasks"] or [None])[0], "basis": "same_item"}
                 for i in items if i["rule"] and i["case"]]
    polarity = []
    for key, entry in contexts.items():
        refusing = [i for i in items if i["id"] in entry["item_ids"] and any(
            f["task_key"] == key and f["use_status"] in NON_USE_LIKE for f in i["facts"])]
        using = [i for i in items if i["id"] in entry["item_ids"] and any(
            f["task_key"] == key and f["use_status"] in USE_LIKE for f in i["facts"])]
        pairs = [(r, u) for r in refusing for u in using if r["id"] != u["id"]]
        if not pairs:
            continue
        polarity.append({"context": entry["context"], "non_use_item_ids": sorted({r["id"] for r, _ in pairs}),
                         "use_item_ids": sorted({u["id"] for _, u in pairs})})
        for r, u in pairs:
            if r["rule"]:
                rule_case.append({"item_ids": [r["id"], u["id"]], "context": entry["context"],
                                  "basis": "exception_marker" if u["case"] else "contrary_use"})
    ordinary = [i for i in items if i["status"] == "ordinary_practice"]
    ordinary_contexts = [{"context": c["context"], "item_ids": [i for i in c["item_ids"]
                                                                 if i in {o["id"] for o in ordinary}]}
                         for c in contexts.values()]
    return {
        "repeated_moves": {t: ids for t, ids in moves.items() if len(ids) >= 2},
        "repeated_boundaries": [b for b in boundaries.values() if len(b["episode_ids"]) >= 2],
        "shared_contexts": [c for c in contexts.values() if len(c["item_ids"]) >= 2],
        "rule_and_case": rule_case,
        "polarity_contrasts": polarity,
        "signaled_tensions": [e["id"] for e in episodes if CONTRADICTION_SIGNAL in e["signal_types"]],
        "ordinary_item_ids": [o["id"] for o in ordinary],
        "ordinary_by_context": [c for c in ordinary_contexts if len(c["item_ids"]) >= 2],
    }


def public(material: dict) -> dict:
    """Ce que le fichier de sortie garde de la préparation (sans textes : ils sont dans l'étape 4)."""
    return {"preprocessor_version": PREPROCESSOR_VERSION, **material["summary"],
            "excluded_episodes": material["excluded_episodes"],
            "temporal_anchors": [{k: a[k] for k in ("anchor_id", "turn_id", "text", "kind", "item_ids")}
                                 for a in material["temporal_anchors"]],
            "regularities": material["regularities"]}


# --- Représentation envoyée (normalisée, identifiants abrégés) -------------------------------------

def id_prefix(interview_id: str) -> str:
    return f"{interview_id}_"


def build_payload(transcript: dict, material: dict, speaker_warnings: dict | None = None) -> dict:
    warnings = speaker_warnings or {}
    prefix = id_prefix(material["interview_id"])
    speakers = {t["turn_id"]: t["speaker"] for t in transcript["turns"]}

    def short(identifier: str | None) -> str | None:
        return identifier[len(prefix):] if identifier and identifier.startswith(prefix) else identifier

    quotes_by_id: dict[str, dict] = {}
    quote_key: dict[tuple[str, str], str] = {}

    def quote_refs(quotes: list[dict]) -> list[str]:
        refs = []
        for q in quotes:
            key = (q["turn_id"], q["quote"])
            if key not in quote_key:
                quote_key[key] = f"Q{len(quote_key) + 1:03d}"
                entry = {"turn": short(q["turn_id"]), "quote": q["quote"]}
                if speakers.get(q["turn_id"]) != SPEAKER_INTERVIEWEE:
                    entry["speaker"] = speakers.get(q["turn_id"])
                if q["turn_id"] in warnings:
                    entry["speaker_warning"] = True
                quotes_by_id[quote_key[key]] = entry
            refs.append(quote_key[key])
        return list(dict.fromkeys(refs))

    def compact(data: dict) -> dict:
        return {k: v for k, v in data.items() if v not in (None, [], {}, "")}

    episodes, unmarked, practices = [], [], {}
    for item in material["items"].values():
        for pid, fact in zip(item["practice_ids"], item["facts"]):
            practices[short(pid)] = compact({  # valeurs par défaut omises : use = "use", domain = "academic"
                "use": None if fact["use_status"] == "use" else fact["use_status"], "task": fact["academic_task"],
                "domain": None if fact["practice_domain"] == "academic" else fact["practice_domain"],
                "discipline": fact["discipline"], "frequency": fact["stated_frequency"]})
        if item["kind"] == "episode":
            episodes.append(compact({
                "id": short(item["id"]), "status": item["status"],
                "turns": [short(item["turn_start"]), short(item["turn_end"])],
                "practices": [short(p) for p in item["practice_ids"]],
                "moves": [[m["type"], *(short(t) for t in m["turn_ids"])] for m in item["moves"]],
                "boundaries": item["boundaries"], "problem": item["problem"], "summary": item["summary"],
                "role": item["role"], "external": item["external"], "confidence": item["confidence"],
                "signals": [s for s in item["signal_types"] if s == CONTRADICTION_SIGNAL],
                "quotes": quote_refs(item["quotes"] or item["voiced_quotes"]),
                "anchors": item["anchor_ids"],
                "review": item["review_reasons"] if item["needs_review"] else [],
                "speaker_warning": [short(t) for t in item["speaker_warning_turns"]],
            }))
        else:
            unmarked.append(compact({"id": short(item["id"]), "quotes": quote_refs(item["quotes"]),
                                     "anchors": item["anchor_ids"],
                                     "speaker_warning": [short(t) for t in item["speaker_warning_turns"]]}))
    anchors_by_id = {a["anchor_id"]: compact({"turn": short(a["turn_id"]), "text": a["text"], "kind": a["kind"],
                                              "sentence": a["sentence"]}) for a in material["temporal_anchors"]}
    reg = material["regularities"]
    regularities_short = compact({
        "repeated_moves": {t: [short(i) for i in ids] for t, ids in reg["repeated_moves"].items()},
        "repeated_boundaries": [{"boundary": b["boundary"], "episodes": [short(i) for i in b["episode_ids"]]}
                                for b in reg["repeated_boundaries"]],
        "shared_contexts": [{"context": c["context"], "items": [short(i) for i in c["item_ids"]]}
                            for c in reg["shared_contexts"]],
        "rule_and_case": [{"items": [short(i) for i in r["item_ids"]], "context": r["context"], "basis": r["basis"]}
                          for r in reg["rule_and_case"]],
        "polarity_contrasts": [{"context": p["context"], "non_use": [short(i) for i in p["non_use_item_ids"]],
                                "use": [short(i) for i in p["use_item_ids"]]} for p in reg["polarity_contrasts"]],
        "signaled_tensions": [short(i) for i in reg["signaled_tensions"]],
        "ordinary_items": [short(i) for i in reg["ordinary_item_ids"]],
    })
    return {"interview_id": material["interview_id"], "id_prefix": prefix, "episodes": episodes,
            "unmarked_practices": unmarked, "practices_by_id": practices, "quotes_by_id": quotes_by_id,
            "anchors_by_id": anchors_by_id, "regularities": regularities_short}


PAYLOAD_SECTIONS = ("episodes", "unmarked_practices")


def serialize_payload(payload: dict) -> str:
    """JSON compact et déterministe ; un épisode par ligne. « </ » échappé (le texte ne ferme pas la balise)."""
    head = {k: v for k, v in payload.items() if k not in PAYLOAD_SECTIONS}
    parts = [json.dumps(head, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:-1]]
    for key in PAYLOAD_SECTIONS:
        lines = [json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for x in payload[key]]
        parts.append(f'"{key}":[\n' + ",\n".join(lines) + "\n]")
    return (",\n".join(parts) + "}").replace("</", "<\\/")


def payload_estimate(text: str) -> int:
    return estimate_tokens(text)


_SHORT_ID = re.compile(r"^[EPT]\d+$")


def expand_ids(output: dict, interview_id: str) -> dict:
    """Rétablit le préfixe des identifiants abrégés (E…, P…, T…) dans la réponse du modèle."""
    prefix = id_prefix(interview_id)

    def full(value):
        return prefix + value if isinstance(value, str) and _SHORT_ID.match(value) else value

    result = json.loads(json.dumps(output))
    for claim in result.get("claims", []):
        for key in ("episode_ids", "practice_ids", "evidence_turn_ids"):
            claim[key] = [full(v) for v in claim.get(key, [])]
        for anchor in claim.get("temporal_anchors", []):
            anchor["turn_id"] = full(anchor.get("turn_id"))
    for criterion in result.get("student_role_criteria", []):
        for key in ("episode_ids", "evidence_turn_ids"):
            criterion[key] = [full(v) for v in criterion.get(key, [])]
    return result
