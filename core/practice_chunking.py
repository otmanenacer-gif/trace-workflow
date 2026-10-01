"""Practice Extractor sur un entretien long : découpage et fusion déterministe (étape 3.7).

Fonctions DÉTERMINISTES, sans appel LLM (les appels sont orchestrés par core/analysis.py).
Le découpage réutilise celui de l'Interaction Reader (core/interaction_chunking.py) :

1. Découpage (`plan_chunks`) — sur les tours, jamais au milieu d'un tour :
   - taille cible TRACE_PRACTICE_CHUNK_TOKENS (4 000 tokens estimés par défaut) ;
   - entretien d'au plus 1,4 × la cible (≈ 5 600 tokens, ≈ 85 à 130 tours) : UN seul appel, comme avant ;
   - chaque bloc après le premier reprend les 8 tours qui le précèdent (≈ 4 échanges
     question/réponse) : une situation racontée à la jonction reste lisible en entier.

2. Fusion (`merge_practices`) — sans LLM. Deux pratiques ne sont un doublon que si elles
   décrivent clairement la même conduite :
   - elles viennent de deux blocs DIFFÉRENTS (deux pratiques d'un même bloc ne sont jamais
     fusionnées : l'agent les a distinguées) ;
   - même `use_status` (jamais use / non_use, use / refusal, past_use / non_use…) ;
   - même `non_use_reason` et même `practice_domain` (jamais études / vie personnelle) ;
   - `assessment_context` identique ou inconnu d'un côté ; `academic_task` identique, inclus
     l'un dans l'autre ou absent d'un côté ; outils (`ai_tool`) communs ou absents d'un côté ;
   - et un ANCRAGE commun : une citation de chacune, du MÊME tour, situé dans la zone commune
     aux deux blocs (chevauchement), l'une incluse dans l'autre (casse, apostrophes et blancs
     normalisés).
   Chaque pratique d'un bloc absorbe au plus une pratique d'un autre bloc : les paires sont
   appariées de la plus appuyée (nombre d'ancrages, tâche identique, résumés proches) à la
   moins appuyée. Dans le doute, les deux pratiques sont conservées.
   La pratique conservée est la première relevée, enrichie de toutes les citations, de
   l'intervalle de tours le plus large, des éléments de liste nouveaux, de la lecture la
   plus prudente (`explicitness`), de toutes les `uncertainty_note` et de la provenance.
   Ordre final : ordre de l'entretien (turn_start), puis bloc, puis rang dans le bloc ; les
   identifiants P001, P002… sont attribués ensuite par la validation des preuves.

3. Indices de non-usage (`non_use_cues`) — garde-fou léger, informatif : tours de l'enquêté·e
   contenant une formulation typique de non-usage (« je ne l'utilise pas », « moi-même »,
   « à ma place », « je refuse »…) qu'aucune pratique non_use / refusal ne couvre. Rien n'est
   ajouté ni corrigé : la liste aide seulement la relecture humaine.
"""

from __future__ import annotations

import re

from agents import practice_extractor
from core import interaction_chunking as chunking
from core import interpretation_guard
from core.schemas import SPEAKER_INTERVIEWER

PRACTICE_CHUNKING_VERSION = "1.0"
PRACTICE_OVERLAP_TURNS = 8

_EXPLICITNESS_ORDER = ("direct", "strongly_supported", "unclear")  # du plus au moins certain
_LIST_FIELDS = ("ai_tool", "student_action_before", "ai_action", "student_action_after", "stated_reason",
                "explicit_constraints", "verification_or_control", "other_actors")
_SCALAR_FIELDS = ("academic_task", "discipline", "context", "stated_frequency", "scope_qualifier")
_WORD = re.compile(r"\w+")

# Formulations typiques d'un non-usage ou d'un refus (texte replié, apostrophes → espaces).
_NON_USE_CUES = re.compile(
    r"\b(?:je (?:ne )?(?:l |les |le )?utilise (?:pas|plus|jamais)|je (?:ne )?m en sers (?:pas|plus|jamais)"
    r"|(?:je (?:ne )?)?(?:l |les |le )?ai jamais utilise|moi[- ]meme|a ma place|je refuse|je (?:ne )?veux pas qu"
    r"|je (?:ne )?lui demande (?:pas|jamais|plus)|pas le droit|c est interdit|je prefere (?:le |les |l )?(?:faire|ecrire|rediger))")


def plan_chunks(transcript: dict, target_tokens: int) -> list[chunking.Chunk]:
    return chunking.plan_chunks(transcript, target_tokens, overlap_turns=PRACTICE_OVERLAP_TURNS)


def max_practice_calls(chunks: list) -> int:
    """Appels Practice Extractor au plus : un par bloc (aucune lecture complémentaire)."""
    return len(chunks)


def chunk_request(transcript: dict, chunk: chunking.Chunk, warnings: dict[str, dict]) -> dict:
    return chunking.chunk_request(transcript, chunk, warnings, template=practice_extractor.CHUNK_USER_TEMPLATE,
                                  overlap_note=practice_extractor.CHUNK_OVERLAP_NOTE)


# --- Fusion -------------------------------------------------------------------------------

def _norm(text) -> str:
    return chunking.normalize(text if isinstance(text, str) else "")


def _compatible_task(a, b) -> bool:
    na, nb = _norm(a), _norm(b)
    return not na or not nb or na in nb or nb in na


def _compatible_tools(a: list, b: list) -> bool:
    ta, tb = {_norm(t) for t in a or []}, {_norm(t) for t in b or []}
    return not ta or not tb or bool(ta & tb)


def compatible(a: dict, b: dict) -> bool:
    """Contraintes strictes : deux conduites différentes ne sont jamais rapprochées."""
    return (a.get("use_status") == b.get("use_status")
            and a.get("non_use_reason") == b.get("non_use_reason")
            and a.get("practice_domain") == b.get("practice_domain")
            and (a.get("assessment_context") == b.get("assessment_context")
                 or "unknown" in (a.get("assessment_context"), b.get("assessment_context")))
            and _compatible_task(a.get("academic_task"), b.get("academic_task"))
            and _compatible_tools(a.get("ai_tool"), b.get("ai_tool")))


def shared_anchors(a: dict, b: dict, shared_turns: set[str]) -> int:
    """Nombre de citations de `b` dont le tour (commun aux deux blocs) porte une citation de `a` incluse ou englobante."""
    quotes_a = [(e.get("turn_id"), _norm(e.get("quote"))) for e in a.get("evidence", [])]
    count = 0
    for e in b.get("evidence", []):
        turn, quote = e.get("turn_id"), _norm(e.get("quote"))
        if turn in shared_turns and quote and any(
                ta == turn and qa and (qa in quote or quote in qa) for ta, qa in quotes_a):
            count += 1
    return count


def _summary_similarity(a: dict, b: dict) -> float:
    wa = set(_WORD.findall(_norm(a.get("summary"))))
    wb = set(_WORD.findall(_norm(b.get("summary"))))
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def _merge_into(target: dict, practice: dict, position: dict[str, int]) -> None:
    unknown = len(position)
    evidence = list(target.get("evidence", []))
    seen = {(e.get("turn_id"), e.get("quote")) for e in evidence}
    evidence += [e for e in practice.get("evidence", []) if (e.get("turn_id"), e.get("quote")) not in seen]
    target["evidence"] = evidence
    starts = [t for t in (target.get("turn_start"), practice.get("turn_start")) if t in position]
    ends = [t for t in (target.get("turn_end"), practice.get("turn_end")) if t in position]
    if starts:
        target["turn_start"] = min(starts, key=lambda t: position.get(t, unknown))
    if ends:
        target["turn_end"] = max(ends, key=lambda t: position.get(t, -1))
    for name in _LIST_FIELDS:
        values = list(target.get(name) or [])
        known = {_norm(v) for v in values}
        values += [v for v in practice.get(name) or [] if _norm(v) not in known]
        target[name] = values
    for name in _SCALAR_FIELDS:
        if target.get(name) is None and practice.get(name) is not None:
            target[name] = practice[name]
    if target.get("assessment_context") == "unknown" and practice.get("assessment_context") not in (None, "unknown"):
        target["assessment_context"] = practice["assessment_context"]
    ranks = [_EXPLICITNESS_ORDER.index(x) for x in (target.get("explicitness"), practice.get("explicitness"))
             if x in _EXPLICITNESS_ORDER]
    if ranks:
        target["explicitness"] = _EXPLICITNESS_ORDER[max(ranks)]
    notes = [n for n in (target.get("uncertainty_note"), practice.get("uncertainty_note")) if n]
    target["uncertainty_note"] = " | ".join(dict.fromkeys(notes)) or None


def _shared_turns(transcript: dict, chunks: list[chunking.Chunk]) -> set[str]:
    """Tours contenus dans TOUS ces blocs (blocs contigus : intersection des intervalles)."""
    start, end = max(c.start for c in chunks), min(c.end for c in chunks)
    return {t["turn_id"] for t in transcript["turns"][start:end]} if start < end else set()


def merge_practices(sources: list[dict], chunks: list[chunking.Chunk], transcript: dict) -> dict:
    """Réunit les pratiques de plusieurs blocs. `sources` : [{"chunk": index, "practices": [...]}], dans l'ordre
    des blocs. Renvoie {"practices", "before", "after", "duplicates_removed"} (critères : en-tête du module)."""
    position = {t["turn_id"]: i for i, t in enumerate(transcript["turns"])}
    by_index = {c.index: c for c in chunks}
    unknown = len(position)
    kept: list[dict] = []  # {"practice", "chunks", "rank"}
    removed = 0
    before = 0
    for source in sources:
        chunk = by_index[source["chunk"]]
        practices = source["practices"]
        before += len(practices)
        # paires candidates (pratique de ce bloc, pratique déjà retenue d'un AUTRE bloc), les plus appuyées d'abord
        pairs = []
        for j, practice in enumerate(practices):
            for k, entry in enumerate(kept):
                if chunk.index in entry["chunks"] or not compatible(entry["practice"], practice):
                    continue
                shared = _shared_turns(transcript, [chunk, *(by_index[c] for c in entry["chunks"])])
                anchors = shared_anchors(entry["practice"], practice, shared)
                if anchors:
                    same_task = _norm(entry["practice"].get("academic_task")) == _norm(practice.get("academic_task"))
                    pairs.append((-anchors, not same_task, -_summary_similarity(entry["practice"], practice), j, k))
        absorbed: dict[int, int] = {}
        used: set[int] = set()
        for *_, j, k in sorted(pairs):
            if j not in absorbed and k not in used:
                absorbed[j] = k
                used.add(k)
        for j, practice in enumerate(practices):
            if j in absorbed:
                entry = kept[absorbed[j]]
                _merge_into(entry["practice"], practice, position)
                entry["chunks"].append(chunk.index)
                removed += 1
            else:
                kept.append({"practice": dict(practice), "chunks": [chunk.index], "rank": j})
    kept.sort(key=lambda e: (position.get(e["practice"].get("turn_start"), unknown), e["chunks"][0], e["rank"]))
    practices = [{**e["practice"], "provenance": {"chunks": e["chunks"], "merged_duplicates": len(e["chunks"]) - 1}}
                 for e in kept]
    return {"practices": practices, "before": before, "after": len(practices), "duplicates_removed": removed}


# --- Indices de non-usage -----------------------------------------------------------------

def non_use_cue_turns(transcript: dict) -> list[str]:
    """Tours de l'enquêté·e (ou inconnus) contenant une formulation typique de non-usage ou de refus."""
    return [t["turn_id"] for t in transcript["turns"] if t["speaker"] != SPEAKER_INTERVIEWER
            and _NON_USE_CUES.search(interpretation_guard.fold(t["text"]).replace("'", " "))]


def non_use_cues(transcript: dict, practices: list[dict]) -> dict:
    """Indices lexicaux de non-usage non couverts par une pratique non_use / refusal (information, jamais corrigé)."""
    position = {t["turn_id"]: i for i, t in enumerate(transcript["turns"])}
    covered: set[int] = set()
    for p in practices:
        if p.get("use_status") not in practice_extractor.NON_USE_STATUSES:
            continue
        span = [position[t] for t in (p.get("turn_start"), p.get("turn_end")) if t in position]
        if len(span) == 2:
            covered.update(range(min(span), max(span) + 1))
        covered.update(position[e["turn_id"]] for e in p.get("evidence", []) if e.get("turn_id") in position)
    cues = non_use_cue_turns(transcript)
    uncovered = [t for t in cues if position[t] not in covered]
    return {"cue_turn_count": len(cues), "uncovered_turn_ids": uncovered}
