"""Interaction Signal Reader sur un entretien long : découpage, lecture à longue distance, fusion.

Fonctions DÉTERMINISTES, sans appel LLM (l'orchestration des appels est dans core/analysis.py).

1. Découpage (`plan_chunks`) — sur les tours structurés, jamais au milieu d'un tour :
   - coût d'un tour = longueur de sa représentation JSON envoyée au modèle / 3,5
     (≈ caractères par token en français), arrondi au supérieur ;
   - entretien d'au plus 1,4 × la taille cible (TRACE_INTERACTION_CHUNK_TOKENS,
     5 000 par défaut) : UN seul appel, exactement comme avant ;
   - sinon, blocs successifs remplis jusqu'à la taille cible (au moins un tour,
     au plus 150 tours) ; un dernier bloc trop petit (< 40 % de la cible) est
     rattaché au précédent ;
   - chaque bloc après le premier reprend les 6 tours qui le précèdent
     (≈ 3 échanges question/réponse) : hésitations et réparations à la jonction,
     relations entre tours voisins, petites contradictions locales.
   Le découpage ignore les `speaker_warning` : il est connu avant l'audit des locuteurs.

2. Lecture à longue distance (`select_long_distance_turns`) — une lecture par blocs ne
   peut pas rapprocher deux passages éloignés. Une passe légère reçoit une SÉLECTION
   compacte de tours de l'enquêté·e (jamais l'entretien entier) :
   - tours cités par un signal local « de contenu » (tout type sauf hésitation, rire,
     silence, changement de pronom), classés par nombre de signaux ;
   - tours contenant un marqueur lexical fort de fréquence absolue ou d'exclusivité
     (« jamais », « toujours », « que pour », « pas du tout », « uniquement »…) ;
   - plus, pour le contexte, la question de l'enquêteur qui précède chaque tour retenu ;
   dans un budget de 6 000 tokens estimés (≈ un bloc ; les candidats les moins appuyés
   sont écartés au-delà, et comptés). Elle ne produit que cross_turn_contradiction,
   significant_repetition ou vocabulary_shift (schéma restreint), et seuls les signaux
   dont les tours CITÉS (au moins deux) ne tiennent pas dans un même bloc sont conservés
   (`is_long_distance`). Étape 3.7 : un micro-marqueur relevé seul (« juste », « un peu »…,
   voir core/signal_selectivity.py) ne rend pas un tour candidat.

3. Fusion (`merge_signals`) — sans LLM. Deux signaux de deux blocs DIFFÉRENTS sont un
   même signal si : même signal_type, mêmes turn_ids, tous ces tours appartiennent aux
   deux blocs (zone de chevauchement), et (même surface_form normalisée, ou mêmes
   citations normalisées, ou surface_form ET une citation du même tour incluses l'une
   dans l'autre). On garde le premier relevé, enrichi de toutes les citations, de tous
   les turn_ids, de needs_human_review (OU logique) et de la provenance des blocs.
   Deux signaux d'un MÊME bloc ne sont jamais fusionnés (l'agent les a distingués).
   Ordre final : ordre de l'entretien (premier tour cité), puis bloc, puis rang dans le bloc ;
   les identifiants S001, S002… sont attribués ensuite par la validation des preuves.

Le découpage (`plan_chunks`, `chunk_request`, `Chunk`) sert aussi au Practice Extractor
(étape 3.7, voir core/practice_chunking.py), avec sa propre taille cible et son propre chevauchement.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass

from agents import interaction_signal_reader as reader
from agents.base import build_agent_input, serialize_agent_input
from core import interpretation_guard, signal_selectivity
from core.schemas import SPEAKER_INTERVIEWER

CHUNKING_VERSION = "1.1"  # 1.1 : chevauchement paramétrable (Practice Extractor), longue distance sur les tours cités
CHARS_PER_TOKEN = 3.5
SINGLE_CALL_FACTOR = 1.4      # jusqu'à 1,4 × la cible : un seul appel (évite un 2e bloc minuscule)
MIN_TAIL_FACTOR = 0.4         # dernier bloc < 40 % de la cible : rattaché au précédent
OVERLAP_TURNS = 6             # tours repris du bloc précédent (≈ 3 échanges question/réponse)
MAX_TURNS_PER_CHUNK = 150     # borne la sortie d'un bloc fait de nombreux tours très courts
LONG_DISTANCE_MAX_TOKENS = 6000  # budget (tokens estimés) de la sélection : au plus ≈ un bloc, jamais l'entretien

# Signaux de pure forme : ils ne désignent pas un énoncé susceptible d'être mis en regard à distance.
FORMAL_SIGNAL_TYPES = frozenset({"hesitation", "transcribed_laughter", "transcribed_silence", "pronoun_shift"})

# Marqueurs de fréquence, d'exclusivité ou de négation d'usage (texte replié : minuscules, sans accents).
_LEXICAL_MARKERS = re.compile(
    r"\b(?:jamais|toujours|tout le temps|tous les jours|a chaque fois|systematiquement|pas du tout|"
    r"plus du tout|que pour|uniquement|seulement)\b"
)


# --- Découpage ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    index: int          # 1, 2, …
    start: int          # position du premier tour envoyé (chevauchement compris)
    core_start: int     # position du premier tour propre à ce bloc
    end: int            # position exclusive
    turn_ids: tuple[str, ...]
    estimated_tokens: int

    @property
    def overlap_turns(self) -> int:
        return self.core_start - self.start

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end

    def describe(self) -> dict:
        return {"chunk": self.index, "first_turn_id": self.turn_ids[0] if self.turn_ids else None,
                "last_turn_id": self.turn_ids[-1] if self.turn_ids else None,
                "turn_count": len(self.turn_ids), "overlap_turns": self.overlap_turns,
                "estimated_tokens": self.estimated_tokens}


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def turn_costs(transcript: dict) -> list[int]:
    """Coût estimé de chaque tour, tel qu'il est sérialisé pour le modèle (sans speaker_warning)."""
    compact = build_agent_input(transcript)["turns"]
    return [estimate_tokens(json.dumps(t, ensure_ascii=False, sort_keys=True)) + 1 for t in compact]


def plan_chunks(transcript: dict, target_tokens: int, overlap_turns: int = OVERLAP_TURNS) -> list[Chunk]:
    """Blocs de tours consécutifs (un seul bloc pour un entretien court)."""
    turns = transcript["turns"]
    ids = [t["turn_id"] for t in turns]
    costs = turn_costs(transcript)
    total = sum(costs)
    if not turns or (total <= target_tokens * SINGLE_CALL_FACTOR
                     and len(turns) <= MAX_TURNS_PER_CHUNK * SINGLE_CALL_FACTOR):
        return [Chunk(1, 0, 0, len(turns), tuple(ids), total)]
    cores: list[list[int]] = []  # [début, fin) propres à chaque bloc
    start = 0
    while start < len(turns):
        end, size = start, 0
        while end < len(turns) and (end == start or (size + costs[end] <= target_tokens
                                                      and end - start < MAX_TURNS_PER_CHUNK)):
            size += costs[end]
            end += 1
        cores.append([start, end])
        start = end
    if len(cores) > 1:
        tail = sum(costs[cores[-1][0]:cores[-1][1]])
        if tail < target_tokens * MIN_TAIL_FACTOR and cores[-1][1] - cores[-2][0] <= MAX_TURNS_PER_CHUNK:
            last = cores.pop()
            cores[-1][1] = last[1]
    chunks = []
    for index, (core_start, end) in enumerate(cores, start=1):
        start = max(0, core_start - overlap_turns) if index > 1 else 0
        chunks.append(Chunk(index, start, core_start, end, tuple(ids[start:end]), sum(costs[start:end])))
    return chunks


def chunk_ranges(chunks: list[Chunk]) -> list[dict]:
    return [c.describe() for c in chunks]


def max_interaction_calls(chunks: list[Chunk]) -> int:
    """Appels Interaction Reader au plus : un par bloc, plus la lecture à longue distance s'il y a plusieurs blocs."""
    return len(chunks) + (1 if len(chunks) > 1 else 0)


def routed_warnings(turn_ids, warnings: dict[str, dict]) -> dict[str, dict]:
    """Avertissements de locuteur des seuls tours transmis (bloc ou sélection)."""
    return {t: warnings[t] for t in turn_ids if t in warnings}


def chunk_request(transcript: dict, chunk: Chunk, warnings: dict[str, dict],
                  template: str = reader.CHUNK_USER_TEMPLATE, overlap_note: str = reader.CHUNK_OVERLAP_NOTE) -> dict:
    """Message d'un bloc : représentation compacte des tours du bloc, avec leurs seuls avertissements.

    `template` / `overlap_note` : gabarit du message de bloc de l'agent (Interaction Reader par défaut)."""
    sub = {"interview_id": transcript["interview_id"], "turns": transcript["turns"][chunk.start:chunk.end]}
    chunk_warnings = routed_warnings(chunk.turn_ids, warnings)
    agent_input = build_agent_input(sub, chunk_warnings)
    transcript_json = serialize_agent_input(agent_input)
    note = "" if not chunk.overlap_turns else overlap_note.format(
        overlap_turns=chunk.overlap_turns, overlap_last_turn_id=chunk.turn_ids[chunk.overlap_turns - 1])
    message = template.format(
        interview_id=transcript["interview_id"], turn_count=len(chunk.turn_ids),
        first_turn_id=chunk.turn_ids[0], last_turn_id=chunk.turn_ids[-1], overlap_note=note,
        transcript_json=transcript_json)
    return {"chunk": chunk, "user_message": message, "warnings": chunk_warnings}


# --- Lecture à longue distance ------------------------------------------------------------

def _chunks_of(position: int, chunks: list[Chunk]) -> list[int]:
    return [c.index for c in chunks if c.contains(position)]


def select_long_distance_turns(transcript: dict, chunks: list[Chunk], chunk_signals: list[list[dict]],
                               budget_tokens: int = LONG_DISTANCE_MAX_TOKENS) -> dict:
    """Sélection compacte des passages candidats à un rapprochement entre blocs.

    Renvoie {"positions": tours retenus (ordre de l'entretien), "candidate_count", "dropped_count",
    "needs_llm"} ; needs_llm est faux s'il n'y a pas au moins deux candidats dans des blocs différents.
    """
    turns = transcript["turns"]
    position = {t["turn_id"]: i for i, t in enumerate(turns)}
    costs = turn_costs(transcript)
    score: dict[int, int] = {}
    for signals in chunk_signals:
        for signal in signals:
            if signal.get("signal_type") in FORMAL_SIGNAL_TYPES or signal_selectivity.is_bare_micro_marker(signal):
                continue
            for turn_id in dict.fromkeys(e.get("turn_id") for e in signal.get("evidence", [])):
                if turn_id in position and turns[position[turn_id]]["speaker"] != SPEAKER_INTERVIEWER:
                    score[position[turn_id]] = score.get(position[turn_id], 0) + 1
    lexical = {i for i, t in enumerate(turns)
               if t["speaker"] != SPEAKER_INTERVIEWER and _LEXICAL_MARKERS.search(interpretation_guard.fold(t["text"]))}
    candidates = sorted(set(score) | lexical, key=lambda i: (-score.get(i, 0), i not in lexical, i))

    def with_context(i: int) -> list[int]:
        before = i - 1
        return [before, i] if before >= 0 and turns[before]["speaker"] == SPEAKER_INTERVIEWER else [i]

    selected: set[int] = set()
    used = dropped = 0
    for i in candidates:
        extra = [p for p in with_context(i) if p not in selected]
        cost = sum(costs[p] for p in extra)
        if used + cost > budget_tokens:
            dropped += 1
            continue
        selected.update(extra)
        used += cost
    retained = [i for i in candidates if i in selected]
    # Blocs contigus : si un bloc contient le premier et le dernier tour retenus, il les contient tous.
    spread = len(retained) >= 2 and not any(c.contains(min(retained)) and c.contains(max(retained)) for c in chunks)
    return {"positions": sorted(selected), "estimated_tokens": used, "candidate_count": len(candidates),
            "selected_count": len(retained), "dropped_count": dropped, "needs_llm": spread}


def long_distance_request(transcript: dict, chunks: list[Chunk], selection: dict, warnings: dict[str, dict]) -> dict:
    turns = transcript["turns"]
    sub = {"interview_id": transcript["interview_id"], "turns": [turns[i] for i in selection["positions"]]}
    selected_ids = [t["turn_id"] for t in sub["turns"]]
    sent_warnings = routed_warnings(selected_ids, warnings)
    agent_input = build_agent_input(sub, sent_warnings)
    for item, i in zip(agent_input["turns"], selection["positions"]):
        item["blocks"] = _chunks_of(i, chunks)
    transcript_json = serialize_agent_input(agent_input)
    ranges = ", ".join(f"bloc {c.index} : {c.turn_ids[0]} à {c.turn_ids[-1]}" for c in chunks)
    message = reader.LONG_DISTANCE_USER_TEMPLATE.format(
        interview_id=transcript["interview_id"], chunk_count=len(chunks), chunk_ranges=ranges,
        turn_count=len(selected_ids), transcript_json=transcript_json)
    return {"user_message": message, "turn_ids": selected_ids, "warnings": sent_warnings}


def is_long_distance(signal: dict, chunks: list[Chunk], position: dict[str, int]) -> bool:
    """Vrai si les tours CITÉS par le signal (au moins deux, existants) ne tiennent dans aucun bloc.

    Les passages mis en regard sont ceux des citations : un `turn_ids` qui annonce deux tours éloignés
    sans citer chacun d'eux ne suffit pas. Un turn_id inexistant est laissé à la validation des preuves."""
    cited = list(dict.fromkeys(e.get("turn_id") for e in signal.get("evidence", [])))
    if any(t not in position for t in [*cited, *signal.get("turn_ids", [])]):
        return len(set(cited) | set(signal.get("turn_ids", []))) >= 2
    if len(cited) < 2:
        return False
    return not any(all(c.contains(position[t]) for t in cited) for c in chunks)


# --- Fusion -------------------------------------------------------------------------------

_LOOSE = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "«": '"', "»": '"', "“": '"', "”": '"', "…": "...",
                        " ": " ", " ": " "})


def normalize(text: str | None) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(_LOOSE).casefold()
    return " ".join(text.split())


def _evidence_set(signal: dict) -> frozenset:
    return frozenset((e.get("turn_id"), normalize(e.get("quote"))) for e in signal.get("evidence", []))


def same_signal(a: dict, b: dict) -> bool:
    """Critères déterministes (voir l'en-tête du module) ; la zone commune aux blocs est vérifiée à part."""
    if a.get("signal_type") != b.get("signal_type") or set(a.get("turn_ids", [])) != set(b.get("turn_ids", [])):
        return False
    sa, sb = normalize(a.get("surface_form")), normalize(b.get("surface_form"))
    if sa and sa == sb:
        return True
    ea, eb = _evidence_set(a), _evidence_set(b)
    if ea and ea == eb:
        return True
    if not (sa and sb and (sa in sb or sb in sa)):
        return False
    return any(ta == tb and qa and qb and (qa in qb or qb in qa) for ta, qa in ea for tb, qb in eb)


def merge_signals(sources: list[dict], chunks: list[Chunk], transcript: dict) -> dict:
    """Réunit les signaux de plusieurs lectures. `sources` : [{"label", "chunk", "signals"}], chunk=None
    pour la lecture à longue distance. Renvoie {"signals", "before", "after", "duplicates_removed"}."""
    position = {t["turn_id"]: i for i, t in enumerate(transcript["turns"])}
    by_index = {c.index: c for c in chunks}
    unknown = len(position)
    entries = []
    for rank, source in enumerate(sources):
        for order, signal in enumerate(source["signals"]):
            known = [position[t] for t in signal.get("turn_ids", []) if t in position]
            entries.append((min(known) if known else unknown, rank, order, source, signal))
    entries.sort(key=lambda e: e[:3])

    kept: list[dict] = []
    removed = 0
    for _, _, _, source, signal in entries:
        positions = [position.get(t) for t in signal.get("turn_ids", [])]
        target = None
        if source["chunk"] is not None and None not in positions:
            for candidate in kept:
                prov = candidate["provenance"]
                if prov["pass"] != "local" or source["chunk"] in prov["chunks"]:
                    continue
                shared = all(all(by_index[c].contains(p) for c in (*prov["chunks"], source["chunk"]))
                             for p in positions)
                if shared and same_signal(candidate["signal"], signal):
                    target = candidate
                    break
        if target is None:
            kept.append({"signal": dict(signal), "provenance": {
                "pass": "local" if source["chunk"] is not None else "long_distance",
                "chunks": [source["chunk"]] if source["chunk"] is not None else [],
                "merged_duplicates": 0}})
            continue
        removed += 1
        merged = target["signal"]
        evidence = list(merged.get("evidence", []))
        seen = {(e.get("turn_id"), e.get("quote")) for e in evidence}
        evidence += [e for e in signal.get("evidence", []) if (e.get("turn_id"), e.get("quote")) not in seen]
        merged["evidence"] = evidence
        # tours cités compris : toute citation ajoutée par la fusion a son tour dans turn_ids
        cited = [e.get("turn_id") for e in evidence if e.get("turn_id") in position]
        merged["turn_ids"] = sorted(dict.fromkeys([*merged.get("turn_ids", []), *signal.get("turn_ids", []), *cited]),
                                    key=lambda t: position.get(t, unknown))
        merged["needs_human_review"] = bool(merged.get("needs_human_review") or signal.get("needs_human_review"))
        for key in ("explicit_affect", "cross_turn_reference", "topic"):
            if merged.get(key) is None and signal.get(key) is not None:
                merged[key] = signal[key]
        target["provenance"]["chunks"].append(source["chunk"])
        target["provenance"]["merged_duplicates"] += 1

    signals = [{**k["signal"], "provenance": k["provenance"]} for k in kept]
    return {"signals": signals, "before": len(entries), "after": len(signals), "duplicates_removed": removed}
