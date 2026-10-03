"""Interaction Signal Reader — mise en forme et sélectivité DÉTERMINISTES des signaux (étape 3.7).

Appliquées au résultat de l'agent (appel unique) ou au résultat fusionné (entretien long),
AVANT la validation des preuves. Sans LLM. Rien n'est supprimé en silence : un signal
écarté va dans `set_aside_signals`, avec sa raison ; le texte produit par l'agent n'est
jamais réécrit.

1. Cohérence des tours (`complete_turn_ids`) : un tour cité dans `evidence` qui existe dans
   l'entretien mais manque dans `turn_ids` y est ajouté (ordre de l'entretien), et noté dans
   `turn_ids_added`. Une citation, sa validité et son tour ne changent pas : seule la liste des
   tours concernés est complétée (cause des avertissements EVIDENCE_TURN_NOT_LISTED).

2. Appui sur la seule parole de l'enquêteur (`INTERVIEWER_ONLY_EVIDENCE`) : un signal dont
   TOUTES les citations, valides, proviennent de tours `enqueteur` sans `speaker_warning` est
   écarté. Une question de l'enquêteur n'est pas une formulation de l'enquêté·e. Un tour
   `enqueteur` signalé par l'audit des locuteurs reste un appui possible (à revoir).

3. Micro-marqueur isolé (`ISOLATED_MICRO_MARKER`) : un signal est écarté si
   - son type est hesitation, minimization, intensification, modalization, ou self_correction,
     self_reformulation, other (types sous lesquels un remplisseur seul est typiquement codé) ;
   - sa `surface_form` ne contient QUE des remplisseurs ou adverbes de degré (« euh », « ben »,
     « voilà », « enfin », « juste », « un peu », « on va dire », « vraiment », « énormément »…,
     voir FILLER_EXPRESSIONS) : le marqueur est relevé seul, sans ce qu'il borne ou modifie ;
   - aucun autre signal (non écarté) ne porte sur un même tour : le marqueur n'accompagne
     aucune autocorrection, reformulation, exception, restriction, préférence… ;
   - et rien n'y appelle une vérification : ni `needs_human_review`, ni `explicit_affect`, ni
     `cross_turn_reference`, ni citation invalide (ces signaux restent visibles et validés).
4. Micro-marqueur redondant (`REDUNDANT_MICRO_MARKER`) : même définition, mais le marqueur figure
   déjà dans la `surface_form` d'un autre signal du même tour (« juste » seul, à côté d'une
   restriction « juste reformuler, jamais écrire ») : il ne décrit aucune opération supplémentaire.
   Un marqueur absent des autres formes (« Euh… » avant une autocorrection « enfin non ») est conservé.

   Ce n'est pas un quota : aucun plafond, aucun tri par importance. « Je lui fais juste
   reformuler, jamais écrire » relevé comme `restriction` (« juste reformuler, jamais écrire »)
   est conservé ; « juste » relevé seul, sans rien d'autre sur le tour, ne l'est pas.
"""

from __future__ import annotations

import re

from core import evidence_validator, interpretation_guard
from core.schemas import SPEAKER_INTERVIEWER

SELECTIVITY_VERSION = "1.1"  # 1.1 : contradiction entre tours démontrée (même objet, opposition identifiable)

REASON_INTERVIEWER_ONLY = "INTERVIEWER_ONLY_EVIDENCE"
REASON_MICRO_MARKER = "ISOLATED_MICRO_MARKER"
REASON_REDUNDANT_MICRO = "REDUNDANT_MICRO_MARKER"
REASON_CONTRADICTION_TURNS = "CONTRADICTION_NOT_TWO_TURNS"
REASON_CONTRADICTION_OBJECT = "CONTRADICTION_NO_SHARED_OBJECT"
REASON_CONTRADICTION_OPPOSITION = "CONTRADICTION_NO_OPPOSITION"
SET_ASIDE_REASONS = {
    REASON_INTERVIEWER_ONLY: "Toutes les citations proviennent de tours de l'enquêteur (sans avertissement de locuteur).",
    REASON_MICRO_MARKER: "Micro-marqueur relevé seul (remplisseur ou adverbe de degré), sans autre signal sur le tour.",
    REASON_REDUNDANT_MICRO: "Micro-marqueur déjà compris dans la forme relevée d'un autre signal du même tour.",
    REASON_CONTRADICTION_TURNS: "Contradiction entre tours appuyée sur moins de deux tours cités distincts.",
    REASON_CONTRADICTION_OBJECT: "Contradiction non démontrée : les passages cités ne portent sur aucun objet commun "
                                 "(aucun mot plein partagé, et pas tous deux sur l'usage de l'outil).",
    REASON_CONTRADICTION_OPPOSITION: "Contradiction non démontrée : aucune opposition identifiable entre les passages "
                                     "cités (négation, fréquence, exclusivité ou repère temporel d'un seul côté).",
}

# --- Contradiction entre tours : démontrable seulement si les passages cités portent sur le MÊME objet (un mot plein
# partagé, ou l'usage de l'outil lui-même) ET marquent une opposition identifiable (négation, fréquence, exclusivité ou repère temporel présents
# d'un côté et pas de l'autre). Une différence de sujet, une plaisanterie ou deux émotions éloignées ne suffisent pas.
CONTRADICTION_TYPE = "cross_turn_contradiction"
_STOPWORDS = frozenset("""
avec pour dans sans mais donc alors comme quand parce puis tout tous toute toutes tres plus moins bien aussi meme
encore juste vraiment enfin voila cette ceux celle celles cela elle elles nous vous leur leurs notre votre etre suis
etait etaient sera serait avoir avais avait aurait faire fais fait faites faut fallait dire dis dit peux peut pouvait
veux veut voulais vais sais sait savoir chose choses truc trucs genre devant derriere apres avant maintenant toujours
jamais souvent parfois rarement fois normalement rien aucun aucune personne quelque quelques autre autres beaucoup
trop assez ainsi depuis pendant entre chez vers selon voir vois voit aller allait mettre prendre donner donne
quoi ouais bref style limite tellement carrement franchement clairement certain certains certaine certaines
""".split())
_AI_NAMES = frozenset({"chatgpt", "chat", "openai", "copilot", "gemini", "mistral", "perplexity", "deepl"})
_MARKERS = {
    "negation": r"\b(?:ne|n|pas|jamais|rien|aucun|aucune|non|sans|ni)\b",
    "frequency": r"\b(?:toujours|jamais|souvent|parfois|rarement|une fois|normalement|d habitude|systematiquement|"
                 r"tout le temps|chaque fois|des fois)\b",
    "exclusivity": r"\b(?:moi meme|moi qui|tout seul|toute seule|seul|seule|que moi|personnellement)\b",
    "temporal": r"\b(?:avant|maintenant|aujourd hui|au lycee|desormais|autrefois|a l epoque|plus maintenant)\b",
}
_MARKER_PATTERNS = {name: re.compile(pattern) for name, pattern in _MARKERS.items()}


def _words(text: str) -> list[str]:
    return _NON_WORD.sub(" ", interpretation_guard.fold(text or "").replace("'", " ")).split()


def _content_stems(text: str) -> set[str]:
    return {w[:4] for w in _words(text) if len(w) >= 4 and w not in _STOPWORDS and w not in _AI_NAMES}


def _markers(text: str) -> set[str]:
    folded = " ".join(_words(text))
    return {f"{name}:{match}" for name, pattern in _MARKER_PATTERNS.items() for match in pattern.findall(folded)}


def contradiction_problem(signal: dict) -> str | None:
    """Raison d'écarter une contradiction entre tours (citations valides), ou None si elle est démontrée."""
    by_turn: dict[str, list[str]] = {}
    for evidence in signal.get("evidence", []):
        by_turn.setdefault(evidence.get("turn_id"), []).append(evidence.get("quote") or "")
    if len(by_turn) < 2:
        return REASON_CONTRADICTION_TURNS
    sides = [" ".join(quotes) for quotes in by_turn.values()]
    pairs = [(a, b) for i, a in enumerate(sides) for b in sides[i + 1:]]
    same = [(a, b) for a, b in pairs if _same_object(a, b)]
    if not same:
        return REASON_CONTRADICTION_OBJECT
    if not any(_markers(a) != _markers(b) for a, b in same):
        return REASON_CONTRADICTION_OPPOSITION
    return None


def _same_object(a: str, b: str) -> bool:
    """Objet commun : un mot plein partagé, ou l'usage de l'outil lui-même (les deux passages nomment une IAG ou
    s'adressent à l'outil : « je ne l'utilise jamais… » / « je lui demande toujours… »)."""
    from core.practice_selectivity import addresses_ai, names_ai
    if _content_stems(a) & _content_stems(b):
        return True
    return all(names_ai(side) or addresses_ai(side) for side in (a, b))

# Types sous lesquels un remplisseur relevé seul est typiquement codé (« enfin » seul en autocorrection,
# « voilà » seul en `other`…). Un signal de ces types n'est écarté que si sa forme est un remplisseur nu.
MICRO_SIGNAL_TYPES = frozenset({"hesitation", "minimization", "intensification", "modalization",
                                "self_correction", "self_reformulation", "other"})

# Remplisseurs et adverbes de degré qui, relevés SEULS, ne portent aucune opération identifiable (texte replié).
FILLER_EXPRESSIONS = (
    "un tout petit peu", "un petit peu", "un peu", "on va dire", "on va dire ca", "du coup", "en fait", "tu vois",
    "vous voyez", "je veux dire", "voila", "enfin", "juste", "genre", "quoi", "bon", "ben", "bah", "beh", "hein",
    "bref", "donc", "alors", "vraiment", "enormement", "beaucoup", "trop", "carrement", "grave", "tellement",
    "assez", "plutot", "franchement", "clairement", "limite",
)
_FILLER_ALTERNATIVES = "|".join(sorted((re.escape(e) for e in FILLER_EXPRESSIONS), key=len, reverse=True))
_BARE_FILLERS = re.compile(r"^(?:\s*(?:" + _FILLER_ALTERNATIVES + r"|h?e+u+h*|hu+m+|hm+|m+h+|ah+|oh+)\b)*\s*$")
_NON_WORD = re.compile(r"[^\w\s]+")


def is_bare_filler(surface_form: str | None) -> bool:
    """Vrai si la forme relevée ne contient que des remplisseurs / adverbes de degré (ou rien : « … »)."""
    folded = interpretation_guard.fold(surface_form or "").replace("'", " ")
    return bool(_BARE_FILLERS.match(_NON_WORD.sub(" ", folded)))


def is_bare_micro_marker(signal: dict) -> bool:
    return signal.get("signal_type") in MICRO_SIGNAL_TYPES and is_bare_filler(signal.get("surface_form"))


def _bare_form(surface_form: str | None) -> str:
    return " ".join(_NON_WORD.sub(" ", interpretation_guard.fold(surface_form or "").replace("'", " ")).split())


def _contained_in(marker: str, form: str) -> bool:
    return bool(marker) and re.search(r"\b" + re.escape(marker) + r"\b", form) is not None


def _signal_turns(signal: dict) -> set[str]:
    return set(signal.get("turn_ids", [])) | {e.get("turn_id") for e in signal.get("evidence", [])}


def complete_turn_ids(signal: dict, position: dict[str, int]) -> dict:
    """Copie du signal dont `turn_ids` contient tous les tours cités (existants), dans l'ordre de l'entretien."""
    listed = list(signal.get("turn_ids", []))
    missing = [t for t in dict.fromkeys(e.get("turn_id") for e in signal.get("evidence", []))
               if t in position and t not in listed]
    if not missing:
        return signal
    unknown = len(position)
    ordered = sorted(dict.fromkeys([*listed, *missing]), key=lambda t: position.get(t, unknown))
    added = list(dict.fromkeys([*signal.get("turn_ids_added", []), *missing]))
    return {**signal, "turn_ids": ordered, "turn_ids_added": added}


def _needs_check(signal: dict, results: list[dict]) -> bool:
    """Un signal qui appelle une vérification (de l'agent ou de la validation) n'est jamais écarté."""
    return bool(signal.get("needs_human_review") or signal.get("explicit_affect")
                or signal.get("cross_turn_reference") or not results or not all(r["valid"] for r in results))


def apply(signals: list[dict], transcript: dict, speaker_warnings: dict | None = None) -> dict:
    """Mise en forme puis sélectivité. Renvoie {"signals", "set_aside", "summary"} (ordre conservé)."""
    speaker_warnings = speaker_warnings or {}
    turns = evidence_validator.index_turns(transcript)
    position = {turn_id: info["position"] for turn_id, info in turns.items()}
    interview_id = transcript["interview_id"]
    completed = [complete_turn_ids(s, position) for s in signals]
    turn_ids_completed = sum(c is not s for c, s in zip(completed, signals))

    results = [evidence_validator.validate_evidence(s.get("evidence", []), turns, interview_id) for s in completed]
    reasons: list[str | None] = [None] * len(completed)
    for i, (signal, checked) in enumerate(zip(completed, results)):
        cited = [e.get("turn_id") for e in signal.get("evidence", [])]
        if (checked and all(r["valid"] for r in checked)
                and all(turns[t]["speaker"] == SPEAKER_INTERVIEWER and t not in speaker_warnings for t in cited)):
            reasons[i] = REASON_INTERVIEWER_ONLY
        elif signal.get("signal_type") == CONTRADICTION_TYPE and checked and all(r["valid"] for r in checked):
            reasons[i] = contradiction_problem(signal)  # citations exactes, mais contradiction démontrée ?

    candidates = [i for i, s in enumerate(completed)
                  if reasons[i] is None and is_bare_micro_marker(s) and not _needs_check(s, results[i])]
    supported: dict[str, list[str]] = {}  # tour → formes relevées des signaux substantiels qui le citent
    for i, signal in enumerate(completed):
        if reasons[i] is None and i not in candidates:
            for turn_id in _signal_turns(signal):
                supported.setdefault(turn_id, []).append(_bare_form(signal.get("surface_form")))
    for i in candidates:
        turns_i = _signal_turns(completed[i])
        siblings = [form for t in turns_i for form in supported.get(t, [])]
        if not siblings:
            reasons[i] = REASON_MICRO_MARKER
        elif any(_contained_in(_bare_form(completed[i].get("surface_form")), form) for form in siblings):
            reasons[i] = REASON_REDUNDANT_MICRO

    kept = [s for s, r in zip(completed, reasons) if r is None]
    # un signal écarté garde la vérification de ses citations (lecture humaine), sans identifiant ni statut
    set_aside = [{**s, "evidence": [{**e, "validation": v} for e, v in zip(s.get("evidence", []), checked)],
                  "set_aside_reason": r}
                 for s, checked, r in zip(completed, results, reasons) if r is not None]
    by_reason = {code: reasons.count(code) for code in SET_ASIDE_REASONS if code in reasons}
    return {"signals": kept, "set_aside": set_aside, "summary": {
        "selectivity_version": SELECTIVITY_VERSION, "signals_received": len(signals), "signals_kept": len(kept),
        "signals_set_aside": len(set_aside), "set_aside_by_reason": by_reason,
        "signals_with_turn_ids_completed": turn_ids_completed}}
