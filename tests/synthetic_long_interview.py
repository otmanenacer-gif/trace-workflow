"""Entretien LONG SYNTHÉTIQUE (349 tours, comme le premier vrai entretien long) et lecteur simulé.

Aucun vrai entretien. Le texte contient volontairement :
- des hésitations « Euh… » régulières, dont certaines dans les zones de chevauchement des blocs ;
- une contradiction entre deux passages TRÈS éloignés (T0010 : « jamais » pour rédiger ;
  T0330 : rédige « souvent » les introductions) ;
- en option, un tour volontairement mal attribué (réponse à la première personne marquée
  « Enquêteur ») pour vérifier l'acheminement des speaker_warning vers les seuls blocs concernés.

`simulated_chunk_reader` imite le modèle de façon DÉTERMINISTE à partir du message reçu :
les mêmes tours produisent les mêmes signaux, quel que soit le bloc (d'où des doublons
dans les zones de chevauchement, que TRACE doit fusionner). Toutes les citations sont exactes.
"""

from __future__ import annotations

import json
import re

from tests.fake_llm import text_response

LONG_FILENAME = "Entretien_long.txt"
LONG_INTERVIEW_ID = "ENTRETIEN_LONG"
TURN_COUNT = 349
EARLY_TURN, LATE_TURN = 10, 330
EARLY_QUOTE = "je n'utilise jamais ChatGPT pour rédiger mes dissertations"
LATE_QUOTE = "ChatGPT rédige souvent mes introductions"
MISATTRIBUTED_TURN = 201
MISATTRIBUTED_TEXT = ("Moi j'ai fait mes fiches de révision du semestre numéro vingt avec mes notes de cours, "
                      "et j'ai relu mes cahiers.")

_TOPICS = ("les fiches de lecture", "les exposés", "les révisions", "les mails aux professeurs", "les traductions",
           "les exercices de statistiques", "les dossiers de groupe", "la prise de notes", "les recherches documentaires",
           "le mémoire", "les oraux", "les comptes rendus")
_QUESTIONS = ("Et pour {topic}, comment tu t'organises, au semestre {n} ?",
              "Tu peux me raconter une fois récente pour {topic}, au semestre {n} ?",
              "Concrètement, pour {topic}, tu fais comment, au semestre {n} ?")
_ANSWERS = ("Pour {topic}, je travaille d'abord avec mes notes de cours, puis je compare avec les lectures conseillées "
            "par l'enseignante ; au semestre {n} j'ai gardé la même méthode parce qu'elle me fait gagner du temps "
            "et que je retrouve plus facilement mes sources.",
            "Alors {topic}, ça dépend des semaines. Au semestre {n} je m'y prenais tard, je relisais le cours la veille "
            "et je faisais un petit plan à la main, avec des couleurs, avant d'écrire au propre sur l'ordinateur.",
            "Pour {topic}, je demande parfois à une amie de relire, on s'échange nos brouillons. Au semestre {n} on "
            "avait un groupe de travail à la bibliothèque le jeudi, on restait jusqu'à la fermeture.")


def _turns(misattributed: bool) -> list[tuple[str, str]]:
    turns = []
    for number in range(1, TURN_COUNT + 1):
        k = number // 2
        topic = _TOPICS[k % len(_TOPICS)]
        if number % 2:  # tours impairs : enquêteur
            text = _QUESTIONS[k % len(_QUESTIONS)].format(topic=topic, n=k)
            speaker = "Enquêteur"
            if misattributed and number == MISATTRIBUTED_TURN:
                text = MISATTRIBUTED_TEXT
        else:
            speaker = "Enquêté"
            text = _ANSWERS[k % len(_ANSWERS)].format(topic=topic, n=k)
            if k % 7 == 0:
                text = "Euh… " + text
        if number == EARLY_TURN:
            text = f"Non, {EARLY_QUOTE}, c'est moi qui écris tout, du début à la fin."
        if number == LATE_TURN:
            text = f"Pour les dissertations, {LATE_QUOTE}, et après je reprends le reste."
        turns.append((speaker, text))
    return turns


def long_text(misattributed: bool = False) -> str:
    return "\n".join(f"{speaker} : {text}" for speaker, text in _turns(misattributed)) + "\n"


def long_files(misattributed: bool = False) -> list[tuple[str, bytes]]:
    return [(LONG_FILENAME, long_text(misattributed).encode("utf-8"))]


def ltid(number: int) -> str:
    return f"{LONG_INTERVIEW_ID}_T{number:04d}"


# --- Lecteur simulé -----------------------------------------------------------------------

_TRANSCRIPT_RE = re.compile(r"<transcript>\n(.*)\n</transcript>", re.S)


def sent_turns(params: dict) -> list[dict]:
    """Tours transmis dans un message (bloc ou sélection à longue distance)."""
    content = params["messages"][0]["content"]
    return json.loads(_TRANSCRIPT_RE.search(content).group(1))["turns"]


def _signal(turn: dict, signal_type: str, surface: str, quote: str, description: str) -> dict:
    warned = "speaker_warning" in turn
    if warned:
        description += " L'attribution du locuteur de ce tour est douteuse."
    return {"turn_ids": [turn["turn_id"]], "signal_type": signal_type, "surface_form": surface,
            "description": description, "topic": None, "evidence": [{"turn_id": turn["turn_id"], "quote": quote}],
            "explicit_affect": None, "cross_turn_reference": None, "explicitness": "direct",
            "needs_human_review": warned}


def chunk_signals(params: dict) -> dict:
    signals = []
    for turn in sent_turns(params):
        text = turn["text"]
        if turn["speaker"] == "enquete" and text.startswith("Euh…"):
            signals.append(_signal(turn, "hesitation", "Euh…", "Euh…", "Hésitation transcrite en début de réponse."))
        if "jamais" in text:
            signals.append(_signal(turn, "intensification", "jamais", EARLY_QUOTE, "« jamais » renforce la négation."))
        if "souvent" in text:
            signals.append(_signal(turn, "modalization", "souvent", LATE_QUOTE, "Fréquence indiquée par « souvent »."))
        if "speaker_warning" in turn:
            signals.append(_signal(turn, "other", "Moi j'ai fait", "Moi j'ai fait mes fiches de révision",
                                   "Récit à la première personne."))
    return {"signals": signals, "reading_notes": None}


def simulated_chunk_reader(params: dict):
    return text_response(chunk_signals(params))


CONTRADICTION = {
    "turn_ids": [ltid(EARLY_TURN), ltid(LATE_TURN)], "signal_type": "cross_turn_contradiction",
    "surface_form": "jamais / souvent",
    "description": f"Au tour {ltid(EARLY_TURN)}, l'enquêté dit ne jamais utiliser ChatGPT pour rédiger ses "
                   f"dissertations ; au tour {ltid(LATE_TURN)}, il dit que ChatGPT rédige souvent ses introductions.",
    "topic": "rédaction des dissertations", "evidence": [{"turn_id": ltid(EARLY_TURN), "quote": EARLY_QUOTE},
                                                         {"turn_id": ltid(LATE_TURN), "quote": LATE_QUOTE}],
    "explicit_affect": None, "cross_turn_reference": "Usage de ChatGPT pour rédiger les dissertations.",
    "explicitness": "direct", "needs_human_review": False,
}


def long_distance_signals(params: dict) -> dict:
    ids = {t["turn_id"] for t in sent_turns(params)}
    found = ltid(EARLY_TURN) in ids and ltid(LATE_TURN) in ids
    return {"signals": [CONTRADICTION] if found else [], "reading_notes": None}


def simulated_long_distance_reader(params: dict):
    return text_response(long_distance_signals(params))
