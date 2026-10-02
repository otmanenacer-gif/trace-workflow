"""Entretien long SYNTHÉTIQUE de l'étape 3.7 (330 tours) et lecteurs simulés. Aucun vrai entretien.

Le texte contient volontairement :
- BEAUCOUP de remplisseurs sans fonction (« euh », « ben », « voilà », « juste », « un peu »,
  « on va dire », « enfin », « énormément ») dans des réponses ordinaires ;
- plusieurs usages, non-usages, refus, une préférence, un usage passé suivi d'un non-usage actuel,
  une exception (« enfin sauf quand je suis vraiment en retard »), une émotion nommée liée au
  jugement d'un professeur, une évaluation de sa propre parole ;
- deux contradictions à longue distance (T0024 / T0300 sur la rédaction des devoirs ;
  T0070 / T0260 sur l'usage hors des cours) ;
- deux tours marqués « Enquêteur » qui sont des réponses à la première personne (T0201, T0241) :
  l'audit des locuteurs les signale ; le transcript n'est jamais modifié.

Lecteurs simulés, DÉTERMINISTES (les mêmes tours produisent les mêmes objets, quel que soit
le bloc : d'où des doublons dans les zones de chevauchement, que TRACE doit fusionner) :
- `practice_reader` : une pratique par conduite déclarée dans un tour clé ;
- `selective_reader` : un Interaction Reader qui applique la règle de pertinence de l'étape 3.7 ;
- `naive_reader` : un Interaction Reader qui SURCODE (un signal par remplisseur, plus un signal
  appuyé sur la seule question de l'enquêteur et un signal aux turn_ids incomplets) : il imite le
  comportement observé sur le vrai entretien long (498 signaux) ;
- `long_distance_reader` : les deux contradictions, si les deux tours sont dans la sélection.
Toutes les citations sont des sous-chaînes exactes des tours.
"""

from __future__ import annotations

import re

from tests.fake_llm import text_response
from tests.synthetic_interviews import practice as base_practice
from tests.synthetic_interviews import signal as base_signal
from tests.synthetic_long_interview import sent_turns

FILENAME = "Entretien_etape_3_7.txt"
INTERVIEW_ID = "ENTRETIEN_ETAPE_3_7"
TURN_COUNT = 330
WARNED_TURNS = (201, 241)

_TOPICS = ("les fiches de révision", "les exposés", "les partiels blancs", "les mails administratifs",
           "les lectures obligatoires", "les exercices de statistiques", "les dossiers de groupe", "la prise de notes",
           "les recherches en bibliothèque", "le mémoire", "les oraux", "les comptes rendus de TD")
_QUESTIONS = ("Et pour {topic}, tu fais comment ?", "Tu peux me raconter comment ça se passe pour {topic} ?",
              "D'accord. Et {topic}, concrètement ?", "Et pour {topic}, ça se passe comment en ce moment ?")
# Réponses ordinaires saturées de remplisseurs, sans fonction dans le récit des pratiques (aucune IAG).
_FILLER_ANSWERS = (
    "Euh, ben voilà, pour {topic} je fais un peu comme d'habitude, on va dire, avec mes notes de cours.",
    "Ben pour {topic}, euh, je reprends juste mes cours et je fais un peu de fiches, voilà.",
    "Alors {topic}, on va dire que ça dépend, euh, enfin je relis un peu mes notes la veille, voilà.",
    "Euh oui, pour {topic} je lis énormément les manuels, enfin les livres de la bibliothèque, ben voilà.",
    "Ben oui.",
)

# Tours clés de l'enquêté·e (numéro → texte). Les citations des lecteurs simulés en sont des sous-chaînes.
KEY_TURNS = {
    10: "Je ne l'utilise jamais pour mes devoirs… enfin sauf quand je suis vraiment en retard.",
    20: "Il m'arrive de lui faire rédiger des paragraphes quand je bloque sur un devoir.",
    24: "Je ne fais jamais rédiger mes devoirs par ChatGPT.",
    40: "Je préfère écrire moi-même mes dissertations.",
    60: "Oui je m'en sers pour réviser. Bon, c'est un peu facile ce que je dis, je sais.",
    70: "Je ne l'utilise jamais en dehors des cours.",
    80: "J'ai peur que le prof pense que j'ai triché si je rends un texte trop lisse.",
    90: "Pour les mails, je lui demande de corriger l'orthographe.",  # zone de chevauchement des blocs Practice
    116: "Je lui demande parfois un plan, mais je rédige moi-même.",  # zone de chevauchement des blocs Interaction
    100: "Avant je lui faisais faire mes fiches de lecture, mais maintenant je les fais moi-même.",
    120: "Je ne veux pas qu'il réfléchisse à ma place.",
    140: "Je lui fais juste reformuler, jamais écrire.",
    150: "Normalement mes travaux je les fais moi-même.",
    260: "Le week-end je lui demande toujours des recettes de cuisine.",
    300: "Pour le rapport de stage, je lui ai fait rédiger certains passages, seulement l'introduction.",
}
# Tours marqués « Enquêteur » qui sont en fait des réponses à la première personne.
WARNED_TEXTS = {
    201: "Moi personnellement j'utilise ChatGPT pour mes exposés, mais je préfère faire les diapos moi-même.",
    241: "Moi personnellement je lui demande des plans pour mes cours, jamais de texte entier.",
}
# Question de l'enquêteur qu'un lecteur qui surcode transforme à tort en signal de l'enquêté·e.
LEADING_QUESTION_TURN = 79
LEADING_QUESTION = "Tu as peur que tes profs le voient ?"

FILLER_RE = re.compile(r"\b(?:euh|ben|voilà|juste|un peu|on va dire|enfin|énormément)\b", re.IGNORECASE)


def tid(number: int) -> str:
    return f"{INTERVIEW_ID}_T{number:04d}"


def ev(number: int, quote: str) -> dict:
    text = turn_text(number)
    assert quote in text, (number, quote)
    return {"turn_id": tid(number), "quote": quote}


def turn_text(number: int) -> str:
    k = number // 2
    topic = _TOPICS[k % len(_TOPICS)]
    if number in WARNED_TEXTS:
        return WARNED_TEXTS[number]
    if number == LEADING_QUESTION_TURN:
        return LEADING_QUESTION
    if number % 2:
        return _QUESTIONS[k % len(_QUESTIONS)].format(topic=topic)
    if number in KEY_TURNS:
        return KEY_TURNS[number]
    return _FILLER_ANSWERS[k % len(_FILLER_ANSWERS)].format(topic=topic)


def text() -> str:
    lines = []
    for number in range(1, TURN_COUNT + 1):
        speaker = "Enquêteur" if number % 2 else "Enquêté"
        lines.append(f"{speaker} : {turn_text(number)}")
    return "\n".join(lines) + "\n"


def files(edit: tuple[str, str] | None = None) -> list[tuple[str, bytes]]:
    content = text() if edit is None else text().replace(*edit)
    return [(FILENAME, content.encode("utf-8"))]


def micro_marker_count() -> int:
    """Remplisseurs présents dans les tours de l'enquêté·e (ce qu'un lecteur qui surcode relèverait)."""
    return sum(len(FILLER_RE.findall(turn_text(n))) for n in range(2, TURN_COUNT + 1, 2))


# --- Pratiques attendues (Practice Extractor simulé) ---------------------------------------

def _p(number: int, quote: str, **fields) -> dict:
    return base_practice(turn_start=tid(number - 1), turn_end=tid(number), evidence=[ev(number, quote)], **fields)


KEY_PRACTICES = {
    10: [_p(10, "Je ne l'utilise jamais pour mes devoirs", summary="Il indique ne jamais utiliser l'outil pour ses devoirs.",
            use_status="non_use", non_use_reason="not_stated", academic_task="devoirs", stated_frequency="jamais"),
         _p(10, "sauf quand je suis vraiment en retard", summary="Il indique l'utiliser pour ses devoirs quand il est en retard.",
            use_status="use", academic_task="devoirs", stated_reason=["quand je suis vraiment en retard"])],
    20: [_p(20, "Il m'arrive de lui faire rédiger des paragraphes quand je bloque sur un devoir.",
            summary="Il indique lui faire parfois rédiger des paragraphes quand il bloque.", use_status="use",
            academic_task="devoirs", ai_action=["rédige des paragraphes"], stated_frequency="parfois")],
    24: [_p(24, "Je ne fais jamais rédiger mes devoirs par ChatGPT.", summary="Il dit ne jamais faire rédiger ses devoirs.",
            use_status="refusal", non_use_reason="not_stated", academic_task="devoirs", ai_tool=["ChatGPT"],
            stated_frequency="jamais")],
    40: [_p(40, "Je préfère écrire moi-même mes dissertations.", summary="Il indique écrire lui-même ses dissertations.",
            use_status="non_use", non_use_reason="preference", academic_task="dissertations")],
    60: [_p(60, "Oui je m'en sers pour réviser.", summary="Il indique s'en servir pour réviser.", use_status="use",
            academic_task="révisions")],
    70: [_p(70, "Je ne l'utilise jamais en dehors des cours.", summary="Il indique ne jamais l'utiliser hors des cours.",
            use_status="non_use", non_use_reason="not_stated", practice_domain="personal", assessment_context="personal",
            stated_frequency="jamais")],
    90: [_p(90, "je lui demande de corriger l'orthographe", summary="Il indique lui faire corriger l'orthographe de ses mails.",
            use_status="use", academic_task="mails", ai_action=["corrige l'orthographe"])],
    116: [_p(116, "Je lui demande parfois un plan", summary="Il indique lui demander parfois un plan.", use_status="use",
             ai_action=["propose un plan"], stated_frequency="parfois"),
          _p(116, "je rédige moi-même", summary="Il indique rédiger lui-même.", use_status="non_use",
             non_use_reason="not_stated", academic_task="rédaction")],
    100: [_p(100, "Avant je lui faisais faire mes fiches de lecture", summary="Il indique qu'avant, il lui faisait faire ses fiches de lecture.",
             use_status="past_use", academic_task="fiches de lecture"),
          _p(100, "maintenant je les fais moi-même", summary="Il indique faire maintenant ses fiches de lecture lui-même.",
             use_status="non_use", non_use_reason="not_stated", academic_task="fiches de lecture")],
    120: [_p(120, "Je ne veux pas qu'il réfléchisse à ma place.", summary="Il dit ne pas vouloir qu'il réfléchisse à sa place.",
             use_status="refusal", non_use_reason="personal_rule")],
    140: [_p(140, "Je lui fais juste reformuler", summary="Il indique lui faire reformuler.", use_status="use",
             academic_task="rédaction", ai_action=["reformule"]),
          _p(140, "jamais écrire", summary="Il indique ne jamais lui faire écrire.", use_status="refusal",
             non_use_reason="personal_rule", academic_task="rédaction", stated_frequency="jamais")],
    150: [_p(150, "Normalement mes travaux je les fais moi-même.", summary="Il indique faire lui-même ses travaux.",
             use_status="non_use", non_use_reason="not_stated", academic_task="travaux")],
    201: [_p(201, "Moi personnellement j'utilise ChatGPT pour mes exposés", summary="Le passage décrit un usage de ChatGPT pour des exposés.",
             use_status="use", academic_task="exposés", ai_tool=["ChatGPT"], explicitness="unclear",
             uncertainty_note="Le tour T0201 est marqué enquêteur : attribution du locuteur douteuse.")],
    260: [_p(260, "Le week-end je lui demande toujours des recettes de cuisine.", summary="Il indique lui demander des recettes le week-end.",
             use_status="use", practice_domain="personal", assessment_context="personal", stated_frequency="toujours")],
    300: [_p(300, "Pour le rapport de stage, je lui ai fait rédiger certains passages", summary="Il indique lui avoir fait rédiger des passages de son rapport de stage.",
             use_status="use", academic_task="rapport de stage", ai_action=["rédige certains passages"])],
}
EXPECTED_PRACTICE_COUNT = sum(len(v) for v in KEY_PRACTICES.values())


def practice_output(params: dict) -> dict:
    present = {t["turn_id"] for t in sent_turns(params)}
    practices = [p for number, items in sorted(KEY_PRACTICES.items()) if tid(number) in present for p in items]
    return {"practices": practices, "extraction_notes": None}


def practice_reader(params: dict):
    return text_response(practice_output(params))


# --- Signaux attendus (Interaction Reader simulé) ---------------------------------------------

def _s(number: int, signal_type: str, surface: str, quote: str, description: str, **fields) -> dict:
    return base_signal(turn_ids=[tid(number)], signal_type=signal_type, surface_form=surface, description=description,
                       evidence=[ev(number, quote)], **fields)


KEY_SIGNALS = {
    10: [_s(10, "self_correction", "enfin sauf", "Je ne l'utilise jamais pour mes devoirs… enfin sauf quand je suis vraiment en retard.",
            "L'enquêté dit « jamais » puis ajoute « enfin sauf » : il revient sur ce qu'il vient de dire."),
         _s(10, "exception", "sauf quand je suis vraiment en retard", "sauf quand je suis vraiment en retard",
            "L'enquêté signale une exception au non-usage : quand il est en retard.")],
    40: [_s(40, "preference_statement", "Je préfère écrire moi-même", "Je préfère écrire moi-même mes dissertations.",
            "L'enquêté formule une préférence : écrire lui-même ses dissertations.")],
    60: [_s(60, "metadiscursive_self_evaluation", "c'est un peu facile ce que je dis", "c'est un peu facile",
            "L'enquêté qualifie de « facile » ce qu'il vient de dire.")],
    80: [_s(80, "explicit_emotion", "J'ai peur", "J'ai peur que le prof pense que j'ai triché",
            "L'enquêté emploie « peur » à propos de ce que le professeur pourrait penser.", explicit_affect="peur"),
         _s(80, "reference_to_teacher_judgment", "que le prof pense que j'ai triché",
            "J'ai peur que le prof pense que j'ai triché si je rends un texte trop lisse.",
            "L'enquêté évoque ce que le professeur pourrait penser d'un texte trop lisse.")],
    100: [_s(100, "contrast", "Avant… mais maintenant", "Avant je lui faisais faire mes fiches de lecture, mais maintenant je les fais moi-même.",
             "L'enquêté oppose ce qu'il faisait avant et ce qu'il fait maintenant pour ses fiches de lecture.")],
    116: [_s(116, "contrast", "parfois un plan, mais je rédige moi-même",
             "Je lui demande parfois un plan, mais je rédige moi-même.",
             "L'enquêté oppose la demande d'un plan et la rédaction, qu'il dit faire lui-même.")],
    120: [_s(120, "normative_formulation", "Je ne veux pas qu'il réfléchisse à ma place",
             "Je ne veux pas qu'il réfléchisse à ma place.", "L'enquêté formule une règle qu'il se donne.")],
    140: [_s(140, "restriction", "juste reformuler, jamais écrire", "Je lui fais juste reformuler, jamais écrire.",
             "L'enquêté borne l'usage à la reformulation et exclut l'écriture.")],
    201: [_s(201, "preference_statement", "je préfère faire les diapos moi-même", "je préfère faire les diapos moi-même",
             "L'enquêté formule une préférence pour les diapositives. L'attribution du locuteur de ce tour est douteuse.",
             needs_human_review=True)],
    300: [_s(300, "restriction", "seulement l'introduction", "seulement l'introduction",
             "L'enquêté limite les passages rédigés à l'introduction.")],
}
EXPECTED_LOCAL_SIGNAL_COUNT = sum(len(v) for v in KEY_SIGNALS.values())

CONTRADICTIONS = [
    base_signal(turn_ids=[tid(24), tid(300)], signal_type="cross_turn_contradiction", surface_form="jamais / je lui ai fait rédiger",
                description="Au tour T0024, l'enquêté dit ne jamais faire rédiger ses devoirs ; au tour T0300, il dit lui avoir "
                            "fait rédiger certains passages de son rapport de stage.",
                topic="rédaction des devoirs", cross_turn_reference="Faire rédiger des travaux par ChatGPT.",
                evidence=[ev(24, "Je ne fais jamais rédiger mes devoirs par ChatGPT."),
                          ev(300, "je lui ai fait rédiger certains passages")]),
    base_signal(turn_ids=[tid(70), tid(260)], signal_type="cross_turn_contradiction", surface_form="jamais / toujours",
                description="Au tour T0070, l'enquêté dit ne jamais l'utiliser en dehors des cours ; au tour T0260, il dit lui "
                            "demander des recettes de cuisine le week-end.",
                topic="usage hors des cours", cross_turn_reference="Usage de l'outil en dehors des cours.",
                evidence=[ev(70, "Je ne l'utilise jamais en dehors des cours."),
                          ev(260, "Le week-end je lui demande toujours des recettes de cuisine.")]),
]

_MICRO_CODING = (  # ce qu'un lecteur qui surcode relève pour chaque remplisseur (forme exacte cherchée dans le tour)
    ("Euh", "hesitation"), ("euh", "hesitation"), ("Ben", "hesitation"), ("ben", "hesitation"), ("voilà", "other"),
    ("juste", "minimization"), ("un peu", "minimization"), ("on va dire", "modalization"), ("enfin", "self_correction"),
    ("énormément", "intensification"),
)


def selective_output(params: dict) -> dict:
    present = {t["turn_id"]: t for t in sent_turns(params)}
    signals = [s for number, items in sorted(KEY_SIGNALS.items()) if tid(number) in present for s in items]
    return {"signals": signals, "reading_notes": None}


def naive_output(params: dict) -> dict:
    """Surcodage : les signaux pertinents, plus un signal par remplisseur et deux objets mal formés."""
    turns = sent_turns(params)
    present = {t["turn_id"] for t in turns}
    signals = list(selective_output(params)["signals"])
    for turn in turns:
        if turn["speaker"] != "enquete":
            continue
        for surface, signal_type in _MICRO_CODING:
            for match in re.finditer(r"\b" + re.escape(surface) + r"\b", turn["text"]):
                signals.append(base_signal(turn_ids=[turn["turn_id"]], signal_type=signal_type, surface_form=surface,
                                           description=f"« {surface} » dans la réponse.",
                                           evidence=[{"turn_id": turn["turn_id"], "quote": surface}]))
    if tid(LEADING_QUESTION_TURN) in present:  # appui sur la seule question de l'enquêteur
        signals.append(base_signal(turn_ids=[tid(LEADING_QUESTION_TURN)], signal_type="reference_to_teacher_judgment",
                                   surface_form="tes profs", description="La question évoque ce que les professeurs verraient.",
                                   evidence=[ev(LEADING_QUESTION_TURN, LEADING_QUESTION)]))
    if tid(119) in present and tid(120) in present:  # citation d'un tour absent de turn_ids
        signals.append(base_signal(turn_ids=[tid(120)], signal_type="reference_to_rule",
                                   surface_form="réfléchisse à ma place",
                                   description="L'enquêté répond à la question en posant une limite.",
                                   evidence=[ev(119, turn_text(119)), ev(120, "réfléchisse à ma place")]))
    position = {t["turn_id"]: i for i, t in enumerate(turns)}
    signals.sort(key=lambda s: position.get(s["turn_ids"][0], 0))
    return {"signals": signals, "reading_notes": None}


def selective_reader(params: dict):
    return text_response(selective_output(params))


def naive_reader(params: dict):
    return text_response(naive_output(params))


def long_distance_output(params: dict) -> dict:
    present = {t["turn_id"] for t in sent_turns(params)}
    return {"signals": [c for c in CONTRADICTIONS if set(c["turn_ids"]) <= present], "reading_notes": None}


def long_distance_reader(params: dict):
    return text_response(long_distance_output(params))
