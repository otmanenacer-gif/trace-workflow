"""Entretien long SYNTHÉTIQUE de l'étape 4 (320 tours) et lecteurs simulés. Aucun vrai entretien.

Beaucoup de pratiques racontées comme allant de soi, quelques épisodes seulement :
- 20 pratiques ordinaires (« Pour {tâche}, je lui demande {action}. »), accompagnées de signaux
  faibles (hésitation « comment dire », rire transcrit, intensification) qui ne créent jamais
  de candidat ;
- 5 passages qui donnent un épisode d'accountability : restriction + refus (T0040), contradiction
  à distance T0060 / T0280 (« jamais » / « je lui ai fait écrire la conclusion »), jugement de la
  professeure + restriction (T0100), non-usage + exception (T0140), auto-évaluation (T0200) ;
- 3 passages qui deviennent candidats mais restent des pratiques ordinaires une fois examinés :
  préférence entre deux outils (T0120), reformulation portant sur l'objet (T0170), contraste entre
  deux outils (T0240) ;
- des signaux sans pratique proche (« je ne sais pas si je réponds bien à tes questions »).

Les lecteurs de l'étape 3 sont DÉTERMINISTES et suivent les blocs envoyés (un objet n'est renvoyé
que si tous ses tours figurent dans le bloc) ; le lecteur de l'étape 4 applique une table de
décisions par tour d'ancrage (tests/synthetic_stage4.scripted_builder).
"""

from __future__ import annotations

from tests.fake_llm import text_response
from tests.synthetic_interviews import practice as base_practice
from tests.synthetic_interviews import signal as base_signal
from tests.synthetic_long_interview import sent_turns
from tests.synthetic_stage4 import decision, move, scripted_builder

FILENAME = "Entretien_etape_4_long.txt"
INTERVIEW_ID = "ENTRETIEN_ETAPE_4_LONG"
TURN_COUNT = 320

_TASKS = (("les fiches de révision", "des questions d'entraînement"), ("les exposés", "des idées de plan d'exposé"),
          ("le vocabulaire d'anglais", "des listes de mots"), ("les mails aux profs", "une formule de politesse"),
          ("les statistiques", "d'expliquer une formule"), ("la méthodologie", "des exemples de problématique"),
          ("les lectures", "le contexte historique d'un texte"), ("le droit", "la définition d'un terme juridique"),
          ("l'économie", "un schéma de l'offre et de la demande"), ("la prise de notes", "de remettre mes notes au propre"),
          ("les candidatures de stage", "une liste d'entreprises"), ("la philosophie", "de résumer un argument"),
          ("le code en Python", "de trouver une erreur dans mon code"), ("la bibliographie", "le format d'une citation"),
          ("les partiels blancs", "des sujets d'entraînement"), ("la sociologie", "des exemples de concepts"),
          ("l'agenda", "d'organiser ma semaine"), ("les révisions de dernière minute", "un quiz rapide"),
          ("l'histoire", "une chronologie"), ("les TD de gestion", "de vérifier un calcul"))
ORDINARY_TURNS = (6, 14, 22, 30, 50, 70, 80, 90, 110, 130, 150, 160, 180, 190, 210, 220, 230, 250, 260, 300)

KEY_TURNS = {
    40: "Pour reformuler mes phrases oui, mais jamais pour écrire un paragraphe entier.",
    60: "Je ne lui fais jamais écrire mes dissertations.",
    100: "Ma prof dirait que ce n'est pas du travail, alors je lui fais seulement vérifier mes références.",
    120: "Je préfère DeepL à ChatGPT pour traduire, il est plus précis.",
    140: "Je ne l'utilise pas pour les exercices de maths, sauf quand je bloque depuis une heure.",
    170: "Le soir je lui demande des recettes, enfin plutôt des listes de courses.",
    200: "Je lui demande des explications de cours. Bon, dit comme ça, ça a l'air trop simple ce que je raconte.",
    240: "J'utilise plutôt Gemini que ChatGPT pour les mails, c'est pareil.",
    270: "Je ne sais pas si je réponds bien à tes questions.",  # aucune pratique : signal sans pratique proche
    280: "Pour la dissertation d'histoire, je lui ai fait écrire la conclusion, seulement la conclusion.",
}
_FILLERS = ("Oui, voilà, c'est ça, comment dire, je fais comme tout le monde.",
            "Ça dépend des semaines (rires), mais en gros oui.",
            "D'accord, je vois ce que tu veux dire.")


def tid(number: int) -> str:
    return f"{INTERVIEW_ID}_T{number:04d}"


def task_of(number: int) -> tuple[str, str]:
    return _TASKS[ORDINARY_TURNS.index(number) % len(_TASKS)]


def turn_text(number: int) -> str:
    if number % 2:
        nxt = number + 1
        if nxt in ORDINARY_TURNS:
            return f"Et pour {task_of(nxt)[0]}, comment tu fais ?"
        return ("Tu peux m'en dire plus ?", "Et ensuite ?", "D'accord. Et concrètement ?")[(number // 2) % 3]
    if number in KEY_TURNS:
        return KEY_TURNS[number]
    if number in ORDINARY_TURNS:
        topic, action = task_of(number)
        tail = " C'est vraiment super pratique." if ORDINARY_TURNS.index(number) % 2 else " (rires) Voilà, comment dire."
        return f"Pour {topic}, je lui demande {action}.{tail}"
    return _FILLERS[(number // 2) % len(_FILLERS)]


def text() -> str:
    return "\n".join(f"{'Enquêteur' if n % 2 else 'Enquêté'} : {turn_text(n)}" for n in range(1, TURN_COUNT + 1)) + "\n"


def files() -> list[tuple[str, bytes]]:
    return [(FILENAME, text().encode("utf-8"))]


def ev(number: int, quote: str) -> dict:
    assert quote in turn_text(number), (number, quote)
    return {"turn_id": tid(number), "quote": quote}


def _p(number: int, quote: str, **fields) -> dict:
    return base_practice(turn_start=tid(number - 1), turn_end=tid(number), evidence=[ev(number, quote)], **fields)


def _s(number: int, signal_type: str, surface: str, quote: str, description: str, **fields) -> dict:
    return base_signal(turn_ids=[tid(number)], signal_type=signal_type, surface_form=surface, description=description,
                       evidence=[ev(number, quote)], **fields)


# --- Étape 3 simulée --------------------------------------------------------------------------

def _practices_by_turn() -> dict[int, list[dict]]:
    items: dict[int, list[dict]] = {}
    for n in ORDINARY_TURNS:
        topic, action = task_of(n)
        items[n] = [_p(n, f"Pour {topic}, je lui demande {action}.", summary=f"L'étudiant indique lui demander {action}.",
                       academic_task=topic)]
    items.update({
        40: [_p(40, "Pour reformuler mes phrases oui", summary="L'étudiant indique lui faire reformuler ses phrases.",
                academic_task="rédaction"),
             _p(40, "mais jamais pour écrire un paragraphe entier", summary="L'étudiant dit ne jamais lui faire écrire un paragraphe entier.",
                use_status="refusal", non_use_reason="personal_rule", academic_task="rédaction", stated_frequency="jamais")],
        60: [_p(60, "Je ne lui fais jamais écrire mes dissertations.", summary="L'étudiant dit ne jamais lui faire écrire ses dissertations.",
                use_status="non_use", non_use_reason="not_stated", academic_task="dissertations", stated_frequency="jamais")],
        100: [_p(100, "je lui fais seulement vérifier mes références", summary="L'étudiant indique lui faire vérifier ses références.",
                 academic_task="références bibliographiques", other_actors=["sa professeure"])],
        120: [_p(120, "Je préfère DeepL à ChatGPT pour traduire", summary="L'étudiant indique utiliser DeepL pour traduire.",
                 academic_task="traduction", ai_tool=["DeepL"])],
        140: [_p(140, "Je ne l'utilise pas pour les exercices de maths", summary="L'étudiant dit ne pas l'utiliser pour les exercices de maths.",
                 use_status="non_use", non_use_reason="not_stated", academic_task="exercices de maths"),
              _p(140, "sauf quand je bloque depuis une heure", summary="L'étudiant indique l'utiliser quand il bloque depuis une heure.",
                 academic_task="exercices de maths", stated_reason=["quand je bloque depuis une heure"])],
        170: [_p(170, "Le soir je lui demande des recettes", summary="L'étudiant indique lui demander des recettes le soir.",
                 practice_domain="personal", assessment_context="personal")],
        200: [_p(200, "Je lui demande des explications de cours.", summary="L'étudiant indique lui demander des explications de cours.",
                 academic_task="cours")],
        240: [_p(240, "J'utilise plutôt Gemini que ChatGPT pour les mails", summary="L'étudiant indique utiliser Gemini pour ses mails.",
                 academic_task="mails", ai_tool=["Gemini"])],
        280: [_p(280, "Pour la dissertation d'histoire, je lui ai fait écrire la conclusion",
                 summary="L'étudiant indique lui avoir fait écrire la conclusion d'une dissertation d'histoire.",
                 use_status="past_use", academic_task="dissertation", discipline="histoire")],
    })
    return items


def _signals_by_turn() -> dict[int, list[dict]]:
    items: dict[int, list[dict]] = {}
    for i, n in enumerate(ORDINARY_TURNS):
        if i % 2:
            items[n] = [_s(n, "intensification", "vraiment super pratique", "C'est vraiment super pratique.",
                           "L'enquêté qualifie l'usage de « vraiment super pratique »."),
                        _s(n, "generalization", "c'est pratique", "super pratique", "Qualification générale de l'usage.")]
        else:
            items[n] = [_s(n, "transcribed_laughter", "(rires)", "(rires)", "Rire transcrit."),
                        _s(n, "hesitation", "comment dire", "comment dire", "L'enquêté cherche ses mots.")]
    for n in range(2, TURN_COUNT + 1, 2):
        if n in items or n in KEY_TURNS:
            continue
        if turn_text(n) == _FILLERS[0] and n % 4 == 0:
            items[n] = [_s(n, "generalization", "comme tout le monde", "je fais comme tout le monde",
                           "L'enquêté rapporte sa manière de faire à celle de tout le monde.")]
        elif turn_text(n) == _FILLERS[1] and n % 3 == 0:
            items[n] = [_s(n, "transcribed_laughter", "(rires)", "(rires)", "Rire transcrit.")]
    items.update({
        40: [_s(40, "restriction", "oui, mais jamais pour écrire", "Pour reformuler mes phrases oui, mais jamais pour écrire un paragraphe entier.",
                "L'enquêté accepte la reformulation et exclut l'écriture d'un paragraphe entier.")],
        100: [_s(100, "reference_to_teacher_judgment", "Ma prof dirait", "Ma prof dirait que ce n'est pas du travail",
                 "L'enquêté rapporte ce que sa professeure dirait."),
              _s(100, "restriction", "seulement vérifier", "je lui fais seulement vérifier mes références",
                 "L'enquêté limite l'usage à la vérification des références.")],
        120: [_s(120, "preference_statement", "Je préfère DeepL à ChatGPT", "Je préfère DeepL à ChatGPT pour traduire, il est plus précis.",
                 "L'enquêté formule une préférence entre deux outils.")],
        140: [_s(140, "exception", "sauf quand je bloque", "sauf quand je bloque depuis une heure",
                 "L'enquêté signale une exception au non-usage.")],
        170: [_s(170, "self_reformulation", "enfin plutôt des listes de courses", "enfin plutôt des listes de courses",
                 "L'enquêté reformule l'objet de sa demande.")],
        200: [_s(200, "metadiscursive_self_evaluation", "dit comme ça, ça a l'air trop simple",
                 "dit comme ça, ça a l'air trop simple ce que je raconte", "L'enquêté qualifie ce qu'il raconte.")],
        240: [_s(240, "contrast", "plutôt Gemini que ChatGPT", "J'utilise plutôt Gemini que ChatGPT pour les mails",
                 "L'enquêté oppose deux outils pour ses mails.")],
        270: [_s(270, "metadiscursive_self_evaluation", "si je réponds bien", "Je ne sais pas si je réponds bien à tes questions.",
                 "L'enquêté s'interroge sur ses réponses.")],
        280: [_s(280, "restriction", "seulement la conclusion", "seulement la conclusion",
                 "L'enquêté limite l'écriture par l'outil à la conclusion.")],
    })
    return items


PRACTICES = _practices_by_turn()
SIGNALS = _signals_by_turn()
EXPECTED_PRACTICE_COUNT = sum(len(v) for v in PRACTICES.values())
EXPECTED_LOCAL_SIGNAL_COUNT = sum(len(v) for v in SIGNALS.values())
CONTRADICTION = base_signal(
    turn_ids=[tid(60), tid(280)], signal_type="cross_turn_contradiction", surface_form="jamais / je lui ai fait écrire",
    description="Au tour T0060, l'enquêté dit ne jamais lui faire écrire ses dissertations ; au tour T0280, il dit lui "
                "avoir fait écrire la conclusion d'une dissertation.",
    topic="écriture des dissertations", cross_turn_reference="Faire écrire une dissertation.",
    evidence=[ev(60, "Je ne lui fais jamais écrire mes dissertations."),
              ev(280, "je lui ai fait écrire la conclusion")])


def _present(params: dict) -> set[str]:
    return {t["turn_id"] for t in sent_turns(params)}


def _covered(item: dict, present: set[str]) -> bool:
    turns = {e["turn_id"] for e in item["evidence"]} | set(item.get("turn_ids", []))
    turns |= {item[k] for k in ("turn_start", "turn_end") if k in item}
    return turns <= present


def practice_reader(params: dict):
    present = _present(params)
    items = [p for n, ps in sorted(PRACTICES.items()) for p in ps if _covered(p, present)]
    return text_response({"practices": items, "extraction_notes": None}, input_tokens=4500, output_tokens=2500)


def signal_reader(params: dict):
    present = _present(params)
    items = [s for n, ss in sorted(SIGNALS.items()) for s in ss if _covered(s, present)]
    return text_response({"signals": items, "reading_notes": None}, input_tokens=5500, output_tokens=2500)


def long_distance_reader(params: dict):
    present = _present(params)
    items = [CONTRADICTION] if _covered(CONTRADICTION, present) else []
    return text_response({"signals": items, "reading_notes": None}, input_tokens=2000, output_tokens=400)


def stage3_responders():
    from tests.fake_llm import AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE
    return {PRACTICE: practice_reader, INTERACTION: signal_reader, LONG_DISTANCE: long_distance_reader,
            AUDITOR: lambda p: text_response({"assessments": [], "audit_notes": None})}


# --- Étape 4 simulée --------------------------------------------------------------------------

ACCOUNTABILITY_ANCHORS = (40, 60, 100, 140, 200)
ORDINARY_EXAMINED_ANCHORS = (120, 170, 240)

RULES = {
    40: decision("accountability_episode", "L'étudiant accepte la reformulation de ses phrases et exclut l'écriture d'un paragraphe entier.",
                 moves=[move("restriction", "L'étudiant limite l'usage à la reformulation de ses phrases.", 40),
                        move("refusal", "L'étudiant exclut l'écriture d'un paragraphe entier.", 40)],
                 boundary=["écrire / reformuler"], problem="Jusqu'où l'outil peut-il intervenir dans l'écriture ?"),
    60: decision("accountability_episode",
                 "L'étudiant dit ne jamais faire écrire ses dissertations, puis rapporte avoir fait écrire la conclusion "
                 "d'une dissertation d'histoire, en la limitant (« seulement la conclusion »).",
                 moves=[move("general_rule", "L'étudiant énonce qu'il ne fait jamais écrire ses dissertations.", 60),
                        move("exception", "L'étudiant rapporte une conclusion écrite par l'outil.", 280),
                        move("restriction", "L'étudiant limite ce passage à la conclusion.", 280)],
                 boundary=["faire / faire faire"], problem="Qui écrit une dissertation ?",
                 evidence=[(60, "Je ne lui fais jamais écrire mes dissertations."),
                           (280, "je lui ai fait écrire la conclusion, seulement la conclusion")]),
    100: decision("accountability_episode",
                  "L'étudiant rapporte ce que sa professeure dirait et limite l'usage à la vérification de ses références.",
                  moves=[move("appeal_to_external_judgment", "L'étudiant évoque ce que sa professeure dirait.", 100),
                         move("restriction", "L'étudiant limite l'usage à la vérification des références.", 100)],
                  external="sa professeure", problem="Qu'est-ce qui compte comme du travail ?",
                  evidence=[(100, "Ma prof dirait que ce n'est pas du travail, alors je lui fais seulement vérifier mes références.")]),
    140: decision("accountability_episode",
                  "L'étudiant dit ne pas l'utiliser pour les exercices de maths et présente un usage comme exception.",
                  moves=[move("refusal", "L'étudiant dit ne pas l'utiliser pour les exercices de maths.", 140),
                         move("exception", "L'étudiant signale une exception : quand il bloque depuis une heure.", 140)],
                  problem="Quand l'outil peut-il intervenir dans un exercice ?"),
    200: decision("accountability_episode", "L'étudiant indique demander des explications de cours, puis évalue ce qu'il raconte.",
                  moves=[move("self_evaluation", "L'étudiant qualifie ce qu'il raconte de « trop simple ».", 200)],
                  problem="Comment dire la demande d'explications ?", confidence="medium",
                  evidence=[(200, "Je lui demande des explications de cours."),
                            (200, "dit comme ça, ça a l'air trop simple ce que je raconte")]),
    120: decision("ordinary_practice", "L'étudiant raconte traduire avec DeepL ; la préférence porte sur l'outil.",
                  confidence="medium"),
    170: decision("ordinary_practice", "L'étudiant raconte demander des recettes ; la reformulation porte sur l'objet de la demande.",
                  confidence="medium"),
    240: decision("ordinary_practice", "L'étudiant raconte écrire ses mails avec Gemini ; le contraste porte sur l'outil.",
                  confidence="medium"),
}
BUILDER = scripted_builder(RULES)
