"""Entretiens SYNTHÉTIQUES de l'étape 4 et réponses simulées (aucun vrai entretien, aucun appel réel).

Entretien de référence (ENTRETIEN_ETAPE_4, 25 tours) :
A. T0002 « Pour réviser je lui demande des explications… » : aucun trouble → pratique ordinaire ;
B. T0004 « Pour reformuler oui, mais je ne veux pas qu'il écrive à ma place. » → épisode
   (restriction, distinction, refus) ;
C. T0006 « Normalement je fais mes plans moi-même. » … T0012 « Une fois je lui ai demandé un plan
   parce que j'étais en retard. » → épisode à distance (règle générale / exception) ;
D. T0014 « J'aurais peur que le professeur pense que je n'ai rien fait. » → jugement d'autrui ;
E. T0016 « Dit comme ça c'est un peu facile ce que je dis. » → auto-évaluation ;
F. T0018 recettes de cuisine, idées de films → pratique ordinaire ;
G. T0020 préférence entre deux outils de correction → candidat examiné, pratique ordinaire ;
H. T0022 tour marqué « Enquêteur » qui est une réponse à la première personne → audit des
   locuteurs, épisode incertain, avertissement propagé ;
I. T0023 question de l'enquêteur proposant la catégorie « triche », réponse évasive (T0024).

Les sorties de l'étape 3 sont simulées (Practice Extractor, Interaction Reader, audit des
locuteurs) ; l'étape 4 est simulée par `scripted_builder`, un lecteur DÉTERMINISTE qui lit les
candidats réellement envoyés et applique une table de décisions (par tour d'ancrage).
"""

from __future__ import annotations

import json
import re

from tests.fake_llm import text_response
from tests.synthetic_interviews import assessment, practice, signal

FILENAME = "Entretien_etape_4.txt"
INTERVIEW_ID = "ENTRETIEN_ETAPE_4"

TURNS = [
    ("Enquêteur", "Est-ce que tu utilises ChatGPT pour tes études ?"),
    ("Enquêté", "Oui. Pour réviser je lui demande des explications sur les notions du cours."),
    ("Enquêteur", "Et pour écrire tes devoirs ?"),
    ("Enquêté", "Pour reformuler oui, mais je ne veux pas qu'il écrive à ma place."),
    ("Enquêteur", "Et les plans de dissertation ?"),
    ("Enquêté", "Normalement je fais mes plans moi-même."),
    ("Enquêteur", "D'accord. Et pour chercher des sources ?"),
    ("Enquêté", "Je lui demande des idées de lectures, et après je vais à la bibliothèque."),
    ("Enquêteur", "Et pour les langues ?"),
    ("Enquêté", "Je lui fais traduire des articles en anglais quand ils sont longs."),
    ("Enquêteur", "Il t'est déjà arrivé de lui demander un plan ?"),
    ("Enquêté", "Une fois je lui ai demandé un plan parce que j'étais en retard."),
    ("Enquêteur", "Et pour les dossiers à rendre ?"),
    ("Enquêté", "Je ne lui fais pas corriger mes dossiers. J'aurais peur que le professeur pense que je n'ai rien fait."),
    ("Enquêteur", "Et pour les fiches de lecture ?"),
    ("Enquêté", "Je lui demande des résumés des textes. Dit comme ça c'est un peu facile ce que je dis."),
    ("Enquêteur", "Et en dehors des études ?"),
    ("Enquêté", "Le week-end je lui demande des recettes de cuisine ou des idées de films."),
    ("Enquêteur", "Et pour tes mails ?"),
    ("Enquêté", "Je préfère lui demander de corriger l'orthographe, c'est plus rapide que le correcteur du téléphone."),
    ("Enquêteur", "Et les exposés ?"),
    ("Enquêteur", "Moi personnellement je lui fais écrire mes intros d'exposé, enfin ça dépend des fois."),
    ("Enquêteur", "D'accord. Tu ne trouves pas que c'est de la triche, tout ça ?"),
    ("Enquêté", "Je sais pas trop."),
    ("Enquêteur", "Merci beaucoup pour cet entretien."),
]
TEXT = "\n".join(f"{speaker} : {text}" for speaker, text in TURNS) + "\n"
FILES = [(FILENAME, TEXT.encode("utf-8"))]
WARNED_TURN = 22
INTERVIEWER_CATEGORY_TURN = 23


def tid(number: int, interview_id: str = INTERVIEW_ID) -> str:
    return f"{interview_id}_T{number:04d}"


def ev(number: int, quote: str) -> dict:
    assert quote in TURNS[number - 1][1], (number, quote)
    return {"turn_id": tid(number), "quote": quote}


def _p(start: int, end: int, quote_turn: int, quotes: list[str], **fields) -> dict:
    return practice(turn_start=tid(start), turn_end=tid(end), evidence=[ev(quote_turn, q) for q in quotes], **fields)


# --- Étape 3 simulée ------------------------------------------------------------------------

STAGE3_PRACTICES = {"practices": [
    _p(1, 2, 2, ["Pour réviser je lui demande des explications sur les notions du cours."],
       summary="L'étudiant indique lui demander des explications pour réviser.", academic_task="révisions",
       ai_tool=["ChatGPT"], ai_action=["explique des notions du cours"]),
    _p(3, 4, 4, ["Pour reformuler oui"], summary="L'étudiant indique l'utiliser pour reformuler ses devoirs.",
       academic_task="devoirs", ai_action=["reformule"]),
    _p(3, 4, 4, ["je ne veux pas qu'il écrive à ma place"], summary="L'étudiant dit ne pas vouloir qu'il écrive à sa place.",
       use_status="refusal", non_use_reason="personal_rule", academic_task="devoirs"),
    _p(5, 6, 6, ["Normalement je fais mes plans moi-même."], summary="L'étudiant indique faire normalement ses plans lui-même.",
       use_status="non_use", non_use_reason="not_stated", academic_task="plans de dissertation"),
    _p(7, 8, 8, ["Je lui demande des idées de lectures"], summary="L'étudiant indique lui demander des idées de lectures.",
       academic_task="recherche de sources", student_action_after=["va à la bibliothèque"]),
    _p(9, 10, 10, ["Je lui fais traduire des articles en anglais quand ils sont longs."],
       summary="L'étudiant indique lui faire traduire des articles en anglais quand ils sont longs.",
       academic_task="traduction", stated_reason=["quand ils sont longs"]),
    _p(11, 12, 12, ["Une fois je lui ai demandé un plan parce que j'étais en retard."],
       summary="L'étudiant indique lui avoir demandé un plan une fois, parce qu'il était en retard.",
       use_status="past_use", academic_task="plan de dissertation", stated_reason=["parce que j'étais en retard"],
       stated_frequency="une fois"),
    _p(13, 14, 14, ["Je ne lui fais pas corriger mes dossiers.",
                    "J'aurais peur que le professeur pense que je n'ai rien fait."],
       summary="L'étudiant indique ne pas lui faire corriger ses dossiers.", use_status="non_use",
       non_use_reason="other", academic_task="dossiers",
       stated_reason=["J'aurais peur que le professeur pense que je n'ai rien fait."]),
    _p(15, 16, 16, ["Je lui demande des résumés des textes."], summary="L'étudiant indique lui demander des résumés des textes.",
       academic_task="fiches de lecture"),
    _p(17, 18, 18, ["Le week-end je lui demande des recettes de cuisine ou des idées de films."],
       summary="L'étudiant indique lui demander des recettes et des idées de films le week-end.",
       practice_domain="personal", assessment_context="personal"),
    _p(19, 20, 20, ["Je préfère lui demander de corriger l'orthographe"],
       summary="L'étudiant indique lui demander de corriger l'orthographe de ses mails.", academic_task="mails",
       stated_reason=["c'est plus rapide que le correcteur du téléphone"]),
    _p(21, 22, 22, ["je lui fais écrire mes intros d'exposé"],
       summary="Le passage décrit l'écriture d'introductions d'exposé par l'outil.", academic_task="exposés",
       explicitness="unclear", uncertainty_note="Le tour T0022 est marqué enquêteur : attribution du locuteur douteuse."),
], "extraction_notes": None}

STAGE3_SIGNALS = {"signals": [
    signal(turn_ids=[tid(4)], signal_type="restriction", surface_form="Pour reformuler oui, mais",
           description="L'enquêté accepte la reformulation et exclut l'écriture à sa place.",
           evidence=[ev(4, "Pour reformuler oui, mais je ne veux pas qu'il écrive à ma place.")]),
    signal(turn_ids=[tid(6), tid(12)], signal_type="cross_turn_contradiction", surface_form="Normalement / Une fois",
           description="Au tour T0006, l'enquêté dit faire normalement ses plans lui-même ; au tour T0012, il dit lui "
                       "avoir demandé un plan une fois.",
           topic="plans de dissertation", cross_turn_reference="Faire ses plans soi-même.",
           evidence=[ev(6, "Normalement je fais mes plans moi-même."), ev(12, "Une fois je lui ai demandé un plan")]),
    signal(turn_ids=[tid(12)], signal_type="exception", surface_form="Une fois",
           description="L'enquêté présente la demande de plan comme unique.",
           evidence=[ev(12, "Une fois je lui ai demandé un plan")]),
    signal(turn_ids=[tid(14)], signal_type="explicit_emotion", surface_form="J'aurais peur",
           description="L'enquêté emploie « peur » à propos de ce que le professeur pourrait penser.",
           explicit_affect="peur", evidence=[ev(14, "J'aurais peur que le professeur pense que je n'ai rien fait.")]),
    signal(turn_ids=[tid(14)], signal_type="reference_to_teacher_judgment", surface_form="que le professeur pense",
           description="L'enquêté évoque ce que le professeur pourrait penser.",
           evidence=[ev(14, "que le professeur pense que je n'ai rien fait")]),
    signal(turn_ids=[tid(16)], signal_type="metadiscursive_self_evaluation", surface_form="c'est un peu facile ce que je dis",
           description="L'enquêté qualifie de « un peu facile » ce qu'il vient de dire.",
           evidence=[ev(16, "Dit comme ça c'est un peu facile ce que je dis.")]),
    signal(turn_ids=[tid(20)], signal_type="preference_statement", surface_form="Je préfère",
           description="L'enquêté formule une préférence pour la correction par l'outil plutôt que par le correcteur du téléphone.",
           evidence=[ev(20, "Je préfère lui demander de corriger l'orthographe, c'est plus rapide que le correcteur du téléphone.")]),
    signal(turn_ids=[tid(22)], signal_type="self_correction", surface_form="enfin ça dépend des fois",
           description="Le passage revient sur ce qui vient d'être dit. L'attribution du locuteur de ce tour est douteuse.",
           needs_human_review=True, evidence=[ev(22, "enfin ça dépend des fois")]),
    signal(turn_ids=[tid(24)], signal_type="hesitation", surface_form="Je sais pas trop",
           description="L'enquêté répond qu'il ne sait pas trop.", evidence=[ev(24, "Je sais pas trop.")]),
    # Lecteur maladroit : un signal appuyé sur la seule question de l'enquêteur (écarté par l'étape 3.7).
    signal(turn_ids=[tid(23)], signal_type="normative_formulation", surface_form="c'est de la triche",
           description="La question qualifie l'usage.", evidence=[ev(23, "Tu ne trouves pas que c'est de la triche, tout ça ?")]),
], "reading_notes": None}

STAGE3_AUDIT = {"assessments": [
    assessment(turn_id=tid(22), suggested_speaker="enquete", confidence="high",
               reason="Réponse à la première personne qui suit directement la question du tour précédent.",
               evidence=[ev(22, "Moi personnellement je lui fais écrire mes intros d'exposé, enfin ça dépend des fois."),
                         ev(21, "Et les exposés ?")]),
], "audit_notes": None}


def stage3_responders(audit=True):
    from tests.fake_llm import AUDITOR, INTERACTION, PRACTICE
    responders = {PRACTICE: lambda p: text_response(STAGE3_PRACTICES),
                  INTERACTION: lambda p: text_response(STAGE3_SIGNALS)}
    if audit:
        responders[AUDITOR] = lambda p: text_response(STAGE3_AUDIT)
    return responders


# --- Étape 4 simulée : lecteur DÉTERMINISTE des candidats envoyés ----------------------------

_CANDIDATES_RE = re.compile(r"<candidates>\n(.*)\n</candidates>", re.DOTALL)
_TURN_NUMBER = re.compile(r"T(\d+)$")  # identifiant complet ou abrégé (représentation normalisée)


def sent_payload(params: dict) -> dict:
    """Candidats, pratiques, signaux et tours réellement transmis à l'Accountability Episode Builder."""
    return json.loads(_CANDIDATES_RE.search(params["messages"][0]["content"]).group(1))


def turn_number(turn_id: str) -> int:
    return int(_TURN_NUMBER.search(turn_id).group(1))


def move(move_type: str, description: str, *turns: int) -> tuple:
    return move_type, description, turns


def decision(status: str, summary: str, moves=(), boundary=(), problem=None, external=None, role=None,
             confidence="high", needs_review=False, evidence=None) -> dict:
    return {"status": status, "summary": summary, "moves": list(moves), "boundary": list(boundary), "problem": problem,
            "external": external, "role": role, "confidence": confidence, "needs_review": needs_review,
            "evidence": evidence}


def practice_quotes(payload: dict, practice_id: str) -> list[dict]:
    """Citations d'une pratique dans la représentation normalisée (format 2) : evidence → evidence_by_id."""
    return [payload["evidence_by_id"][ref] for ref in payload["practices_by_id"][practice_id]["evidence"]]


def build_episode(candidate: dict, payload: dict, rule: dict) -> dict:
    interview_id = payload["interview_id"]
    turn = lambda n: f"{interview_id}_T{n:04d}"  # noqa: E731
    quotes = [q for pid in candidate["practice_ids"] for q in practice_quotes(payload, pid)]
    evidence = ([{"turn_id": turn(n), "quote": q} for n, q in rule["evidence"]] if rule.get("evidence")
                else list({(q["turn_id"], q["quote"]): q for q in quotes}.values()))
    numbers = sorted(turn_number(t) for t in candidate["turn_ids"])
    return {
        "candidate_ids": [candidate["candidate_id"]], "turn_start": turn(numbers[0]), "turn_end": turn(numbers[-1]),
        "practice_ids": candidate["practice_ids"], "signal_ids": candidate["signal_ids"],
        "episode_status": rule["status"], "accountability_problem": rule["problem"],
        "accounting_moves": [{"type": t, "description": d, "evidence_turn_ids": [turn(n) for n in ns]}
                             for t, d, ns in rule["moves"]],
        "boundary_objects": rule["boundary"], "student_role_reference": rule["role"],
        "external_reference": rule["external"], "episode_summary": rule["summary"], "confidence": rule["confidence"],
        "needs_review": rule["needs_review"], "evidence": evidence,
    }


def anchor(candidate: dict, payload: dict) -> int:
    return min(turn_number(q["turn_id"]) for pid in candidate["practice_ids"] for q in practice_quotes(payload, pid))


UNKNOWN = decision("uncertain", "Le matériau de ce candidat n'a pas été prévu par le lecteur simulé.",
                   confidence="low", needs_review=True)


def scripted_output(params: dict, rules: dict[int, dict]) -> dict:
    payload = sent_payload(params)
    episodes = [build_episode(c, payload, rules.get(anchor(c, payload), UNKNOWN)) for c in payload["candidates"]]
    return {"episodes": episodes, "builder_notes": None}


def scripted_builder(rules: dict[int, dict], output_tokens_per_episode: int = 260):
    def respond(params: dict):
        output = scripted_output(params, rules)
        return text_response(output)
    return respond


REFERENCE_RULES = {
    4: decision("accountability_episode",
                "L'étudiant accepte que l'outil reformule et exclut qu'il écrive à sa place.",
                moves=[move("restriction", "L'étudiant limite l'usage à la reformulation (« Pour reformuler oui »).", 4),
                       move("distinction", "L'étudiant sépare reformuler et écrire.", 4),
                       move("refusal", "L'étudiant dit ne pas vouloir que l'outil écrive à sa place.", 4)],
                boundary=["écrire / reformuler", "faire / faire faire"],
                problem="Jusqu'où l'outil peut-il intervenir dans l'écriture d'un devoir ?",
                evidence=[(4, "Pour reformuler oui, mais je ne veux pas qu'il écrive à ma place.")]),
    6: decision("accountability_episode",
                "L'étudiant dit faire normalement ses plans lui-même, puis rapporte une demande de plan présentée comme "
                "unique (« Une fois »), avec la raison qu'il donne : « parce que j'étais en retard ».",
                moves=[move("general_rule", "L'étudiant énonce ce qu'il fait d'ordinaire pour ses plans.", 6),
                       move("exception", "L'étudiant présente la demande de plan comme une seule fois, liée au retard.", 12)],
                boundary=["faire / faire faire"],
                problem="Qui fait le plan d'une dissertation ?",
                evidence=[(6, "Normalement je fais mes plans moi-même."),
                          (12, "Une fois je lui ai demandé un plan parce que j'étais en retard.")]),
    14: decision("accountability_episode",
                 "L'étudiant dit ne pas faire corriger ses dossiers et évoque ce que le professeur pourrait penser : "
                 "« que je n'ai rien fait ».",
                 moves=[move("refusal", "L'étudiant dit ne pas lui faire corriger ses dossiers.", 14),
                        move("appeal_to_external_judgment", "L'étudiant rapporte ce non-usage à ce que le professeur pourrait penser.", 14),
                        move("appeal_to_authorship", "Le jugement évoqué porte sur le fait d'avoir fait ou non le travail.", 14)],
                 problem="Un dossier corrigé par l'outil reste-t-il le travail de l'étudiant ?",
                 external="le professeur",
                 evidence=[(14, "Je ne lui fais pas corriger mes dossiers."),
                           (14, "J'aurais peur que le professeur pense que je n'ai rien fait.")]),
    16: decision("accountability_episode",
                 "L'étudiant indique demander des résumés des textes, puis qualifie sa propre formulation de « un peu facile ».",
                 moves=[move("self_evaluation", "L'étudiant évalue ce qu'il vient de dire (« Dit comme ça c'est un peu facile »).", 16)],
                 problem="Comment dire la demande de résumés de textes ?", confidence="medium",
                 evidence=[(16, "Je lui demande des résumés des textes."),
                           (16, "Dit comme ça c'est un peu facile ce que je dis.")]),
    20: decision("ordinary_practice",
                 "L'étudiant raconte demander une correction orthographique ; la préférence énoncée porte sur l'outil, "
                 "sans réserve ni limite sur l'usage.", confidence="medium",
                 evidence=[(20, "Je préfère lui demander de corriger l'orthographe")]),
    22: decision("uncertain",
                 "Le tour T0022 est attribué à l'enquêteur dans la transcription ; l'audit y signale une réponse probable "
                 "de l'enquêté. L'épisode ne peut pas être établi.", confidence="low", needs_review=True,
                 evidence=[(22, "je lui fais écrire mes intros d'exposé")]),
}
REFERENCE_BUILDER = scripted_builder(REFERENCE_RULES)

# Statuts attendus, par tour d'ancrage (A et F : pratiques sans marqueur, jamais envoyées au modèle).
EXPECTED_STATUS_BY_ANCHOR = {4: "accountability_episode", 6: "accountability_episode", 14: "accountability_episode",
                             16: "accountability_episode", 20: "ordinary_practice", 22: "uncertain"}
EXPECTED_UNMARKED_TURNS = (2, 8, 10, 18)


# --- Entretien SANS épisode d'accountability ---------------------------------------------

PLAIN_ID = "ENTRETIEN_ORDINAIRE"
PLAIN_FILENAME = "Entretien_ordinaire.txt"
PLAIN_TURNS = [
    ("Enquêteur", "Tu utilises des IA génératives ?"),
    ("Enquêté", "Oui, je lui demande des définitions quand je lis un article."),
    ("Enquêteur", "Et pour les cours de langue ?"),
    ("Enquêté", "Je lui fais traduire des phrases que je ne comprends pas."),
    ("Enquêteur", "Et pour l'organisation ?"),
    ("Enquêté", "Je lui demande de faire mon planning de la semaine."),
    ("Enquêteur", "Et pour les mails ?"),
    ("Enquêté", "Je préfère Gemini à ChatGPT pour écrire mes mails, il est plus rapide."),
    ("Enquêteur", "Et en dehors ?"),
    ("Enquêté", "Il me conseille des séries à regarder."),
]
PLAIN_TEXT = "\n".join(f"{s} : {t}" for s, t in PLAIN_TURNS) + "\n"
PLAIN_FILES = [(PLAIN_FILENAME, PLAIN_TEXT.encode("utf-8"))]


def _pev(number: int, quote: str) -> dict:
    assert quote in PLAIN_TURNS[number - 1][1]
    return {"turn_id": tid(number, PLAIN_ID), "quote": quote}


def _pp(n: int, quote: str, summary: str, **fields) -> dict:
    return practice(turn_start=tid(n - 1, PLAIN_ID), turn_end=tid(n, PLAIN_ID), evidence=[_pev(n, quote)],
                    summary=summary, **fields)


PLAIN_PRACTICES = {"practices": [
    _pp(2, "je lui demande des définitions quand je lis un article", "L'étudiant indique lui demander des définitions.",
        academic_task="lectures"),
    _pp(4, "Je lui fais traduire des phrases que je ne comprends pas.", "L'étudiant indique lui faire traduire des phrases.",
        academic_task="cours de langue"),
    _pp(6, "Je lui demande de faire mon planning de la semaine.", "L'étudiant indique lui faire faire son planning.",
        academic_task="organisation"),
    _pp(8, "Je préfère Gemini à ChatGPT pour écrire mes mails", "L'étudiant indique utiliser Gemini pour écrire ses mails.",
        academic_task="mails", ai_tool=["Gemini"]),
    _pp(10, "Il me conseille des séries à regarder.", "L'étudiant indique qu'il lui conseille des séries.",
        practice_domain="personal"),
], "extraction_notes": None}
PLAIN_SIGNALS = {"signals": [
    signal(turn_ids=[tid(8, PLAIN_ID)], signal_type="preference_statement", surface_form="Je préfère Gemini à ChatGPT",
           description="L'enquêté formule une préférence entre deux outils.",
           evidence=[_pev(8, "Je préfère Gemini à ChatGPT pour écrire mes mails, il est plus rapide.")]),
    signal(turn_ids=[tid(6, PLAIN_ID)], signal_type="intensification", surface_form="de la semaine",
           description="Intensification sans opération identifiable.",
           evidence=[_pev(6, "de la semaine")]),
], "reading_notes": None}
PLAIN_RULES = {8: decision("ordinary_practice",
                           "L'étudiant raconte écrire ses mails avec Gemini ; la préférence porte sur l'outil, sans réserve "
                           "ni limite sur l'usage.", confidence="medium")}


def plain_responders():
    from tests.fake_llm import INTERACTION, PRACTICE
    return {PRACTICE: lambda p: text_response(PLAIN_PRACTICES), INTERACTION: lambda p: text_response(PLAIN_SIGNALS)}
