"""Entretien SYNTHÉTIQUE long « OTMANE-like » pour l'étape 5 (190 tours). Aucune donnée réelle.

Propriétés (section 19 de la consigne de l'étape 5) :
- 50 pratiques d'étape 3 (lecture par blocs, lecture à longue distance), ~20 épisodes d'étape 4 et
  ~20 pratiques sans marqueur (recherches d'information, vie personnelle) ;
- 2 règles répétées : « juste reformuler, jamais écrire » (dissertations, notes de synthèse, rapports
  de stage) et « uniquement quand je manque de temps » (fiches, exposés, TD) ;
- 2 variations contextuelles : maths (réponse demandée) / dissertations (écriture gardée) ; dossiers
  notés (non-usage) / vie personnelle ;
- 1 vraie évolution temporelle : « Au lycée, je lui faisais rédiger… » / « Maintenant, je ne lui fais
  plus rien rédiger… » ;
- 1 exception : « Je ne fais jamais faire mes plans » / « Une fois, faute de temps… » ;
- 1 tension, entre deux tours éloignés : « Je vérifie toujours tout » / « je ne les vérifie pas »
  (contradiction relevée par la lecture à longue distance) ; la question de l'enquêteur qui précède
  propose « coupable », l'enquêtée ne reprend pas le mot ;
- quelques épisodes à revoir : un tour mal attribué (audit des locuteurs) et un épisode signalé par
  le modèle de l'étape 4.

Les lecteurs simulés de l'étape 3 suivent les blocs envoyés (tests/synthetic_stage4_otmane.py) ; l'étape 4
est simulée par tests/synthetic_stage5.component_builder ; l'étape 5 par un plan « par tour ».
"""

from __future__ import annotations

from tests import synthetic_stage4 as S4
from tests import synthetic_stage5 as S5
from tests.fake_llm import AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, text_response
from tests.synthetic_interviews import assessment as base_assessment
from tests.synthetic_interviews import practice as base_practice
from tests.synthetic_interviews import signal as base_signal
from tests.synthetic_long_interview import sent_turns
from tests.synthetic_stage4_otmane import _covered

FILENAME = "Etape5_long.txt"
INTERVIEW_ID = "ETAPE5_LONG"
TURN_COUNT = 190
MISATTRIBUTED_TURN = 150

FILLERS = (
    "Après, ça dépend beaucoup des semaines et de la charge de travail, parce qu'en licence on a souvent plusieurs "
    "rendus en même temps, des lectures à préparer, des exposés à organiser avec des camarades qui n'ont pas les "
    "mêmes horaires, et il faut aussi gérer le travail à côté, les trajets et les repas.",
    "Les semaines de partiels, on révise en groupe à la bibliothèque, on se partage les fiches, on se pose des "
    "questions les uns aux autres, et ces semaines-là je dors moins et je mange un peu n'importe comment, mais on "
    "tient le rythme en se fixant des objectifs pour chaque matinée.",
    "Mes parents me demandent souvent comment je m'organise, parce qu'ils n'ont pas fait d'études longues, alors je "
    "leur explique le fonctionnement des cours magistraux, des travaux dirigés et des rendus, et ça m'aide aussi à "
    "voir ce qui marche pour moi et ce qui ne marche pas.",
)
QUESTIONS = ("Et ensuite ?", "Tu peux m'en dire plus ?", "D'accord. Et concrètement ?", "Comment ça se passe pour toi ?")
SPECIAL_QUESTIONS = {
    111: "Et maintenant que tu es en licence ?",
    125: "Ça ne t'est jamais arrivé pour un plan ?",
    149: "Et pour les mémos ?",
    169: "Tu te sens coupable de ne pas vérifier les références ?",
}

# Pratiques sans marqueur : tour → (tâche, demande, domaine)
ORDINARY = {
    2: ("les horaires de la bibliothèque", "les horaires d'ouverture", "academic"),
    4: ("les définitions", "la définition des notions que je découvre en cours", "academic"),
    6: ("les sorties", "des idées de restaurants", "personal"),
    12: ("les trajets", "le meilleur itinéraire en métro", "personal"),
    14: ("les jeux de société", "les règles du jeu", "personal"),
    16: ("les salles de travail", "où trouver une salle libre près de la fac", "academic"),
    24: ("l'actualité", "un résumé des informations du jour", "personal"),
    26: ("le budget", "une formule de tableur pour mes dépenses", "personal"),
    32: ("les synonymes", "des synonymes quand je cherche un mot", "academic"),
    34: ("les repères historiques", "la date des événements dont parle le cours", "academic"),
    44: ("la cuisine", "des recettes avec ce qu'il reste dans le frigo", "personal"),
    54: ("les cadeaux", "des idées de cadeaux", "personal"),
    56: ("les inscriptions", "les étapes pour m'inscrire aux examens", "academic"),
    64: ("le sport", "un programme d'entraînement", "personal"),
    74: ("les films", "des idées de films", "personal"),
    84: ("les abréviations", "le sens des abréviations dans les articles", "academic"),
    94: ("les auteurs", "qui est un auteur cité en cours", "academic"),
    104: ("les voyages", "des idées de week-end", "personal"),
    106: ("la musique", "des playlists pour travailler", "personal"),
    116: ("les associations", "la liste des associations du campus", "academic"),
    136: ("les stages", "des entreprises où postuler", "academic"),
    146: ("le logement", "les démarches pour les aides au logement", "personal"),
}
R1 = {10: "les dissertations", 40: "les notes de synthèse", 70: "les rapports de stage"}
R2 = {20: "les fiches de révision", 50: "les exposés", 80: "les TD de statistiques"}
KEY_TURNS = {
    30: "Je vérifie toujours tout ce qu'il me donne, je ne lui fais pas confiance.",
    36: "Pour le vocabulaire d'anglais, je lui demande des listes, mais je les apprends moi-même, sinon je ne retiens rien.",
    46: "Quand il m'explique un théorème, je refais toujours la démonstration seule pour comprendre.",
    60: "Je préfère Gemini à ChatGPT pour les mails, il est plus rapide.",
    66: "Je préfère DeepL à ChatGPT pour traduire, il est plus précis.",
    76: "Pour les mails à l'administration, je lui fais juste corriger les fautes.",
    86: "Mes camarades l'utilisent tous pour les QCM, et moi aussi pour m'entraîner.",
    90: "En maths, je lui demande directement la solution des exercices.",
    100: "Pour les dossiers notés, je ne l'utilise pas du tout, je préfère tout faire moi-même.",
    110: "Au lycée, je lui faisais rédiger mes commentaires de texte en entier.",
    112: "Maintenant, je ne lui fais plus rien rédiger, je lui demande seulement de relire mes commentaires de texte.",
    120: "Je ne fais jamais faire mes plans, c'est mon travail.",
    126: "Une fois, faute de temps, je lui ai demandé un plan détaillé.",
    150: "Moi personnellement je lui fais seulement corriger l'orthographe de mes mémos, enfin ça dépend des fois.",
    160: "Ma prof dirait que ce n'est pas mon travail, alors je lui fais seulement vérifier mes références bibliographiques.",
    170: "Non. Les références qu'il me donne, franchement, je ne les vérifie pas.",
    176: "Je lui demande des quiz. Bon, c'est un peu facile ce que je dis.",
    180: "Je ne veux pas qu'il réfléchisse à ma place pour le mémoire.",
}


def tid(number: int) -> str:
    return f"{INTERVIEW_ID}_T{number:04d}"


def speaker(number: int) -> str:
    return "Enquêteur" if number % 2 or number == MISATTRIBUTED_TURN else "Enquêté"


def turn_text(number: int) -> str:
    if number % 2:
        return SPECIAL_QUESTIONS.get(number, QUESTIONS[(number // 2) % 4])
    if number in ORDINARY:
        task, ask, _ = ORDINARY[number]
        return f"Pour {task}, je lui demande {ask}."
    if number in R1:
        return f"Pour {R1[number]}, je lui fais juste reformuler des phrases, jamais écrire."
    if number in R2:
        return f"Pour {R2[number]}, je l'utilise uniquement quand je manque de temps."
    if number in KEY_TURNS:
        return KEY_TURNS[number]
    return FILLERS[(number // 2) % 3]


def text() -> str:
    return "\n".join(f"{speaker(n)} : {turn_text(n)}" for n in range(1, TURN_COUNT + 1)) + "\n"


def files() -> list[tuple[str, bytes]]:
    return [(FILENAME, text().encode("utf-8"))]


def ev(number: int, quote: str) -> dict:
    assert quote in turn_text(number), (number, quote)
    return {"turn_id": tid(number), "quote": quote}


def _p(number: int, quote: str, **fields) -> dict:
    task = fields.get("academic_task") or "une situation personnelle"
    fields.setdefault("summary", f"L'étudiante décrit, à propos de {task}, ce qu'elle demande ou ne demande pas à l'outil.")
    return base_practice(turn_start=tid(number - 1), turn_end=tid(number), evidence=[ev(number, quote)], **fields)


def _s(number: int, signal_type: str, quote: str, **fields) -> dict:
    fields.setdefault("description", f"L'enquêtée emploie une formulation relevée comme {signal_type}.")
    return base_signal(turn_ids=[tid(number)], signal_type=signal_type, surface_form=quote[:60],
                       evidence=[ev(number, quote)], **fields)


def _practices() -> dict[int, list[dict]]:
    items: dict[int, list[dict]] = {}
    for n, (task, ask, domain) in ORDINARY.items():
        items[n] = [_p(n, f"Pour {task}, je lui demande {ask}.", academic_task=task, practice_domain=domain,
                       assessment_context="personal" if domain == "personal" else "unknown")]
    for n, task in R1.items():
        items[n] = [_p(n, "je lui fais juste reformuler des phrases", academic_task=task),
                    _p(n, "jamais écrire", use_status="refusal", non_use_reason="personal_rule", academic_task=task)]
    for n, task in R2.items():
        items[n] = [_p(n, f"Pour {task}, je l'utilise uniquement quand je manque de temps.", academic_task=task)]
    items[30] = [_p(30, "Je vérifie toujours tout ce qu'il me donne", academic_task="vérification des réponses")]
    items[36] = [_p(36, "je lui demande des listes, mais je les apprends moi-même", academic_task="vocabulaire d'anglais")]
    items[46] = [_p(46, "Quand il m'explique un théorème", academic_task="théorèmes")]
    items[60] = [_p(60, "Je préfère Gemini à ChatGPT pour les mails", academic_task="mails", ai_tool=["Gemini"])]
    items[66] = [_p(66, "Je préfère DeepL à ChatGPT pour traduire", academic_task="traduction", ai_tool=["DeepL"])]
    items[76] = [_p(76, "je lui fais juste corriger les fautes", academic_task="mails à l'administration")]
    items[86] = [_p(86, "et moi aussi pour m'entraîner", academic_task="QCM")]
    items[90] = [_p(90, KEY_TURNS[90], academic_task="exercices de maths")]
    items[100] = [_p(100, "je ne l'utilise pas du tout", use_status="refusal", non_use_reason="personal_rule",
                     academic_task="dossiers notés")]
    items[110] = [_p(110, KEY_TURNS[110], use_status="past_use", academic_task="commentaires de texte")]
    items[112] = [_p(112, "je ne lui fais plus rien rédiger", use_status="non_use", non_use_reason="not_stated",
                     academic_task="commentaires de texte"),
                  _p(112, "je lui demande seulement de relire mes commentaires de texte",
                     academic_task="commentaires de texte")]
    items[120] = [_p(120, "Je ne fais jamais faire mes plans", use_status="refusal", non_use_reason="personal_rule",
                     academic_task="plans", stated_frequency="jamais")]
    items[126] = [_p(126, KEY_TURNS[126], use_status="past_use", academic_task="plan", stated_frequency="une fois",
                     stated_reason=["faute de temps"])]
    items[150] = [_p(150, "je lui fais seulement corriger l'orthographe de mes mémos", academic_task="mémos",
                     explicitness="unclear", uncertainty_note="Tour marqué enquêteur : attribution douteuse.")]
    items[160] = [_p(160, "je lui fais seulement vérifier mes références bibliographiques",
                     academic_task="références bibliographiques")]
    items[170] = [_p(170, "je ne les vérifie pas", use_status="non_use", non_use_reason="not_stated",
                     academic_task="vérification des références")]
    items[176] = [_p(176, "Je lui demande des quiz.", academic_task="quiz")]
    items[180] = [_p(180, KEY_TURNS[180], use_status="refusal", non_use_reason="personal_rule", academic_task="mémoire")]
    return items


def _signals() -> dict[int, list[dict]]:
    items: dict[int, list[dict]] = {}
    for n in R1:
        items[n] = [_s(n, "restriction", "juste reformuler des phrases, jamais écrire")]
    for n in R2:
        items[n] = [_s(n, "restriction", "uniquement quand je manque de temps")]
    items[30] = [_s(30, "normative_formulation", "Je vérifie toujours tout ce qu'il me donne")]
    items[46] = [_s(46, "normative_formulation", "je refais toujours la démonstration seule pour comprendre")]
    items[60] = [_s(60, "preference_statement", "Je préfère Gemini à ChatGPT")]
    items[66] = [_s(66, "preference_statement", "Je préfère DeepL à ChatGPT")]
    items[76] = [_s(76, "restriction", "juste corriger les fautes")]
    items[86] = [_s(86, "reference_to_peer_judgment", "Mes camarades l'utilisent tous pour les QCM")]
    items[100] = [_s(100, "preference_statement", "je préfère tout faire moi-même")]
    items[112] = [_s(112, "contrast", "Maintenant, je ne lui fais plus rien rédiger")]
    items[120] = [_s(120, "normative_formulation", KEY_TURNS[120])]
    items[126] = [_s(126, "exception", "Une fois")]
    items[150] = [_s(150, "self_correction", "enfin ça dépend des fois", needs_human_review=True)]
    items[160] = [_s(160, "reference_to_teacher_judgment", "Ma prof dirait que ce n'est pas mon travail")]
    items[170] = [_s(170, "contrast", "je ne les vérifie pas")]
    items[176] = [_s(176, "metadiscursive_self_evaluation", "c'est un peu facile ce que je dis")]
    items[180] = [_s(180, "normative_formulation", "Je ne veux pas qu'il réfléchisse à ma place")]
    return items


PRACTICES = _practices()
SIGNALS = _signals()
PRACTICE_COUNT = sum(len(v) for v in PRACTICES.values())
CONTRADICTION = base_signal(
    turn_ids=[tid(30), tid(170)], signal_type="cross_turn_contradiction",
    surface_form="toujours tout / je ne les vérifie pas",
    description="Au tour T0030, l'enquêtée dit vérifier toujours tout ce que l'outil lui donne ; au tour T0170, elle dit "
                "ne pas vérifier les références qu'il lui donne.", topic="vérification", cross_turn_reference="Vérifier.",
    evidence=[ev(30, "Je vérifie toujours tout ce qu'il me donne"), ev(170, "je ne les vérifie pas")])
AUDIT = {"assessments": [base_assessment(
    turn_id=tid(MISATTRIBUTED_TURN), suggested_speaker="enquete", confidence="high",
    reason="Réponse à la première personne qui suit directement la question du tour précédent.",
    evidence=[ev(MISATTRIBUTED_TURN, "Moi personnellement je lui fais seulement corriger l'orthographe de mes mémos"),
              ev(MISATTRIBUTED_TURN - 1, SPECIAL_QUESTIONS[MISATTRIBUTED_TURN - 1])])], "audit_notes": None}


def practice_reader(params: dict):
    present = {t["turn_id"] for t in sent_turns(params)}
    items = [p for _, ps in sorted(PRACTICES.items()) for p in ps if _covered(p, present)]
    return text_response({"practices": items, "extraction_notes": None})


def signal_reader(params: dict):
    present = {t["turn_id"] for t in sent_turns(params)}
    items = [s for _, ss in sorted(SIGNALS.items()) for s in ss if _covered(s, present)]
    return text_response({"signals": items, "reading_notes": None})


def long_distance_reader(params: dict):
    present = {t["turn_id"] for t in sent_turns(params)}
    items = [CONTRADICTION] if _covered(CONTRADICTION, present) else []
    return text_response({"signals": items, "reading_notes": None})


def stage3_responders() -> dict:
    return {PRACTICE: practice_reader, INTERACTION: signal_reader, LONG_DISTANCE: long_distance_reader,
            AUDITOR: lambda p: text_response(AUDIT)}


# --- Étape 4 simulée ------------------------------------------------------------------------------------

decision, move = S4.decision, S4.move
ORDINARY_SUMMARY = S5.ORDINARY_SUMMARY


def _rules() -> dict[int, dict]:
    rules = {}
    for n, task in R1.items():
        rules[n] = decision(
            "accountability_episode", f"Pour {task}, l'étudiante limite l'outil à la reformulation et exclut l'écriture.",
            moves=[move("restriction", "L'étudiante limite l'usage à la reformulation.", n),
                   move("refusal", "L'étudiante exclut que l'outil écrive.", n)],
            boundary=["reformuler / écrire"], problem="Jusqu'où l'outil peut-il intervenir dans l'écriture ?",
            evidence=[(n, turn_text(n))])
    for n, task in R2.items():
        rules[n] = decision(
            "accountability_episode", f"Pour {task}, l'étudiante limite l'usage aux moments où elle manque de temps.",
            moves=[move("restriction", "L'étudiante limite l'usage au manque de temps.", n)],
            problem="Quand l'usage est-il admis ?", evidence=[(n, turn_text(n))],
            confidence="medium" if n == 50 else "high", needs_review=n == 50)
    rules[30] = decision(
        "accountability_episode",
        "L'étudiante dit vérifier toujours tout ce que l'outil lui donne ; plus loin, elle dit ne pas vérifier les "
        "références qu'il lui donne.",
        moves=[move("general_rule", "L'étudiante énonce ce qu'elle fait de ce que l'outil lui donne.", 30),
               move("other_explicit_move", "L'étudiante dit ne pas vérifier les références.", 170)],
        problem="Que faut-il vérifier de ce que donne l'outil ?",
        evidence=[(30, "Je vérifie toujours tout ce qu'il me donne"), (170, "je ne les vérifie pas")])
    rules[170] = decision(  # si la contradiction n'était pas relevée : épisode séparé
        "accountability_episode", "L'étudiante dit ne pas vérifier les références que l'outil lui donne.",
        moves=[move("other_explicit_move", "L'étudiante dit ne pas vérifier les références.", 170)],
        problem="Que faut-il vérifier de ce que donne l'outil ?", evidence=[(170, "je ne les vérifie pas")])
    rules[36] = decision(
        "accountability_episode", "L'étudiante demande des listes de vocabulaire et dit les apprendre elle-même, "
                                  "« sinon je ne retiens rien ».",
        moves=[move("distinction", "L'étudiante sépare obtenir les listes et les apprendre.", 36),
               move("appeal_to_learning", "L'étudiante rapporte l'usage à ce qu'elle retient.", 36)],
        boundary=["obtenir / apprendre"], problem="Qui apprend le vocabulaire ?", evidence=[(36, KEY_TURNS[36])])
    rules[46] = decision(
        "accountability_episode", "L'étudiante dit refaire toujours la démonstration seule « pour comprendre ».",
        moves=[move("appeal_to_learning", "L'étudiante rapporte l'usage à la compréhension.", 46),
               move("appeal_to_effort", "L'étudiante dit refaire la démonstration seule.", 46)],
        problem="Qui fait la démonstration ?", evidence=[(46, KEY_TURNS[46])])
    for n in (60, 66):
        rules[n] = decision("ordinary_practice", "L'étudiante énonce une préférence entre deux outils. " + ORDINARY_SUMMARY,
                            confidence="medium", evidence=[(n, KEY_TURNS[n])])
    rules[76] = decision(
        "accountability_episode", "Pour les mails à l'administration, l'étudiante limite l'usage à la correction des fautes.",
        moves=[move("restriction", "L'étudiante limite l'usage à la correction des fautes.", 76)],
        problem="Jusqu'où l'outil intervient-il dans un mail ?", evidence=[(76, KEY_TURNS[76])])
    rules[86] = decision(
        "accountability_episode", "L'étudiante rapporte que ses camarades l'utilisent tous pour les QCM, et elle aussi.",
        moves=[move("normalization", "L'étudiante présente l'usage comme courant chez ses camarades.", 86),
               move("comparison", "L'étudiante se compare à ses camarades.", 86)],
        external="les camarades", problem="L'usage pour les QCM est-il courant ?", evidence=[(86, KEY_TURNS[86])])
    rules[100] = decision(
        "accountability_episode", "Pour les dossiers notés, l'étudiante dit ne pas utiliser l'outil et préférer tout faire "
                                  "elle-même.",
        moves=[move("refusal", "L'étudiante dit ne pas utiliser l'outil pour les dossiers notés.", 100),
               move("preference", "L'étudiante dit préférer tout faire elle-même.", 100)],
        boundary=["faire soi-même / faire faire"], problem="Qui fait un dossier noté ?", evidence=[(100, KEY_TURNS[100])])
    rules[110] = decision(
        "accountability_episode", "L'étudiante situe au lycée la rédaction de ses commentaires par l'outil et dit ne plus "
                                  "rien lui faire rédiger maintenant, seulement relire.",
        moves=[move("comparison", "L'étudiante met en regard le lycée et maintenant.", 110, 112),
               move("restriction", "L'étudiante limite l'usage actuel à la relecture.", 112)],
        boundary=["rédiger / relire"], problem="Qui rédige le commentaire de texte ?",
        evidence=[(110, KEY_TURNS[110]), (112, KEY_TURNS[112])])
    rules[120] = decision(
        "accountability_episode", "L'étudiante dit ne jamais faire faire ses plans (« c'est mon travail »), puis rapporte une "
                                  "demande de plan présentée comme unique, « faute de temps ».",
        moves=[move("general_rule", "L'étudiante énonce ce qu'elle fait pour ses plans.", 120),
               move("appeal_to_authorship", "L'étudiante rapporte le plan à son travail.", 120),
               move("exception", "L'étudiante présente la demande de plan comme une seule fois.", 126)],
        boundary=["faire / faire faire"], problem="Qui fait le plan ?", role="« c'est mon travail »",
        evidence=[(120, KEY_TURNS[120]), (126, KEY_TURNS[126])])
    rules[150] = decision(
        "uncertain", "Le tour T0150 est attribué à l'enquêteur dans la transcription ; l'audit y signale une réponse "
                     "probable de l'enquêtée. L'épisode ne peut pas être établi.", confidence="low", needs_review=True,
        evidence=[(150, "je lui fais seulement corriger l'orthographe de mes mémos")])
    rules[160] = decision(
        "accountability_episode", "L'étudiante évoque ce que dirait sa professeure et limite l'usage à la vérification de "
                                  "ses références bibliographiques.",
        moves=[move("appeal_to_external_judgment", "L'étudiante évoque le jugement de sa professeure.", 160),
               move("restriction", "L'étudiante limite l'usage à la vérification des références.", 160)],
        external="la professeure", problem="Jusqu'où l'outil intervient-il dans les références ?",
        evidence=[(160, KEY_TURNS[160])])
    rules[176] = decision(
        "accountability_episode", "L'étudiante dit demander des quiz, puis qualifie sa formulation de « un peu facile ».",
        moves=[move("self_evaluation", "L'étudiante évalue ce qu'elle vient de dire.", 176)],
        problem="Comment dire la demande de quiz ?", confidence="medium", evidence=[(176, KEY_TURNS[176])])
    rules[180] = decision(
        "accountability_episode", "L'étudiante dit ne pas vouloir que l'outil réfléchisse à sa place pour le mémoire.",
        moves=[move("refusal", "L'étudiante exclut que l'outil réfléchisse à sa place.", 180),
               move("appeal_to_authorship", "L'étudiante rapporte le mémoire à sa propre réflexion.", 180)],
        boundary=["réfléchir / faire réfléchir"], problem="Qui réfléchit au mémoire ?", evidence=[(180, KEY_TURNS[180])])
    return rules


RULES = _rules()
BUILDER = S5.component_builder(RULES)

# --- Étape 5 simulée ------------------------------------------------------------------------------------

claim, criterion, plan = S5.claim, S5.criterion, S5.plan
INFO_TURNS = (2, 4, 16, 32, 34, 56, 84, 94, 116, 136)
PERSONAL_TURNS = (6, 12, 14, 24, 26, 44, 54, 64, 74, 104, 106, 146)

GOOD_CLAIMS = [
    claim("stable_boundary", "Pour trois écrits (dissertations, notes de synthèse, rapports de stage), l'étudiante formule "
                             "la même limite : « juste reformuler des phrases, jamais écrire ».",
          items=list(R1), evidence=list(R1), contexts=list(R1.values()), confidence="high"),
    claim("stable_boundary", "Pour les fiches, les exposés et les TD, l'étudiante borne l'usage de la même manière : "
                             "« uniquement quand je manque de temps ».",
          items=list(R2), evidence=list(R2), contexts=list(R2.values())),
    claim("recurring_accounting_move", "La restriction est l'opération la plus reprise : six épisodes bornent l'usage.",
          items=[*R1, *R2], evidence=[*R1, *R2], moves=["restriction"], confidence="high"),
    claim("contextual_variation", "En maths, l'étudiante dit demander directement la solution des exercices ; pour les "
                                  "dissertations, elle limite l'outil à la reformulation.",
          items=[("P", 90), 10], evidence=[90, 10], contexts=["exercices de maths", "dissertations"]),
    claim("contextual_variation", "Pour les dossiers notés, l'étudiante dit ne pas utiliser l'outil ; pour ses week-ends "
                                  "et sa musique, elle raconte des demandes sans restriction.",
          items=[100, ("P", 104), ("P", 106)], evidence=[100, 104, 106], contexts=["dossiers notés", "vie personnelle"]),
    claim("explicit_temporal_change", "L'étudiante situe au lycée la rédaction de ses commentaires de texte par l'outil "
                                      "(« Au lycée ») et dit ne plus rien lui faire rédiger « Maintenant », seulement "
                                      "relire.",
          items=[110], evidence=[110, 112], anchors=[(110, "Au lycée"), (112, "Maintenant")],
          contexts=["commentaires de texte"], confidence="high"),
    claim("exception", "L'étudiante énonce une règle (« Je ne fais jamais faire mes plans ») et rapporte un cas présenté "
                       "comme unique (« Une fois, faute de temps »).", items=[120], evidence=[120, 126], contexts=["plans"]),
    claim("unresolved_tension", "Deux formulations coexistent : « Je vérifie toujours tout ce qu'il me donne » et, à propos "
                                "des références, « je ne les vérifie pas » ; l'entretien ne les articule pas.",
          items=[30], evidence=[30, 170], contexts=["vérification"]),
    claim("ordinary_zone", "Des recherches d'information pratiques (horaires, définitions, salles, dates, inscriptions…) "
                           "sont racontées sans restriction, justification ni évaluation explicite.",
          items=[("P", n) for n in INFO_TURNS], evidence=list(INFO_TURNS), contexts=["recherche d'information"],
          confidence="high"),
    claim("ordinary_zone", "Les demandes de la vie personnelle (sorties, trajets, jeux, cuisine, sport…) sont racontées "
                           "sans restriction, justification ni évaluation explicite.",
          items=[("P", n) for n in PERSONAL_TURNS], evidence=list(PERSONAL_TURNS), contexts=["vie personnelle"]),
]
GOOD_CRITERIA = [
    criterion("être l'auteure de son plan", "Dans cet entretien, l'étudiante associe le fait de faire ses plans à « mon "
                                            "travail ».", items=[120], evidence=[120]),
    criterion("temps disponible", "Dans cet entretien, l'étudiante associe l'usage au manque de temps (« uniquement quand je "
                                  "manque de temps », « faute de temps »).", items=[20, 50, 80, 120], evidence=[20, 80, 126]),
    criterion("apprendre et comprendre par soi-même", "Dans cet entretien, l'étudiante associe certains usages au fait "
              "d'apprendre ou de comprendre elle-même (« je les apprends moi-même », « pour comprendre »).",
              items=[36, 46], evidence=[36, 46]),
    criterion("jugement de l'enseignante", "Dans cet entretien, l'étudiante associe la limite de l'usage à ce que dirait sa "
                                           "professeure (« Ma prof dirait que ce n'est pas mon travail »).",
              items=[160], evidence=[160]),
    criterion("faire soi-même les travaux notés", "Dans cet entretien, l'étudiante associe les dossiers notés au fait de "
                                                  "« tout faire moi-même ».", items=[100], evidence=[100]),
]
GOOD_SUMMARY = ("Deux limites reviennent dans plusieurs tâches (« juste reformuler, jamais écrire » ; « uniquement quand je "
                "manque de temps »), et les usages varient selon les tâches et les contextes. Un seul changement est "
                "explicitement daté par l'étudiante : la rédaction des commentaires de texte « au lycée », la seule "
                "relecture « maintenant ». Une règle sur les plans coexiste avec un cas présenté comme unique, et deux "
                "formulations sur la vérification restent juxtaposées. Les recherches d'information et la vie personnelle "
                "forment des zones racontées sans justification.")
GOOD_PLAN = plan("mixed", claims=GOOD_CLAIMS, criteria=GOOD_CRITERIA, summary=GOOD_SUMMARY)
ADVERSARIAL_PLAN = plan(
    "mixed",
    claims=[*GOOD_CLAIMS,
            claim("explicit_temporal_change", "Au fil de l'entretien, l'étudiante devient plus prudente : d'abord elle fait "
                                              "reformuler ses dissertations, puis elle refuse qu'il réfléchisse pour le "
                                              "mémoire.", items=[10, 180], evidence=[10, 180], confidence="high")],
    criteria=[*GOOD_CRITERIA,
              criterion("culpabilité", "Dans cet entretien, l'étudiante associe la vérification à sa culpabilité.",
                        items=[30], evidence=[170])],
    summary=GOOD_SUMMARY)


def mapper(mode: str = "good"):
    return S5.scripted_mapper(GOOD_PLAN if mode == "good" else ADVERSARIAL_PLAN)
