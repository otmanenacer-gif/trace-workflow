"""Entretiens SYNTHÉTIQUES de l'étape 5 et réponses simulées (aucun vrai entretien, aucun appel réel).

Chaque cas est un petit entretien complet : l'étape 3 (Practice Extractor, Interaction Reader, audit des
locuteurs) et l'étape 4 (Accountability Episode Builder) sont simulées pour produire de VRAIES sorties
d'étape 4 (pipeline réel, validateurs réels) ; l'étape 5 est simulée par `scripted_mapper`, qui résout
des références « par tour » (numéro de tour → épisode ou pratique sans marqueur) dans la représentation
réellement envoyée au Trajectory Mapper.

Cas (section 18 de la consigne de l'étape 5) :
A. TEMPORAL   — « Au lycée je lui faisais résoudre… » / « Maintenant je m'en sers seulement… » ;
B. CONTEXT    — maths / dissertation, sans temporalité (le lecteur adverse invente une évolution) ;
C. EXCEPTION  — « Je ne fais jamais faire mes plans » / « Une fois, faute de temps… » ;
D. STABLE     — « uniquement quand je manque de temps » dans trois tâches ;
E. ORDINARY   — recherches d'information racontées sans justification ;
F. NOPATTERN  — plusieurs usages sans lien ni temporalité (étape 4 : aucun candidat) ;
G. REVIEW     — un épisode à revoir (tour mal attribué) ;
H. INTERVIEWER — l'enquêteur propose « honte » / « peur », l'enquêté·e ne reprend pas.
REFERENCE_PLAN : plan de l'étape 5 pour l'entretien de référence de l'étape 4 (tests/synthetic_stage4.py).
"""

from __future__ import annotations

import json
import re

from tests import synthetic_stage4 as S4
from tests.fake_llm import AUDITOR, INTERACTION, PRACTICE, text_response
from tests.synthetic_interviews import assessment, practice, signal

ORDINARY_SUMMARY = "Aucune restriction, justification ou évaluation explicite n'est relevée dans ce passage."


# --- Étape 4 simulée : un épisode par composante de candidats ---------------------------------------

def component_builder(rules: dict[int, dict]):
    """Lecteur DÉTERMINISTE des candidats envoyés : un épisode par composante, décision par tour d'ancrage
    (plus petit tour cité par les pratiques de la composante)."""
    def respond(params: dict):
        payload = S4.sent_payload(params)
        prefix = payload["id_prefix"]
        groups: dict[str, list[dict]] = {}
        for cand in payload["candidates"]:
            groups.setdefault(cand["component_id"], []).append(cand)
        episodes = []
        for group in groups.values():
            practice_ids = list(dict.fromkeys(p for c in group for p in c["practice_ids"]))
            signal_ids = list(dict.fromkeys(s for c in group for s in c["signal_ids"]))
            quotes = [q for pid in practice_ids for q in S4.practice_quotes(payload, pid)]
            anchor = min(S4.turn_number(q["turn_id"]) for q in quotes)
            rule = rules.get(anchor, S4.UNKNOWN)
            numbers = sorted(S4.turn_number(t) for c in group for t in c["turn_ids"])
            turn = lambda n: f"{prefix}T{n:04d}"  # noqa: E731
            voiced = [q for q in quotes if payload["turns_by_id"][q["turn_id"]].get("speaker") != "enqueteur"
                      or payload["turns_by_id"][q["turn_id"]].get("speaker_warning")]
            evidence = ([{"turn_id": turn(n), "quote": q} for n, q in rule["evidence"]] if rule.get("evidence")
                        else list({(q["turn_id"], q["quote"]): {"turn_id": prefix + q["turn_id"], "quote": q["quote"]}
                                   for q in (voiced or quotes)}.values()))
            episodes.append({
                "candidate_ids": [c["candidate_id"] for c in group], "turn_start": turn(numbers[0]),
                "turn_end": turn(numbers[-1]), "practice_ids": practice_ids, "signal_ids": signal_ids,
                "episode_status": rule["status"], "accountability_problem": rule["problem"],
                "accounting_moves": [{"type": t, "description": d, "evidence_turn_ids": [turn(n) for n in ns]}
                                     for t, d, ns in rule["moves"]],
                "boundary_objects": rule["boundary"], "student_role_reference": rule["role"],
                "external_reference": rule["external"], "episode_summary": rule["summary"],
                "confidence": rule["confidence"], "needs_review": rule["needs_review"], "evidence": evidence})
        output = {"episodes": episodes, "builder_notes": None}
        return text_response(output)
    return respond


# --- Étape 5 simulée : plan « par tour » résolu dans la représentation envoyée -----------------------

_MATERIAL_RE = re.compile(r"<material>\n(.*)\n</material>", re.DOTALL)
_NUMBER = re.compile(r"T(\d+)$")


def sent_material(params: dict) -> dict:
    """Représentation réellement transmise au Trajectory Mapper."""
    return json.loads(_MATERIAL_RE.search(params["messages"][0]["content"]).group(1))


def claim(claim_type: str, description: str, items=(), evidence=(), anchors=(), contexts=(), moves=(),
          confidence="medium", needs_review=False) -> dict:
    return {"claim_type": claim_type, "description": description, "items": list(items), "evidence": list(evidence),
            "anchors": list(anchors), "contexts": list(contexts), "moves": list(moves), "confidence": confidence,
            "needs_review": needs_review}


def criterion(text: str, description: str, items=(), evidence=(), confidence="medium", needs_review=False) -> dict:
    return {"criterion": text, "description": description, "items": list(items), "evidence": list(evidence),
            "confidence": confidence, "needs_review": needs_review}


def plan(configuration: str, claims=(), criteria=(), summary="", confidence="medium", needs_review=False) -> dict:
    return {"configuration_type": configuration, "claims": list(claims), "criteria": list(criteria),
            "summary": summary, "confidence": confidence, "needs_review": needs_review}


def _turns_of(entry: dict, material: dict) -> set[int]:
    return {int(_NUMBER.search(material["quotes_by_id"][q]["turn"]).group(1)) for q in entry.get("quotes", [])}


def resolve(ref, material: dict) -> tuple[str, str]:
    """Référence « par tour » → ("episode", E…) ou ("practice", P…). ("P", n) force une pratique sans marqueur ;
    ("E", id) passe un identifiant tel quel (épisode inexistant ou rejeté : lecteur adverse)."""
    if isinstance(ref, tuple) and ref[0] == "E":
        return "episode", ref[1]
    want_practice = isinstance(ref, tuple) and ref[0] == "P"
    number = ref[1] if isinstance(ref, tuple) else ref
    if not want_practice:
        for e in material["episodes"]:
            if number in _turns_of(e, material):
                return "episode", e["id"]
    for p in material["unmarked_practices"]:
        if number in _turns_of(p, material):
            return "practice", p["id"]
    raise AssertionError(f"Aucun élément au tour {number} dans la représentation envoyée")


def mapper_output(params: dict, scripted: dict) -> dict:
    material = sent_material(params)

    def refs(items):
        resolved = [resolve(r, material) for r in items]
        return ([i for k, i in resolved if k == "episode"], [i for k, i in resolved if k == "practice"])

    claims = []
    for c in scripted["claims"]:
        episode_ids, practice_ids = refs(c["items"])
        claims.append({"claim_type": c["claim_type"], "description": c["description"], "episode_ids": episode_ids,
                       "practice_ids": practice_ids, "contexts": c["contexts"], "accounting_move_types": c["moves"],
                       "temporal_anchors": [{"text": t, "turn_id": f"T{n:04d}"} for n, t in c["anchors"]],
                       "evidence_turn_ids": [f"T{n:04d}" for n in c["evidence"]], "confidence": c["confidence"],
                       "needs_review": c["needs_review"]})
    criteria = []
    for c in scripted["criteria"]:
        episode_ids, _ = refs(c["items"])
        criteria.append({"criterion": c["criterion"], "description": c["description"], "episode_ids": episode_ids,
                         "evidence_turn_ids": [f"T{n:04d}" for n in c["evidence"]], "confidence": c["confidence"],
                         "needs_review": c["needs_review"]})
    return {"configuration_type": scripted["configuration_type"], "claims": claims, "student_role_criteria": criteria,
            "trajectory_summary": scripted["summary"], "confidence": scripted["confidence"],
            "needs_review": scripted["needs_review"], "mapper_notes": None}


def scripted_mapper(scripted: dict):
    def respond(params: dict):
        output = mapper_output(params, scripted)
        return text_response(output)
    return respond


# --- Cas synthétiques ---------------------------------------------------------------------------------

class Case:
    """Un entretien synthétique et ses réponses simulées aux étapes 3, 4 et 5."""

    def __init__(self, interview_id: str, turns: list[tuple[str, str]]):
        self.interview_id = interview_id
        self.filename = f"{interview_id.capitalize()}.txt"
        self.turns = turns
        self.practices: list[dict] = []
        self.signals: list[dict] = []
        self.assessments: list[dict] = []
        self.rules: dict[int, dict] = {}
        self.good: dict | None = None
        self.adversarial: dict | None = None

    @property
    def text(self) -> str:
        return "\n".join(f"{speaker} : {sentence}" for speaker, sentence in self.turns) + "\n"

    @property
    def files(self) -> list[tuple[str, bytes]]:
        return [(self.filename, self.text.encode("utf-8"))]

    def tid(self, number: int) -> str:
        return f"{self.interview_id}_T{number:04d}"

    def ev(self, number: int, quote: str) -> dict:
        assert quote in self.turns[number - 1][1], (self.interview_id, number, quote)
        return {"turn_id": self.tid(number), "quote": quote}

    def p(self, number: int, quote: str, **fields) -> None:
        fields.setdefault("summary", f"L'étudiant décrit, à propos de {fields.get('academic_task') or 'cet usage'}, "
                                     "ce qu'il demande ou ne demande pas à l'outil.")
        self.practices.append(practice(turn_start=self.tid(number - 1), turn_end=self.tid(number),
                                       evidence=[self.ev(number, quote)], **fields))

    def s(self, number: int, signal_type: str, quote: str, **fields) -> None:
        fields.setdefault("description", f"L'enquêté emploie une formulation relevée comme {signal_type}.")
        self.signals.append(signal(turn_ids=[self.tid(number)], signal_type=signal_type, surface_form=quote[:60],
                                   evidence=[self.ev(number, quote)], **fields))

    def contradiction(self, first: tuple[int, str], second: tuple[int, str], topic: str) -> None:
        self.signals.append(signal(
            turn_ids=[self.tid(first[0]), self.tid(second[0])], signal_type="cross_turn_contradiction",
            surface_form=f"{first[1][:25]} / {second[1][:25]}", topic=topic, cross_turn_reference=topic,
            description=f"Au tour T{first[0]:04d} et au tour T{second[0]:04d}, l'enquêté formule deux choses "
                        f"différentes à propos de : {topic}.",
            evidence=[self.ev(*first), self.ev(*second)]))

    def stage3_responders(self) -> dict:
        practices = {"practices": self.practices, "extraction_notes": None}
        signals = {"signals": self.signals, "reading_notes": None}
        audit = {"assessments": self.assessments, "audit_notes": None}
        return {PRACTICE: lambda p: text_response(practices),
                INTERACTION: lambda p: text_response(signals),
                AUDITOR: lambda p: text_response(audit)}

    def builder(self):
        return component_builder(self.rules)

    def mapper(self, mode: str = "good"):
        return scripted_mapper(self.good if mode == "good" else self.adversarial)


def _q(number: int, text: str) -> tuple[str, str]:
    return ("Enquêteur" if number % 2 else "Enquêté", text)


def _turns(*texts: str) -> list[tuple[str, str]]:
    return [_q(n, t) for n, t in enumerate(texts, start=1)]


decision, move = S4.decision, S4.move

# A. Changement temporel réel -------------------------------------------------------------------------
TEMPORAL = Case("ETAPE5_TEMPORALITE", _turns(
    "Comment tu utilises ChatGPT pour tes études ?",
    "Au lycée je lui faisais résoudre les exercices de maths, je recopiais les réponses.",
    "Et aujourd'hui, à Sciences Po ?",
    "Maintenant je m'en sers seulement pour chercher des sources, les exercices je les fais moi-même.",
    "Et pour les langues ?",
    "Je lui demande des traductions de mots.",
    "Merci.",
))
TEMPORAL.p(2, "Au lycée je lui faisais résoudre les exercices de maths", use_status="past_use",
           academic_task="exercices de maths")
TEMPORAL.p(4, "les exercices je les fais moi-même", use_status="non_use", non_use_reason="not_stated",
           academic_task="exercices de maths")
TEMPORAL.p(6, "Je lui demande des traductions de mots.", academic_task="traduction")
TEMPORAL.s(4, "restriction", "je m'en sers seulement pour chercher des sources")
TEMPORAL.rules = {2: decision(
    "accountability_episode",
    "L'étudiant situe au lycée la résolution des exercices de maths par l'outil et dit faire maintenant ces exercices "
    "lui-même, l'outil servant seulement à chercher des sources.",
    moves=[move("comparison", "L'étudiant met en regard deux usages qu'il situe au lycée et maintenant.", 2, 4),
           move("restriction", "L'étudiant limite l'usage actuel à la recherche de sources.", 4),
           move("general_rule", "L'étudiant dit faire les exercices lui-même.", 4)],
    boundary=["faire / faire faire"], problem="Qui fait les exercices de maths ?",
    evidence=[(2, "Au lycée je lui faisais résoudre les exercices de maths"),
              (4, "Maintenant je m'en sers seulement pour chercher des sources"),
              (4, "les exercices je les fais moi-même")])}
TEMPORAL.good = plan(
    "temporal_trajectory",
    claims=[claim("explicit_temporal_change",
                  "L'étudiant situe au lycée la résolution des exercices de maths par l'outil (« Au lycée je lui faisais "
                  "résoudre les exercices de maths ») et dit faire maintenant ces exercices lui-même, l'outil servant "
                  "« seulement pour chercher des sources ».",
                  items=[2], evidence=[2, 4], anchors=[(2, "Au lycée"), (4, "Maintenant")],
                  contexts=["exercices de maths", "recherche de sources"], confidence="high")],
    criteria=[criterion("faire soi-même les exercices",
                        "Dans cet entretien, l'étudiant associe son usage actuel au fait de faire les exercices "
                        "lui-même (« les exercices je les fais moi-même »).", items=[2], evidence=[4])],
    summary="L'étudiant date explicitement deux états : au lycée, les exercices de maths résolus par l'outil ; "
            "maintenant, un usage limité à la recherche de sources et des exercices faits lui-même. Une demande de "
            "traductions de mots est racontée sans restriction ni évaluation.",
    confidence="high")

# B. Variation contextuelle sans temporalité ---------------------------------------------------------
CONTEXT = Case("ETAPE5_CONTEXTE", _turns(
    "Pour quoi tu utilises l'IA ?",
    "En maths je lui demande directement la réponse aux exercices.",
    "Et pour les dissertations ?",
    "Pour une dissertation je préfère écrire moi-même, je lui demande juste de relire.",
    "Et en anglais ?",
    "Je lui fais corriger mes phrases.",
))
CONTEXT.p(2, "En maths je lui demande directement la réponse aux exercices.", academic_task="exercices de maths")
CONTEXT.p(4, "Pour une dissertation je préfère écrire moi-même", use_status="non_use", non_use_reason="personal_rule",
          academic_task="dissertation")
CONTEXT.p(4, "je lui demande juste de relire", academic_task="dissertation")
CONTEXT.p(6, "Je lui fais corriger mes phrases.", academic_task="anglais")
CONTEXT.s(4, "preference_statement", "je préfère écrire moi-même")
CONTEXT.rules = {4: decision(
    "accountability_episode",
    "Pour une dissertation, l'étudiant dit préférer écrire lui-même et limite l'outil à la relecture.",
    moves=[move("preference", "L'étudiant énonce une préférence pour écrire lui-même.", 4),
           move("restriction", "L'étudiant limite l'usage à la relecture.", 4)],
    boundary=["écrire / relire"], problem="Qui écrit la dissertation ?",
    evidence=[(4, "Pour une dissertation je préfère écrire moi-même, je lui demande juste de relire.")])}
CONTEXT.good = plan(
    "contextual_configuration",
    claims=[claim("contextual_variation",
                  "En maths, l'étudiant dit demander directement la réponse aux exercices ; pour une dissertation, il "
                  "dit préférer écrire lui-même et limiter l'outil à la relecture.",
                  items=[("P", 2), 4], evidence=[2, 4], contexts=["exercices de maths", "dissertation"])],
    criteria=[criterion("écrire soi-même la dissertation",
                        "Dans cet entretien, l'étudiant associe la dissertation au fait d'écrire lui-même "
                        "(« je préfère écrire moi-même »).", items=[4], evidence=[4])],
    summary="Les usages varient selon la tâche : réponse demandée en maths, écriture gardée pour la dissertation, "
            "correction de phrases en anglais. Aucun ancrage temporel n'est formulé.")
CONTEXT.adversarial = plan(
    "temporal_trajectory",
    claims=[claim("explicit_temporal_change",
                  "Au fil de l'entretien, l'étudiant évolue : d'abord il demande la réponse en maths, puis il préfère "
                  "écrire lui-même.", items=[("P", 2), 4], evidence=[2, 4], confidence="high"),
            claim("exception", "Le travail en anglais est une exception.", items=[("P", 6)], evidence=[6])],
    summary="L'étudiant devient progressivement plus autonome.", confidence="high")

# C. Règle + exception ----------------------------------------------------------------------------------
EXCEPTION = Case("ETAPE5_EXCEPTION", _turns(
    "Tu lui fais faire tes plans ?",
    "Je ne fais jamais faire mes plans, c'est mon travail.",
    "Et pour les fiches de lecture ?",
    "Pour les fiches de lecture je lui demande des résumés.",
    "Ça ne t'est jamais arrivé pour un plan ?",
    "Une fois, faute de temps, je lui ai demandé un plan détaillé.",
))
EXCEPTION.p(2, "Je ne fais jamais faire mes plans", use_status="refusal", non_use_reason="personal_rule",
            academic_task="plans", stated_frequency="jamais")
EXCEPTION.p(4, "je lui demande des résumés", academic_task="fiches de lecture")
EXCEPTION.p(6, "Une fois, faute de temps, je lui ai demandé un plan détaillé.", use_status="past_use",
            academic_task="plan", stated_frequency="une fois", stated_reason=["faute de temps"])
EXCEPTION.s(2, "normative_formulation", "Je ne fais jamais faire mes plans, c'est mon travail.")
EXCEPTION.s(6, "exception", "Une fois")
EXCEPTION.contradiction((2, "Je ne fais jamais faire mes plans"), (6, "je lui ai demandé un plan détaillé"), "les plans")
EXCEPTION.rules = {2: decision(
    "accountability_episode",
    "L'étudiant dit ne jamais faire faire ses plans (« c'est mon travail »), puis rapporte une demande de plan "
    "présentée comme unique, « faute de temps ».",
    moves=[move("general_rule", "L'étudiant énonce ce qu'il fait pour ses plans.", 2),
           move("appeal_to_authorship", "L'étudiant rapporte le plan à son travail (« c'est mon travail »).", 2),
           move("exception", "L'étudiant présente la demande de plan comme une seule fois.", 6)],
    boundary=["faire / faire faire"], problem="Qui fait le plan ?", role="« c'est mon travail »",
    evidence=[(2, "Je ne fais jamais faire mes plans, c'est mon travail."),
              (6, "Une fois, faute de temps, je lui ai demandé un plan détaillé.")])}
EXCEPTION.good = plan(
    "contextual_configuration",
    claims=[claim("exception",
                  "L'étudiant énonce une règle (« Je ne fais jamais faire mes plans ») et rapporte un cas présenté comme "
                  "unique (« Une fois, faute de temps »).", items=[2], evidence=[2, 6], contexts=["plans"]),
            claim("unresolved_tension",
                  "Les deux formulations coexistent dans l'entretien : « Je ne fais jamais faire mes plans » et « je lui "
                  "ai demandé un plan détaillé » ; l'entretien ne les articule pas davantage.",
                  items=[2], evidence=[2, 6], contexts=["plans"], confidence="medium")],
    criteria=[criterion("être l'auteur de son plan",
                        "Dans cet entretien, l'étudiant associe le fait de faire ses plans à « mon travail ».",
                        items=[2], evidence=[2]),
              criterion("temps disponible",
                        "Dans cet entretien, l'étudiant rapporte l'unique demande de plan au manque de temps "
                        "(« faute de temps »).", items=[2], evidence=[6])],
    summary="Une règle explicite sur les plans et un cas présenté comme unique coexistent ; les fiches de lecture "
            "sont racontées sans restriction ni évaluation.")
EXCEPTION.adversarial = plan(
    "contextual_configuration",
    claims=[claim("exception",
                  "Sa véritable règle est de faire faire ses plans quand il manque de temps : il rationalise et se montre "
                  "hypocrite.", items=[2], evidence=[2, 6], confidence="high"),
            claim("exception", "Les fiches de lecture sont une exception.", items=[("P", 4)], evidence=[4])],
    summary="Une règle et un cas.")

# D. Stabilité ----------------------------------------------------------------------------------------
STABLE = Case("ETAPE5_STABILITE", _turns(
    "Quand est-ce que tu utilises l'IA ?",
    "Pour les fiches de révision, je l'utilise uniquement quand je manque de temps.",
    "Et pour les exposés ?",
    "Pour les exposés aussi, uniquement quand je manque de temps.",
    "Et pour les TD ?",
    "Pour les TD c'est pareil, uniquement quand je manque de temps.",
    "Et en dehors des cours ?",
    "Le week-end je lui demande des recettes de cuisine.",
))
for _n, _task, _quote in ((2, "fiches de révision", "Pour les fiches de révision, je l'utilise uniquement quand je manque de temps."),
                          (4, "exposés", "Pour les exposés aussi, uniquement quand je manque de temps."),
                          (6, "TD", "Pour les TD c'est pareil, uniquement quand je manque de temps.")):
    STABLE.p(_n, _quote, academic_task=_task)
    STABLE.s(_n, "restriction", "uniquement quand je manque de temps")
    STABLE.rules[_n] = decision(
        "accountability_episode", f"Pour {_task}, l'étudiant limite l'usage aux moments où il manque de temps.",
        moves=[move("restriction", "L'étudiant limite l'usage au manque de temps.", _n)],
        problem="Quand l'usage est-il admis ?", evidence=[(_n, _quote)])
STABLE.p(8, "Le week-end je lui demande des recettes de cuisine.", practice_domain="personal",
         assessment_context="personal")
STABLE.good = plan(
    "contextual_configuration",
    claims=[claim("stable_boundary",
                  "Pour trois tâches (fiches de révision, exposés, TD), l'étudiant borne l'usage de la même manière : "
                  "« uniquement quand je manque de temps ».", items=[2, 4, 6], evidence=[2, 4, 6],
                  contexts=["fiches de révision", "exposés", "TD"], confidence="high"),
            claim("recurring_accounting_move", "La même restriction est formulée dans les trois épisodes.",
                  items=[2, 4, 6], evidence=[2, 4, 6], moves=["restriction"], confidence="high")],
    criteria=[criterion("temps disponible",
                        "Dans cet entretien, l'étudiant associe l'usage au manque de temps (« uniquement quand je manque "
                        "de temps »).", items=[2, 4, 6], evidence=[2, 4, 6])],
    summary="Une même limite est reprise pour trois tâches : l'usage est réservé aux moments où le temps manque. Une "
            "demande de recettes est racontée sans restriction.")

# E. Zone ordinaire -------------------------------------------------------------------------------------
ORDINARY = Case("ETAPE5_ORDINAIRE", _turns(
    "Tu t'en sers pour quoi au quotidien ?",
    "Je lui demande les règles d'un jeu de société quand on joue entre amis.",
    "Et pour la fac ?",
    "Je lui demande où trouver une salle pour réviser près de la fac.",
    "D'accord.",
    "Je lui demande aussi les horaires de la bibliothèque.",
    "Et pour les dissertations ?",
    "Pour les dissertations je ne veux pas qu'il écrive à ma place.",
))
ORDINARY.p(2, "Je lui demande les règles d'un jeu de société", practice_domain="personal", academic_task="règles de jeu",
           assessment_context="personal")
ORDINARY.p(4, "Je lui demande où trouver une salle pour réviser près de la fac.", academic_task="recherche de lieu")
ORDINARY.p(6, "Je lui demande aussi les horaires de la bibliothèque.", academic_task="horaires de la bibliothèque")
ORDINARY.p(8, "je ne veux pas qu'il écrive à ma place", use_status="refusal", non_use_reason="personal_rule",
           academic_task="dissertations")
ORDINARY.s(8, "normative_formulation", "je ne veux pas qu'il écrive à ma place")
ORDINARY.rules = {8: decision(
    "accountability_episode", "L'étudiant dit ne pas vouloir que l'outil écrive ses dissertations à sa place.",
    moves=[move("refusal", "L'étudiant dit ne pas vouloir que l'outil écrive à sa place.", 8)],
    boundary=["écrire / faire écrire"], problem="Qui écrit la dissertation ?",
    evidence=[(8, "Pour les dissertations je ne veux pas qu'il écrive à ma place.")])}
ORDINARY.good = plan(
    "contextual_configuration",
    claims=[claim("ordinary_zone",
                  "Plusieurs recherches d'information (règles d'un jeu, salle pour réviser, horaires de la bibliothèque) "
                  "sont racontées sans restriction, justification ni évaluation explicite.",
                  items=[("P", 2), ("P", 4), ("P", 6)], evidence=[2, 4, 6],
                  contexts=["règles d'un jeu", "recherche de lieu", "horaires"], confidence="high")],
    criteria=[criterion("écrire soi-même ses dissertations",
                        "Dans cet entretien, l'étudiant associe les dissertations au fait d'écrire lui-même (« je ne veux "
                        "pas qu'il écrive à ma place »).", items=[8], evidence=[8])],
    summary="Les recherches d'information forment une zone racontée sans aucune justification ; seule l'écriture des "
            "dissertations fait l'objet d'une limite explicite.")
ORDINARY.adversarial = plan(
    "contextual_configuration",
    claims=[claim("ordinary_zone", "Les usages pour la fac sont ordinaires.", items=[("P", 4), 8], evidence=[4, 8]),
            claim("ordinary_zone", "Les horaires sont une zone ordinaire.", items=[("P", 6)], evidence=[6])],
    summary="Des recherches d'information.")

# F. Pas de trajectoire -----------------------------------------------------------------------------------
NOPATTERN = Case("ETAPE5_SANS_TRAJECTOIRE", _turns(
    "Tu utilises des IA ?",
    "Je lui demande des définitions quand je lis un article.",
    "Et pour les langues ?",
    "Je lui fais traduire des articles en espagnol.",
    "Et en dehors ?",
    "Je lui demande des idées de cadeaux.",
    "Et pour les exposés ?",
    "Je lui demande des exemples pour un exposé.",
))
NOPATTERN.p(2, "Je lui demande des définitions quand je lis un article.", academic_task="lectures")
NOPATTERN.p(4, "Je lui fais traduire des articles en espagnol.", academic_task="traduction")
NOPATTERN.p(6, "Je lui demande des idées de cadeaux.", practice_domain="personal", assessment_context="personal")
NOPATTERN.p(8, "Je lui demande des exemples pour un exposé.", academic_task="exposé")
NOPATTERN.good = plan(
    "no_clear_pattern",
    summary="Quatre usages sont racontés (définitions, traduction, idées de cadeaux, exemples pour un exposé), sans "
            "restriction, justification ni lien explicite entre eux. Aucun ancrage temporel n'est formulé.",
    confidence="medium")
NOPATTERN.adversarial = plan(
    "temporal_trajectory",
    claims=[claim("explicit_temporal_change",
                  "Au fil de l'entretien, l'étudiant devient plus autonome : il passe des définitions aux exemples pour "
                  "un exposé.", items=[("P", 2), ("P", 8)], evidence=[2, 8], confidence="high")],
    summary="L'entretien montre une prise de conscience progressive et une maturation de l'usage.", confidence="high")

# G. Épisode à revoir --------------------------------------------------------------------------------------
REVIEW = Case("ETAPE5_REVUE", [
    ("Enquêteur", "Tu l'utilises pour écrire ?"),
    ("Enquêté", "Je lui fais seulement reformuler mes paragraphes."),
    ("Enquêteur", "Et pour les introductions ?"),
    ("Enquêteur", "Moi personnellement je lui fais seulement reformuler mes intros, mais une fois je lui ai fait écrire "
                  "toute l'intro."),
    ("Enquêteur", "D'accord. Et pour lire ?"),
    ("Enquêté", "Je lui demande des idées de lecture."),
])
REVIEW_WARNED_TURN = 4
REVIEW.p(2, "Je lui fais seulement reformuler mes paragraphes.", academic_task="paragraphes")
REVIEW.p(4, "je lui fais seulement reformuler mes intros", academic_task="introductions", explicitness="unclear",
         uncertainty_note="Tour marqué enquêteur : attribution du locuteur douteuse.")
REVIEW.p(4, "une fois je lui ai fait écrire toute l'intro", use_status="past_use", academic_task="introductions",
         stated_frequency="une fois", explicitness="unclear")
REVIEW.p(6, "Je lui demande des idées de lecture.", academic_task="lectures")
REVIEW.s(2, "restriction", "seulement reformuler mes paragraphes")
REVIEW.s(4, "restriction", "seulement reformuler mes intros", needs_human_review=True)
REVIEW.s(4, "exception", "une fois", needs_human_review=True)
REVIEW.assessments = [assessment(
    turn_id=REVIEW.tid(4), suggested_speaker="enquete", confidence="high",
    reason="Réponse à la première personne qui suit directement la question du tour précédent.",
    evidence=[REVIEW.ev(4, "Moi personnellement je lui fais seulement reformuler mes intros"),
              REVIEW.ev(3, "Et pour les introductions ?")])]
REVIEW.rules = {
    2: decision("accountability_episode", "L'étudiant limite l'usage à la reformulation de ses paragraphes.",
                moves=[move("restriction", "L'étudiant limite l'usage à la reformulation.", 2)],
                problem="Jusqu'où l'outil intervient-il dans l'écriture ?",
                evidence=[(2, "Je lui fais seulement reformuler mes paragraphes.")]),
    4: decision("accountability_episode",
                "Le tour T0004, attribué à l'enquêteur dans la transcription (attribution douteuse), limite l'usage à la "
                "reformulation des introductions et rapporte une fois où l'outil a écrit toute l'introduction.",
                moves=[move("restriction", "Le passage limite l'usage à la reformulation des introductions.", 4),
                       move("exception", "Le passage rapporte une fois où l'outil a écrit l'introduction.", 4)],
                problem="Jusqu'où l'outil intervient-il dans l'écriture ?", confidence="medium", needs_review=True,
                evidence=[(4, "je lui fais seulement reformuler mes intros, mais une fois je lui ai fait écrire toute "
                              "l'intro")]),
}
REVIEW.good = plan(
    "contextual_configuration",
    claims=[claim("exception",
                  "Le passage limite l'usage à la reformulation des introductions et rapporte « une fois » où l'outil a "
                  "écrit toute l'introduction.", items=[4], evidence=[4], contexts=["introductions"], confidence="high"),
            claim("recurring_accounting_move",
                  "La même restriction (« seulement reformuler ») porte sur les paragraphes et sur les introductions.",
                  items=[2, 4], evidence=[2, 4], moves=["restriction"], contexts=["paragraphes", "introductions"])],
    summary="Une même restriction à la reformulation est formulée pour deux écrits ; l'un des passages repose sur un tour "
            "dont l'attribution du locuteur est douteuse.")

# H. Mot de l'enquêteur -------------------------------------------------------------------------------------
INTERVIEWER = Case("ETAPE5_ENQUETEUR", _turns(
    "Tu l'utilises pour tes exposés ?",
    "Oui, pour les exposés je lui demande des exemples.",
    "Tu n'as pas honte de l'utiliser pour tes exposés ? Ou peur que ça se voie ?",
    "Non, je l'utilise pour trouver des exemples, c'est tout, le reste je l'écris moi-même.",
    "Et pour les fiches ?",
    "Pour les fiches je lui demande des résumés.",
))
INTERVIEWER.p(2, "pour les exposés je lui demande des exemples", academic_task="exposés")
INTERVIEWER.p(4, "je l'utilise pour trouver des exemples, c'est tout", academic_task="exposés")
INTERVIEWER.p(4, "le reste je l'écris moi-même", use_status="non_use", non_use_reason="not_stated", academic_task="exposés")
INTERVIEWER.p(6, "Pour les fiches je lui demande des résumés.", academic_task="fiches")
INTERVIEWER.s(4, "restriction", "c'est tout")
INTERVIEWER.rules = {2: decision(
    "accountability_episode",
    "L'étudiant répond « Non » à la question de l'enquêteur, limite l'usage à la recherche d'exemples (« c'est tout ») "
    "et dit écrire le reste lui-même.",
    moves=[move("restriction", "L'étudiant limite l'usage à la recherche d'exemples.", 4),
           move("appeal_to_authorship", "L'étudiant dit écrire le reste lui-même.", 4)],
    boundary=["trouver des exemples / écrire"], problem="Qui écrit l'exposé ?",
    evidence=[(2, "pour les exposés je lui demande des exemples"),
              (4, "Non, je l'utilise pour trouver des exemples, c'est tout, le reste je l'écris moi-même.")])}
INTERVIEWER.good = plan(
    "contextual_configuration",
    claims=[claim("contextual_variation",
                  "Pour les exposés, l'étudiant limite l'usage à la recherche d'exemples et dit écrire le reste ; pour "
                  "les fiches, il raconte demander des résumés sans restriction.",
                  items=[4, ("P", 6)], evidence=[4, 6], contexts=["exposés", "fiches"])],
    criteria=[criterion("écrire soi-même l'exposé",
                        "Dans cet entretien, l'étudiant associe l'exposé au fait d'écrire lui-même (« le reste je l'écris "
                        "moi-même »).", items=[4], evidence=[4])],
    summary="L'étudiant répond « Non » à la question de l'enquêteur et limite l'usage pour les exposés à la recherche "
            "d'exemples ; les fiches sont racontées sans restriction.")
INTERVIEWER.adversarial = plan(
    "contextual_configuration",
    claims=[claim("contextual_variation", "L'étudiant exprime une peur d'être repéré pour ses exposés, pas pour les fiches.",
                  items=[4, ("P", 6)], evidence=[4, 6])],
    criteria=[criterion("ne pas avoir honte de son travail",
                        "Dans cet entretien, l'étudiant associe l'usage à la honte et à la peur que ça se voie.",
                        items=[4], evidence=[4])],
    summary="L'étudiant exprime de la honte à propos de ses exposés.")

CASES = {c.interview_id: c for c in (TEMPORAL, CONTEXT, EXCEPTION, STABLE, ORDINARY, NOPATTERN, REVIEW, INTERVIEWER)}


# --- Entretien de référence de l'étape 4 (tests/synthetic_stage4.py) -------------------------------------

REFERENCE_PLAN = plan(
    "contextual_configuration",
    claims=[claim("exception",
                  "L'étudiant dit faire « normalement » ses plans lui-même et rapporte une demande de plan présentée comme "
                  "unique (« Une fois »), avec la raison qu'il donne (« parce que j'étais en retard »).",
                  items=[6], evidence=[6, 12], contexts=["plans de dissertation"]),
            claim("recurring_accounting_move",
                  "Dans deux épisodes, l'étudiant dit ne pas faire faire (« je ne veux pas qu'il écrive à ma place », "
                  "« Je ne lui fais pas corriger mes dossiers »).", items=[4, 14], evidence=[4, 14], moves=["refusal"]),
            claim("ordinary_zone",
                  "Des explications de cours, des idées de lectures et des traductions d'articles sont racontées sans "
                  "restriction, justification ni évaluation explicite.",
                  items=[("P", 2), ("P", 8), ("P", 10)], evidence=[2, 8, 10],
                  contexts=["révisions", "recherche de sources", "traduction"])],
    criteria=[criterion("faire soi-même ses plans",
                        "Dans cet entretien, l'étudiant associe les plans au fait de les faire lui-même (« Normalement je "
                        "fais mes plans moi-même »).", items=[6], evidence=[6]),
              criterion("jugement du professeur",
                        "Dans cet entretien, l'étudiant associe le non-usage pour ses dossiers à ce que le professeur "
                        "pourrait penser (« J'aurais peur que le professeur pense que je n'ai rien fait »).",
                        items=[14], evidence=[14])],
    summary="Les usages se distribuent selon les tâches : écriture et dossiers tenus à distance, plans faits "
            "« normalement » par l'étudiant avec un cas présenté comme unique, explications, lectures et traductions "
            "racontées sans justification. Aucun ancrage temporel n'est formulé.")


def generic_plan() -> dict:
    """Lecteur simulé générique (entretien sans plan prévu) : aucune configuration décrite."""
    return plan("no_clear_pattern", summary="Le lecteur simulé n'a pas de plan prévu pour cet entretien.",
                confidence="low", needs_review=True)


# --- Pipeline simulé jusqu'à l'étape 4 (tests) -------------------------------------------------------------

def run_to_stage4(tmp_path, files, stage3_responders: dict, builder):
    """Ingestion, étape 3 et étape 4 (agents simulés) dans un dossier temporaire. Renvoie (run, cache)."""
    from core import accountability
    from core.analysis import analyze_run
    from core.analysis_cache import AnalysisCache
    from tests.fake_llm import ACCOUNTABILITY, FakeAgents, fake_settings
    from tests.synthetic_interviews import make_ingested_run

    run = make_ingested_run(tmp_path, files)
    cache = AnalysisCache(tmp_path / "cache")
    run = analyze_run(run, settings=fake_settings(), client=FakeAgents(stage3_responders), cache=cache)
    run = accountability.analyze_run_stage4(run, settings=fake_settings(), client=FakeAgents(
        {ACCOUNTABILITY: builder}), cache=cache)
    return run, cache


def case_to_stage4(tmp_path, case: Case):
    return run_to_stage4(tmp_path, case.files, case.stage3_responders(), case.builder())


def run_stage5(run, cache, mapper, **kwargs):
    """Étape 5 seule (seul le Trajectory Mapper répond : tout autre appel lèverait une erreur)."""
    from core import trajectory
    from tests.fake_llm import TRAJECTORY, FakeAgents, fake_settings

    transport = FakeAgents({TRAJECTORY: mapper})
    run = trajectory.analyze_run_stage5(run, settings=kwargs.pop("settings", fake_settings()), client=transport,
                                        cache=cache, **kwargs)
    return run, transport


def analysis_dir(run, index: int = 0):
    from pathlib import Path

    from core import config
    return Path(run["files"][index]["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR


def load(run, name: str, index: int = 0) -> dict:
    return json.loads((analysis_dir(run, index) / name).read_text(encoding="utf-8"))
