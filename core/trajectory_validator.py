"""Étape 5 — validation DÉTERMINISTE de la configuration produite par le Trajectory Mapper.

Aucun LLM. Pour chaque affirmation (`claims`) :
- appuis : chaque episode_id existe ET est utilisable (`usable_for_next_stages`) ; un épisode rejeté par
  l'étape 4 n'appuie jamais rien ; practice_ids = pratiques sans marqueur (une pratique d'un épisode est
  rapportée à cet épisode) ; au moins un appui ;
- tours : chaque evidence_turn_id existe ; au moins un tour de l'enquêté·e (une question de l'enquêteur
  ne fait pas une position) ; un tour hors du matériau des appuis est signalé ;
- ancrages temporels : copie EXACTE d'un passage du tour cité (même règle que les citations), tour de
  l'enquêté·e, dans le matériau des appuis, et expression réellement temporelle (core.trajectory_candidates) ;
- `explicit_temporal_change` : au moins deux états comparables (deux pratiques), une différence documentée
  (tâche ou statut d'usage) et des ancrages qui ORDONNENT les états (passé / présent, périodes de rangs
  différents, ou changement dit : « je ne … plus ») — sinon REQUALIFIÉ en `contextual_variation` :
  l'ordre des tours ne suffit jamais ;
- `exception` : une règle ou préférence explicite ET un cas (opération `exception`, « une fois », « sauf »,
  fréquence « une fois », ou usage contraire documenté sur la même tâche) — sinon REQUALIFIÉ en
  `contextual_variation` ;
- types de régularité : frontière stable / opération récurrente sur plusieurs épisodes, variation sur au
  moins deux contextes, tension sur au moins deux formulations, zone ordinaire sur au moins deux usages
  racontés sans accountability (jamais un épisode d'accountability) ;
- épisodes à revoir : une affirmation qui ne repose QUE sur des épisodes `needs_review` est marquée à revoir
  par TRACE (et signalée si sa confiance est « high ») ; sinon les épisodes à revoir sont listés ;
- vocabulaire : psychologisant ou récit de conversion (`FORBIDDEN_TERMS`), résolution d'une tension à la place
  de l'enquêté·e, vocabulaire du changement dans une affirmation non temporelle, ordre de l'entretien présenté
  comme un temps, comparaison avec d'autres entretiens ; un terme employé par l'enquêté·e dans une citation
  des appuis n'est pas signalé ; un terme proposé par l'enquêteur et non repris est une ERREUR.

Critères du « métier d'étudiant » (`student_role_criteria`) : mêmes contrôles d'appuis et de tours ; critère
appuyé sur un épisode qui le formule (opération, frontière ou référence au rôle) ; formulation située
(« Dans cet entretien… ») et non généralisée ; un affect non prononcé par l'enquêté·e est une ERREUR.

Configuration : `temporal_trajectory` / `mixed` exigent un changement temporel validé, sinon requalifiés ;
les listes du document (frontières stables, variations…) sont calculées par TRACE à partir des affirmations
retenues. Rien n'est réécrit : TRACE ajoute `validation_status`, `needs_review`, `review_reasons` et, en cas
de requalification, `model_claim_type`. Erreur → objet `rejected` (conservé, exclu des listes).
"""

from __future__ import annotations

import re

from agents.trajectory_mapper import CLAIM_TYPES
from core import evidence_validator, interpretation_guard
from core import trajectory_candidates as tc
from core.accountability_candidates import NON_USE_LIKE, USE_LIKE
from core.accountability_episode_validator import FORBIDDEN_TERMS as STAGE4_FORBIDDEN_TERMS
from core.schemas import SPEAKER_INTERVIEWER

VALIDATOR_VERSION = "1.0"

ERROR, WARNING, INFO = "error", "warning", "info"

ISSUE_CODES = {
    # appuis et tours
    "UNKNOWN_EPISODE_ID": (ERROR, "episode_id inexistant dans les sorties de l'étape 4."),
    "UNUSABLE_EPISODE": (ERROR, "Épisode rejeté par l'étape 4 (usable_for_next_stages = false) : il n'appuie rien."),
    "UNKNOWN_PRACTICE_ID": (ERROR, "practice_id absent des pratiques sans marqueur de cet entretien."),
    "PRACTICE_RESOLVED_TO_EPISODE": (INFO, "Pratique d'un épisode citée directement : rapportée à cet épisode."),
    "NO_SUPPORT": (ERROR, "Aucun épisode ni aucune pratique valide n'appuie cette affirmation."),
    "UNKNOWN_EVIDENCE_TURN": (ERROR, "Tour (evidence_turn_ids) inexistant dans cet entretien."),
    "FOREIGN_INTERVIEW_TURN": (ERROR, "Tour appartenant à un autre entretien."),
    "NO_INTERVIEWEE_EVIDENCE": (ERROR, "Aucun tour de l'enquêté·e parmi les tours cités : une question de "
                                       "l'enquêteur ne fait pas une position."),
    "EVIDENCE_TURN_INTERVIEWER": (WARNING, "Tour cité attribué à l'enquêteur."),
    "EVIDENCE_TURN_OUTSIDE_MATERIAL": (WARNING, "Tour cité absent du matériau des épisodes et pratiques invoqués."),
    # ancrages temporels
    "ANCHOR_TURN_UNKNOWN": (ERROR, "Ancrage temporel sur un tour inexistant."),
    "ANCHOR_NOT_FOUND": (ERROR, "Ancrage temporel absent du texte du tour cité (citation inventée ou inexacte)."),
    "ANCHOR_FROM_INTERVIEWER": (WARNING, "Ancrage temporel prononcé par l'enquêteur : il n'ordonne pas les pratiques "
                                         "de l'enquêté·e."),
    "ANCHOR_NOT_TEMPORAL": (WARNING, "Expression citée comme ancrage, sans expression temporelle reconnue."),
    "ANCHOR_OUTSIDE_MATERIAL": (WARNING, "Ancrage temporel hors du matériau des épisodes et pratiques invoqués."),
    # types d'affirmations
    "TEMPORAL_CHANGE_REQUALIFIED": (WARNING, "Changement temporel sans ancrage explicite permettant d'ordonner deux "
                                             "états comparables : requalifié en variation contextuelle (l'ordre de "
                                             "l'entretien ne suffit pas)."),
    "EXCEPTION_REQUALIFIED": (WARNING, "Exception sans règle explicite ou sans cas présenté comme exception : "
                                       "requalifiée en variation contextuelle."),
    "VARIATION_NOT_DOCUMENTED": (WARNING, "Variation contextuelle appuyée sur un seul contexte documenté."),
    "SINGLE_SUPPORT_PATTERN": (WARNING, "Régularité (frontière stable, opération récurrente, zone ordinaire) appuyée "
                                        "sur un seul élément."),
    "BOUNDARY_NOT_DOCUMENTED": (WARNING, "Frontière stable sans frontière, règle ou restriction explicite dans au "
                                         "moins deux épisodes."),
    "RECURRENCE_NOT_DOCUMENTED": (WARNING, "Opération présentée comme récurrente, absente d'au moins deux épisodes."),
    "TENSION_SINGLE_FORMULATION": (WARNING, "Tension appuyée sur une seule formulation de l'enquêté·e."),
    "ORDINARY_ZONE_WITH_ACCOUNTABILITY": (WARNING, "Zone ordinaire contenant un épisode d'accountability ou incertain."),
    # épisodes à revoir, locuteurs
    "SUPPORTED_ONLY_BY_REVIEW_EPISODES": (INFO, "Affirmation appuyée uniquement sur des épisodes à revoir : marquée à "
                                                "revoir par TRACE."),
    "STRONG_CLAIM_ON_REVIEW_EPISODES": (WARNING, "Confiance « high » alors que tous les appuis sont à revoir."),
    "REVIEW_EPISODE_PROPAGATED": (INFO, "Certains appuis sont des épisodes à revoir."),
    "SPEAKER_WARNING_PROPAGATED": (INFO, "Un tour cité ou un appui a une attribution de locuteur douteuse."),
    "MODEL_FLAGGED_REVIEW": (INFO, "Le modèle demande une vérification humaine."),
    # vocabulaire
    "INTERPRETIVE_VOCABULARY": (WARNING, "Vocabulaire psychologisant, d'intention ou de récit de conversion."),
    "INTERPRETIVE_CRITERION": (ERROR, "Critère formulé avec un affect ou une catégorie psychologique que l'enquêté·e "
                                      "n'emploie pas."),
    "INTERVIEWER_TERM_ATTRIBUTED": (ERROR, "Terme proposé par l'enquêteur, non repris par l'enquêté·e, attribué à "
                                           "l'enquêté·e."),
    "RESOLVING_VOCABULARY": (WARNING, "Tension résolue à la place de l'enquêté·e (« véritable règle », « en réalité », "
                                      "jugement)."),
    "TEMPORAL_WORDING_WITHOUT_ANCHOR": (WARNING, "Vocabulaire du changement (évolue, devient, désormais…) dans une "
                                                 "affirmation sans changement temporel validé."),
    "INTERVIEW_ORDER_AS_TIME": (WARNING, "L'ordre de l'entretien est présenté comme un ordre temporel."),
    "CROSS_INTERVIEW_COMPARISON": (WARNING, "Comparaison avec d'autres entretiens ou étudiant·es (hors étape 5)."),
    # critères du métier d'étudiant
    "CRITERION_NOT_GROUNDED": (WARNING, "Critère sans épisode qui le formule (opération, frontière ou référence au rôle)."),
    "CRITERION_GENERALIZED": (WARNING, "Critère généralisé (« le métier d'étudiant consiste… », « la vraie définition… »)."),
    "CRITERION_NOT_SITUATED": (WARNING, "Critère non situé dans cet entretien (« Dans cet entretien, l'étudiant·e "
                                        "associe… »)."),
    # document
    "CONFIGURATION_REQUALIFIED": (WARNING, "Configuration temporelle sans changement temporel validé : requalifiée."),
    "CONFIGURATION_INCONSISTENT": (WARNING, "Changement temporel validé, mais configuration déclarée non temporelle."),
    "SUMMARY_VOCABULARY": (WARNING, "Synthèse : vocabulaire psychologisant, de conversion ou proposé par l'enquêteur."),
    "SUMMARY_TEMPORAL_WORDING": (WARNING, "Synthèse : vocabulaire du changement sans changement temporel validé."),
    "SUMMARY_CROSS_INTERVIEW": (WARNING, "Synthèse : comparaison avec d'autres entretiens."),
    "PAYLOAD_OVER_THRESHOLD": (WARNING, "Représentation envoyée au-delà du seuil d'un appel unique."),
    "STAGE4_INCOMPLETE": (WARNING, "Sorties de l'étape 4 incomplètes : configuration incomplète."),
}

# Vocabulaire interdit : celui de l'étape 4 (affects, intentions, stratégie, légitimation…) + récit de conversion.
STAGE5_FORBIDDEN_TERMS = {
    "maturation / maturité": r"matur\w*",
    "prise de conscience": r"pri\w* (?:de )?conscience|prend\w* conscience",
    "devenir (plus) responsable": r"(?:devien\w*|devenu\w*|plus) responsabl\w*|responsabilis\w*",
    "apprendre à mieux utiliser": r"apprend\w* a mieux|appris a mieux|mieux utiliser",
    "normalisation (progressive)": r"normalis\w*",
    "dépendance": r"dependan\w*",
    "émancipation": r"emancip\w*",
    "conversion": r"conversion\w*|converti\w*",
    "hypocrisie": r"hypocri\w*",
    "mensonge": r"mensong\w*",
}
FORBIDDEN_TERMS = {**STAGE4_FORBIDDEN_TERMS, **STAGE5_FORBIDDEN_TERMS}
_FORBIDDEN = [(label, re.compile(r"\b(?:" + pattern + ")")) for label, pattern in FORBIDDEN_TERMS.items()]
_RESOLVING = re.compile(r"\b(?:veritabl\w*|vraie? (?:regle|position|raison|definition|pratique)|en realite\b|au fond\b|"
                        r"incoheren\w*|contradiction (?:montre|revele|trahit)|se contredit)")
CHANGE_TERMS = re.compile(r"\b(?:evolu\w*|progressiv\w*|devenu\w*|devient|deviennent|transform\w*|"
                          r"au fil (?:du temps|des (?:annees|mois|semaines)|de l.entretien)|desormais|dorenavant|"
                          r"de plus en plus|de moins en moins|a change\b|ont change\b)")
_INTERVIEW_ORDER = re.compile(r"\b(?:au fil de l.entretien|au cours de l.entretien|plus (?:loin|tard) dans l.entretien|"
                              r"(?:en|au) debut (?:de l.|d.)entretien|(?:en|a la) fin (?:de l.|d.)entretien)")
_CROSS_INTERVIEW = re.compile(r"\b(?:autres? (?:etudiant|enquete|entretien)\w*|typologi\w*|types? d.etudiant\w*|corpus\b|"
                              r"comme (?:beaucoup|la plupart) d|profils? (?:type|d.etudiant)\w*|"
                              r"les etudiant\w* (?:en general|de sciences po))")
_GENERALIZED = re.compile(r"\b(?:metier d.etudiant consiste|consiste a etre|la vraie definition|veritable definition|"
                          r"definition du metier|un bon etudiant|tout etudiant|les etudiant\w* doivent|"
                          r"l.etudiant\w* doit\b)")
_SITUATED = re.compile(r"\bdans cet entretien\b")
ROLE_MOVES = frozenset({"refusal", "restriction", "distinction", "general_rule", "preference", "appeal_to_control",
                        "appeal_to_verification", "appeal_to_effort", "appeal_to_learning", "appeal_to_authorship",
                        "appeal_to_external_judgment"})
PATTERN_TYPES = ("stable_boundary", "recurring_accounting_move", "ordinary_zone")
LIST_KEYS = {"stable_boundary": "stable_boundaries", "contextual_variation": "contextual_variations",
             "explicit_temporal_change": "explicit_temporal_changes", "exception": "exceptions",
             "unresolved_tension": "unresolved_tensions", "ordinary_zone": "ordinary_zones",
             "recurring_accounting_move": "recurring_accounting_moves"}


def make_issue(code: str, **context) -> dict:
    severity, message = ISSUE_CODES[code]
    issue = {"code": code, "severity": severity, "message": message}
    issue.update({k: v for k, v in context.items() if v is not None})
    return issue


def _fold(texts) -> str:
    return interpretation_guard.fold(" ".join(t for t in texts if t))


def vocabulary_findings(authored: str, interviewee: str, interviewer: str) -> list[dict]:
    """Termes interdits d'un texte rédigé, sauf s'ils figurent dans les citations de l'enquêté·e."""
    findings = []
    for label, regex in _FORBIDDEN:
        if regex.search(authored) and not regex.search(interviewee):
            findings.append({"term": label, "from_interviewer": bool(regex.search(interviewer))})
    return findings


class _Context:
    """Index de l'entretien partagé par la validation des affirmations et des critères."""

    def __init__(self, material: dict, transcript: dict, speaker_warnings: dict | None):
        self.interview_id = transcript["interview_id"]
        self.turns = evidence_validator.index_turns(transcript)
        self.order = [t["turn_id"] for t in transcript["turns"]]
        self.warnings = speaker_warnings or {}
        self.items = material["items"]
        self.excluded = {e["episode_id"] for e in material["excluded_episodes"]}
        self.practice_owner = {pid: item["id"] for item in self.items.values() if item["kind"] == "episode"
                               for pid in item["practice_ids"]}

    def voiced(self, turn_id: str) -> bool:
        return turn_id in self.turns and (self.turns[turn_id]["speaker"] != SPEAKER_INTERVIEWER
                                          or turn_id in self.warnings)

    def interviewer_context(self, supports: list[dict]) -> list[str]:
        """Questions de l'enquêteur dans l'intervalle des appuis et juste avant chacun de leurs tours."""
        positions = set()
        for item in supports:
            start, end = (self.turns.get(item["turn_start"]) or {}), (self.turns.get(item["turn_end"]) or {})
            if "position" in start and "position" in end:
                positions.update(range(start["position"], end["position"] + 1))
            for t in item["material_turns"]:
                positions.update({self.turns[t]["position"], self.turns[t]["position"] - 1})
        return [self.turns[self.order[p]]["text"] for p in sorted(positions)
                if 0 <= p < len(self.order) and not self.voiced(self.order[p])]

    def resolve_supports(self, episode_ids: list[str], practice_ids: list[str], issues: list[dict]) -> list[dict]:
        supports: dict[str, dict] = {}
        for eid in episode_ids:
            if eid in self.items and self.items[eid]["kind"] == "episode":
                supports[eid] = self.items[eid]
            elif eid in self.excluded:
                issues.append(make_issue("UNUSABLE_EPISODE", episode_id=eid))
            else:
                issues.append(make_issue("UNKNOWN_EPISODE_ID", episode_id=eid))
        for pid in practice_ids:
            if pid in self.items and self.items[pid]["kind"] == "unmarked":
                supports[pid] = self.items[pid]
            elif pid in self.practice_owner:
                owner = self.practice_owner[pid]
                supports[owner] = self.items[owner]
                issues.append(make_issue("PRACTICE_RESOLVED_TO_EPISODE", practice_id=pid, episode_id=owner))
            else:
                issues.append(make_issue("UNKNOWN_PRACTICE_ID", practice_id=pid))
        if not supports:
            issues.append(make_issue("NO_SUPPORT"))
        return list(supports.values())

    def check_turns(self, turn_ids: list[str], supports: list[dict], issues: list[dict]) -> list[str]:
        """Tours cités valides ; renvoie les tours de l'enquêté·e."""
        material = {t for s in supports for t in s["material_turns"]}
        voiced = []
        for turn_id in dict.fromkeys(turn_ids):
            problem = evidence_validator.turn_problem(turn_id, self.turns, self.interview_id)
            if problem:
                issues.append(make_issue("UNKNOWN_EVIDENCE_TURN" if problem == "UNKNOWN_TURN_ID" else problem,
                                         turn_id=turn_id))
                continue
            if self.voiced(turn_id):
                voiced.append(turn_id)
            else:
                issues.append(make_issue("EVIDENCE_TURN_INTERVIEWER", turn_id=turn_id))
            if supports and turn_id not in material:
                issues.append(make_issue("EVIDENCE_TURN_OUTSIDE_MATERIAL", turn_id=turn_id))
        if turn_ids and not voiced and not any(i["code"] in ("UNKNOWN_EVIDENCE_TURN", "FOREIGN_INTERVIEW_TURN")
                                               for i in issues):
            issues.append(make_issue("NO_INTERVIEWEE_EVIDENCE"))
        return voiced

    def review(self, obj: dict, supports: list[dict], voiced_turns: list[str], issues: list[dict]) -> list[str]:
        """Propagation des épisodes à revoir et des avertissements de locuteur. Renvoie les raisons pour lesquelles
        TRACE force la revue (vide si aucune)."""
        forced = []
        reviewed = [s["id"] for s in supports if s["needs_review"]]
        if supports and len(reviewed) == len(supports):
            forced.append("SUPPORTED_ONLY_BY_REVIEW_EPISODES")
            issues.append(make_issue("SUPPORTED_ONLY_BY_REVIEW_EPISODES", episode_ids=reviewed))
            if obj.get("confidence") == "high":
                issues.append(make_issue("STRONG_CLAIM_ON_REVIEW_EPISODES", episode_ids=reviewed))
        elif reviewed:
            issues.append(make_issue("REVIEW_EPISODE_PROPAGATED", episode_ids=reviewed))
        warned = [t for t in dict.fromkeys([*voiced_turns, *(t for s in supports for t in s["speaker_warning_turns"])])
                  if t in self.warnings]
        if warned:
            issues.append(make_issue("SPEAKER_WARNING_PROPAGATED", turn_ids=warned))
            if voiced_turns and all(t in self.warnings for t in voiced_turns):
                forced.append("SPEAKER_WARNING_PROPAGATED")  # toute la parole citée est d'attribution douteuse
        if obj.get("needs_review"):
            forced.append("MODEL_FLAGGED_REVIEW")
            issues.append(make_issue("MODEL_FLAGGED_REVIEW"))
        obj["_speaker_warnings"] = [{"turn_id": t, "suggested_speaker": self.warnings[t].get("suggested_speaker"),
                                     "confidence": self.warnings[t].get("confidence")} for t in warned]
        obj["_review_episode_ids"] = reviewed
        return forced


def _support_quotes(supports: list[dict]) -> list[str]:
    return [q["quote"] for s in supports for q in s["voiced_quotes"]]


def _status(issues: list[dict]) -> str:
    severities = {i["severity"] for i in issues}
    return "rejected" if ERROR in severities else "needs_review" if WARNING in severities else "valid"


def _check_anchors(claim: dict, supports: list[dict], ctx: _Context, issues: list[dict]) -> list[dict]:
    material = {t for s in supports for t in s["material_turns"]}
    valid = []
    for index, anchor in enumerate(claim.get("temporal_anchors", [])):
        turn_id, text = anchor.get("turn_id"), anchor.get("text") or ""
        if evidence_validator.turn_problem(turn_id, ctx.turns, ctx.interview_id):
            issues.append(make_issue("ANCHOR_TURN_UNKNOWN", anchor_index=index, turn_id=turn_id))
            continue
        if not text.strip() or evidence_validator.match_quote(text, ctx.turns[turn_id]["text"]) != "exact":
            issues.append(make_issue("ANCHOR_NOT_FOUND", anchor_index=index, turn_id=turn_id))
            continue
        if not ctx.voiced(turn_id):
            issues.append(make_issue("ANCHOR_FROM_INTERVIEWER", anchor_index=index, turn_id=turn_id))
            continue
        kind = tc.anchor_kind(text)
        if kind is None:
            issues.append(make_issue("ANCHOR_NOT_TEMPORAL", anchor_index=index, turn_id=turn_id))
            continue
        if turn_id not in material:
            issues.append(make_issue("ANCHOR_OUTSIDE_MATERIAL", anchor_index=index, turn_id=turn_id))
            continue
        valid.append({"turn_id": turn_id, "text": text, "kind": kind[0], "rank": kind[1]})
    return valid


def temporal_requirements(supports: list[dict], anchors: list[dict]) -> dict:
    """Deux états comparables, une différence documentée, des ancrages qui les ordonnent."""
    facts = [f for s in supports for f in s["facts"]]
    practice_ids = {p for s in supports for p in s["practice_ids"]}
    basis = tc.ordering_basis([(a["kind"], a["rank"]) for a in anchors])
    transition = basis == "transition"
    states = len(practice_ids) >= 2 or transition
    difference = transition or len({(f["task_key"], f["use_status"]) for f in facts}) >= 2
    missing = [name for name, ok in (("ordering_anchors", basis), ("two_states", states), ("difference", difference))
               if not ok]
    return {"ordering_basis": basis, "missing": missing}


def exception_requirements(supports: list[dict]) -> dict:
    rule = any(s["rule"] for s in supports)
    case = any(s["case"] for s in supports)
    if not case:  # conduite contraire documentée : non-usage / refus d'un appui et usage d'un autre, même tâche
        refused = {f["task_key"]: s["id"] for s in supports if s["rule"] for f in s["facts"]
                   if f["task_key"] and f["use_status"] in NON_USE_LIKE}
        case = any(f["task_key"] in refused and refused[f["task_key"]] != s["id"]
                   for s in supports for f in s["facts"] if f["use_status"] in USE_LIKE)
    return {"missing": [name for name, ok in (("rule", rule), ("case", case)) if not ok]}


def _type_checks(claim_type: str, claim: dict, supports: list[dict], voiced_turns: list[str], issues: list[dict]):
    episodes = [s for s in supports if s["kind"] == "episode"]
    if claim_type == "contextual_variation":
        keys = {k for s in supports for k in s["task_keys"]} or {d for s in supports for d in s["domains"]}
        if len(keys) < 2:
            issues.append(make_issue("VARIATION_NOT_DOCUMENTED", contexts=sorted(keys)))
    elif claim_type == "stable_boundary":
        if len(episodes) < 2:
            issues.append(make_issue("SINGLE_SUPPORT_PATTERN", support_count=len(episodes)))
        elif sum(bool(s["boundaries"]) or s["rule"] for s in episodes) < 2:
            issues.append(make_issue("BOUNDARY_NOT_DOCUMENTED"))
    elif claim_type == "recurring_accounting_move":
        if len(episodes) < 2:
            issues.append(make_issue("SINGLE_SUPPORT_PATTERN", support_count=len(episodes)))
        else:
            counts: dict[str, int] = {}
            for s in episodes:
                for move_type in {m["type"] for m in s["moves"]}:
                    counts[move_type] = counts.get(move_type, 0) + 1
            wanted = claim.get("accounting_move_types") or []
            missing = [t for t in wanted if counts.get(t, 0) < 2] if wanted else (
                [] if any(c >= 2 for c in counts.values()) else ["(aucune opération commune)"])
            if missing:
                issues.append(make_issue("RECURRENCE_NOT_DOCUMENTED", move_types=missing))
    elif claim_type == "unresolved_tension":
        if len(set(voiced_turns)) < 2:
            issues.append(make_issue("TENSION_SINGLE_FORMULATION"))
    elif claim_type == "ordinary_zone":
        if len(supports) < 2:
            issues.append(make_issue("SINGLE_SUPPORT_PATTERN", support_count=len(supports)))
        accountable = [s["id"] for s in supports if s["status"] in ("accountability_episode", "uncertain")]
        if accountable:
            issues.append(make_issue("ORDINARY_ZONE_WITH_ACCOUNTABILITY", episode_ids=accountable))


def _common_vocabulary(text: str, supports: list[dict], ctx: _Context, issues: list[dict], *, criterion: bool,
                       temporal: bool) -> None:
    authored = interpretation_guard.fold(text)
    interviewee = _fold(_support_quotes(supports))
    interviewer = _fold(ctx.interviewer_context(supports))
    for finding in vocabulary_findings(authored, interviewee, interviewer):
        if finding["from_interviewer"]:
            issues.append(make_issue("INTERVIEWER_TERM_ATTRIBUTED", term=finding["term"]))
        issues.append(make_issue("INTERPRETIVE_CRITERION" if criterion else "INTERPRETIVE_VOCABULARY",
                                 term=finding["term"]))
    if _RESOLVING.search(authored) and not _RESOLVING.search(interviewee):
        issues.append(make_issue("RESOLVING_VOCABULARY"))
    change = CHANGE_TERMS.search(authored)
    if change and not CHANGE_TERMS.search(interviewee):
        if not temporal:
            issues.append(make_issue("TEMPORAL_WORDING_WITHOUT_ANCHOR", term=change.group(0)))
        if _INTERVIEW_ORDER.search(authored):
            issues.append(make_issue("INTERVIEW_ORDER_AS_TIME"))
    if _CROSS_INTERVIEW.search(authored) and not _CROSS_INTERVIEW.search(interviewee):
        issues.append(make_issue("CROSS_INTERVIEW_COMPARISON"))


def _annotate(obj: dict, id_key: str, object_id: str, issues: list[dict], forced: list[str], extra: dict) -> dict:
    status = _status(issues)
    annotated = {id_key: object_id,
                 **{k: v for k, v in obj.items() if not k.startswith("_") and k != "needs_review"}, **extra,
                 "model_needs_review": bool(obj.get("needs_review")),
                 "needs_review": bool(forced) or status != "valid",
                 "review_episode_ids": obj.get("_review_episode_ids", []),
                 "speaker_warnings": obj.get("_speaker_warnings", []),
                 "validation_status": status, "usable_for_next_stages": status != "rejected",
                 "review_reasons": sorted({i["code"] for i in issues if i["severity"] != INFO} | set(forced))}
    return annotated


def validate_claim(claim: dict, claim_id: str, ctx: _Context) -> tuple[dict, list[dict]]:
    issues: list[dict] = []
    supports = ctx.resolve_supports(claim.get("episode_ids", []), claim.get("practice_ids", []), issues)
    voiced_turns = ctx.check_turns(claim.get("evidence_turn_ids", []), supports, issues)
    anchors = _check_anchors(claim, supports, ctx, issues)
    model_type = claim.get("claim_type")
    final_type = model_type
    basis = None
    if model_type == "explicit_temporal_change":
        requirements = temporal_requirements(supports, anchors)
        basis = requirements["ordering_basis"]
        if requirements["missing"]:
            final_type = "contextual_variation"
            issues.append(make_issue("TEMPORAL_CHANGE_REQUALIFIED", missing=requirements["missing"]))
    elif model_type == "exception":
        requirements = exception_requirements(supports)
        if requirements["missing"]:
            final_type = "contextual_variation"
            issues.append(make_issue("EXCEPTION_REQUALIFIED", missing=requirements["missing"]))
    if supports:
        _type_checks(final_type, claim, supports, voiced_turns, issues)
    forced = ctx.review(claim, supports, voiced_turns, issues)
    _common_vocabulary(" ".join([claim.get("description") or "", *claim.get("contexts", [])]), supports, ctx, issues,
                       criterion=False, temporal=final_type == "explicit_temporal_change")
    extra = {"claim_type": final_type, **({"model_claim_type": model_type} if final_type != model_type else {}),
             "validated_temporal_anchors": anchors, "temporal_ordering_basis": basis,
             "support_ids": [s["id"] for s in supports]}
    annotated = _annotate(claim, "claim_id", claim_id, issues, forced, extra)
    return annotated, issues


def validate_criterion(criterion: dict, criterion_id: str, ctx: _Context) -> tuple[dict, list[dict]]:
    issues: list[dict] = []
    supports = ctx.resolve_supports(criterion.get("episode_ids", []), [], issues)
    voiced_turns = ctx.check_turns(criterion.get("evidence_turn_ids", []), supports, issues)
    grounded = any(s["status"] == "accountability_episode" and (
        s["role"] or s["boundaries"] or {m["type"] for m in s["moves"]} & ROLE_MOVES) for s in supports)
    if supports and not grounded:
        issues.append(make_issue("CRITERION_NOT_GROUNDED"))
    forced = ctx.review(criterion, supports, voiced_turns, issues)
    text = " ".join([criterion.get("criterion") or "", criterion.get("description") or ""])
    _common_vocabulary(text, supports, ctx, issues, criterion=True, temporal=True)
    folded = interpretation_guard.fold(text)
    if _GENERALIZED.search(folded):
        issues.append(make_issue("CRITERION_GENERALIZED"))
    if not _SITUATED.search(folded):
        issues.append(make_issue("CRITERION_NOT_SITUATED"))
    annotated = _annotate(criterion, "criterion_id", criterion_id, issues, forced,
                          {"support_ids": [s["id"] for s in supports]})
    return annotated, issues


def validate_trajectory(output: dict, material: dict, transcript: dict, speaker_warnings: dict | None = None) -> dict:
    """Valide la sortie (identifiants entiers) d'UN entretien. Renvoie {"document": champs, "report": bilan}."""
    ctx = _Context(material, transcript, speaker_warnings)
    interview_id = ctx.interview_id
    all_issues: list[dict] = []
    claims = []
    for number, claim in enumerate(output.get("claims", []), start=1):
        claim_id = f"{interview_id}_TC{number:03d}"
        annotated, issues = validate_claim(claim, claim_id, ctx)
        claims.append(annotated)
        all_issues.extend({"object_id": claim_id, **i} for i in issues)
    criteria = []
    for number, criterion in enumerate(output.get("student_role_criteria", []), start=1):
        criterion_id = f"{interview_id}_RC{number:03d}"
        annotated, issues = validate_criterion(criterion, criterion_id, ctx)
        criteria.append(annotated)
        all_issues.extend({"object_id": criterion_id, **i} for i in issues)

    kept = [c for c in claims if c["usable_for_next_stages"]]
    lists = {key: [c["claim_id"] for c in kept if c["claim_type"] == claim_type] for claim_type, key in LIST_KEYS.items()}
    doc_issues: list[dict] = []
    model_configuration = output.get("configuration_type")
    configuration = model_configuration
    temporal = lists["explicit_temporal_changes"]
    if model_configuration in ("temporal_trajectory", "mixed") and not temporal:
        configuration = "contextual_configuration" if kept else "no_clear_pattern"
        doc_issues.append(make_issue("CONFIGURATION_REQUALIFIED", model_configuration=model_configuration,
                                     configuration=configuration))
    elif model_configuration == "contextual_configuration" and not kept:
        configuration = "no_clear_pattern"
        doc_issues.append(make_issue("CONFIGURATION_REQUALIFIED", model_configuration=model_configuration,
                                     configuration=configuration))
    elif temporal and model_configuration in ("contextual_configuration", "no_clear_pattern"):
        doc_issues.append(make_issue("CONFIGURATION_INCONSISTENT", claim_ids=temporal))

    summary = output.get("trajectory_summary") or ""
    folded = interpretation_guard.fold(summary)
    items = list(ctx.items.values())
    interviewee = _fold(_support_quotes(items))
    interviewer = _fold(ctx.interviewer_context(items))
    findings = vocabulary_findings(folded, interviewee, interviewer)
    if findings:
        doc_issues.append(make_issue("SUMMARY_VOCABULARY", terms=[f["term"] for f in findings],
                                     from_interviewer=[f["term"] for f in findings if f["from_interviewer"]] or None))
    change = CHANGE_TERMS.search(folded)
    if change and configuration not in ("temporal_trajectory", "mixed") and not CHANGE_TERMS.search(interviewee):
        doc_issues.append(make_issue("SUMMARY_TEMPORAL_WORDING", term=change.group(0)))
    if _CROSS_INTERVIEW.search(folded) and not _CROSS_INTERVIEW.search(interviewee):
        doc_issues.append(make_issue("SUMMARY_CROSS_INTERVIEW"))
    all_issues.extend({"object_id": interview_id, **i} for i in doc_issues)
    if output.get("needs_review"):
        all_issues.append({"object_id": interview_id, **make_issue("MODEL_FLAGGED_REVIEW")})

    severities = [i["severity"] for i in all_issues]
    document = {
        "configuration_type": configuration,
        **({"model_configuration_type": model_configuration} if configuration != model_configuration else {}),
        "trajectory_claims": claims,
        **lists,
        "student_role_criteria": criteria,
        "trajectory_summary": summary,
        "confidence": output.get("confidence"),
        "model_needs_review": bool(output.get("needs_review")),
        "needs_review": bool(output.get("needs_review")) or ERROR in severities or WARNING in severities,
        "mapper_notes": output.get("mapper_notes"),
    }
    report = {
        "validator_version": VALIDATOR_VERSION,
        "claim_count": len(claims),
        "kept_claim_count": len(kept),
        "rejected_claim_ids": [c["claim_id"] for c in claims if not c["usable_for_next_stages"]],
        "requalified_claim_ids": [c["claim_id"] for c in claims if "model_claim_type" in c],
        "claims_needing_review": [c["claim_id"] for c in claims if c["needs_review"]],
        "criterion_count": len(criteria),
        "rejected_criterion_ids": [c["criterion_id"] for c in criteria if not c["usable_for_next_stages"]],
        "configuration_requalified": configuration != model_configuration,
        "error_count": severities.count(ERROR),
        "warning_count": severities.count(WARNING),
        "info_count": severities.count(INFO),
        "issues": all_issues,
    }
    return {"document": document, "report": report}


def has_problems(report: dict) -> bool:
    return bool(report.get("error_count") or report.get("warning_count"))


assert set(LIST_KEYS) == set(CLAIM_TYPES)
