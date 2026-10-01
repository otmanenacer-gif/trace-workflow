"""Validation déterministe des preuves produites par les agents (après la réponse du LLM).

Pour chaque pratique ou signal, vérifie :
1. que chaque turn_id cité existe dans CET entretien (un turn_id d'un autre
   entretien est signalé comme tel) ;
2. que chaque citation est une sous-chaîne EXACTE du texte du tour cité
   (seule la normalisation Unicode NFC est appliquée : même texte, même
   encodage canonique) ;
3. que turn_start <= turn_end (pratiques) ;
4. quelques cohérences locales (preuve hors de l'intervalle, affect
   « explicite » absent des citations, contradiction appuyée sur un seul
   tour, vocabulaire interprétatif, non_use_reason incohérent avec le
   statut, évaluation « métadiscursive » sans référence à sa propre parole,
   citation d'un tour dont l'attribution du locuteur est douteuse…).

Les mêmes contrôles de citations servent à l'auditeur des locuteurs
(core/speaker_attribution_auditor.py), dont les codes figurent aussi ci-dessous.

Rien n'est corrigé : une citation fausse est marquée invalide, l'objet est
marqué `needs_review`, et l'anomalie est consignée. Aucun contenu d'entretien
n'est écrit dans les journaux.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from agents.practice_extractor import NON_USE_STATUSES
from core import interpretation_guard

logger = logging.getLogger(__name__)

VALIDATOR_VERSION = "1.2"
# 1.1 : non_use_reason, signal métadiscursif, avertissements de locuteur, auditeur.
# 1.2 : tour enquêteur signalé par l'audit des locuteurs = appui possible (à revoir) ; référence à sa propre
#       parole cherchée dans la phrase citée ; affect « explicite » comparé par radical (interpretation_guard).

ERROR = "error"      # la preuve ou l'objet n'est pas valide
WARNING = "warning"  # vérification humaine recommandée
INFO = "info"        # information (ne dégrade pas le statut)

ISSUE_CODES = {
    "UNKNOWN_TURN_ID": (ERROR, "turn_id inexistant dans cet entretien."),
    "FOREIGN_INTERVIEW_TURN": (ERROR, "turn_id appartenant à un autre entretien."),
    "EMPTY_QUOTE": (ERROR, "Citation vide."),
    "QUOTE_NOT_FOUND": (ERROR, "Citation absente du texte du tour cité."),
    "QUOTE_NOT_EXACT": (ERROR, "Citation proche mais pas littérale (blancs, apostrophes, guillemets ou casse modifiés)."),
    "INVALID_TURN_RANGE": (ERROR, "Intervalle inversé : turn_start est postérieur à turn_end."),
    "UNKNOWN_RANGE_TURN": (ERROR, "turn_start ou turn_end inexistant dans cet entretien."),
    "NO_VALID_EVIDENCE": (ERROR, "Aucune citation valide pour cet objet."),
    "EVIDENCE_OUTSIDE_RANGE": (WARNING, "Citation située hors de l'intervalle turn_start–turn_end."),
    "EVIDENCE_TURN_NOT_LISTED": (WARNING, "Citation provenant d'un tour absent de turn_ids."),
    "NO_INTERVIEWEE_EVIDENCE": (WARNING, "Toutes les citations proviennent de tours de l'enquêteur (sans avertissement de locuteur)."),
    "CONTRADICTION_SINGLE_TURN": (WARNING, "Contradiction entre tours appuyée sur moins de deux tours distincts."),
    "AFFECT_NOT_IN_QUOTES": (WARNING, "Affect « explicite » absent des citations du signal."),
    "INTERPRETIVE_VOCABULARY": (WARNING, "Vocabulaire interprétatif dans un champ rédigé par l'agent."),
    "NON_USE_REASON_MISMATCH": (WARNING, "non_use_reason incohérent avec use_status (renseigné seulement pour non_use / refusal)."),
    "METADISCURSIVE_NO_SELF_REFERENCE": (WARNING, "Évaluation métadiscursive sans référence de l'enquêté·e à sa propre parole dans les citations."),
    "AGENT_FLAGGED_REVIEW": (INFO, "L'agent demande une vérification humaine."),
    "EXPLICITNESS_UNCLEAR": (INFO, "L'agent qualifie sa lecture d'incertaine."),
    "EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN": (INFO, "Citation d'un tour dont l'attribution du locuteur est signalée comme douteuse."),
    # Speaker Attribution Auditor (les codes de citation ci-dessus s'appliquent aussi à ses évaluations)
    "NOT_A_CANDIDATE": (WARNING, "Tour évalué par le modèle alors qu'il n'était pas candidat."),
    "CANDIDATE_NOT_ASSESSED": (WARNING, "Tour candidat non évalué par le modèle : signalé par les seules règles déterministes."),
    "DUPLICATE_ASSESSMENT": (WARNING, "Plusieurs évaluations pour le même tour."),
    "SUGGESTION_EQUALS_CURRENT": (WARNING, "Locuteur suggéré identique au locuteur actuel."),
    "EVIDENCE_OUTSIDE_EXCERPT": (WARNING, "Citation d'un tour absent de l'extrait transmis au modèle."),
    "SUGGESTION_WITHOUT_REVIEW": (INFO, "Suggestion de locuteur sans demande de vérification : vérification demandée par TRACE."),
    "CANDIDATE_LIMIT_REACHED": (WARNING, "Trop de tours candidats : les derniers n'ont pas été transmis au modèle."),
    "AUDIT_UNAVAILABLE": (WARNING, "Audit par le modèle indisponible : seuls les candidats déterministes sont signalés."),
}

_TURN_ID_RE = re.compile(r"^(?P<interview>.+)_T\d+$")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "«": '"', "»": '"',
                              "“": '"', "”": '"', "„": '"', "…": "...", " ": " ", " ": " "})


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _loose(text: str) -> str:
    """Forme relâchée, utilisée UNIQUEMENT pour diagnostiquer une citation non littérale."""
    text = _nfc(text).translate(_APOSTROPHES).casefold()
    return " ".join(text.split())


def match_quote(quote: str, text: str) -> str:
    """'exact' | 'not_exact' (retrouvée seulement en forme relâchée) | 'not_found'."""
    if _nfc(quote) in _nfc(text):
        return "exact"
    if _loose(quote) and _loose(quote) in _loose(text):
        return "not_exact"
    return "not_found"


def make_issue(code: str, **context) -> dict:
    severity, message = ISSUE_CODES[code]
    issue = {"code": code, "severity": severity, "message": message}
    issue.update({k: v for k, v in context.items() if v is not None})
    return issue


def index_turns(transcript: dict) -> dict[str, dict]:
    return {t["turn_id"]: {"position": i, "speaker": t["speaker"], "text": t["text"]}
            for i, t in enumerate(transcript["turns"])}


def turn_problem(turn_id: str, turns: dict, interview_id: str) -> str | None:
    """None si le tour existe dans CET entretien, sinon UNKNOWN_TURN_ID ou FOREIGN_INTERVIEW_TURN."""
    if turn_id in turns:
        return None
    match = _TURN_ID_RE.match(turn_id or "")
    if match and match.group("interview") != interview_id:
        return "FOREIGN_INTERVIEW_TURN"
    return "UNKNOWN_TURN_ID"


def validate_evidence(evidence: list[dict], turns: dict, interview_id: str) -> list[dict]:
    """Résultat par citation : {valid, match, code?}. N'altère pas la citation."""
    results = []
    for item in evidence:
        turn_id, quote = item.get("turn_id", ""), item.get("quote", "")
        problem = turn_problem(turn_id, turns, interview_id)
        if problem:
            results.append({"valid": False, "match": None, "code": problem})
        elif not quote.strip():
            results.append({"valid": False, "match": None, "code": "EMPTY_QUOTE"})
        else:
            match = match_quote(quote, turns[turn_id]["text"])
            code = {"exact": None, "not_exact": "QUOTE_NOT_EXACT", "not_found": "QUOTE_NOT_FOUND"}[match]
            results.append({"valid": match == "exact", "match": match, "code": code})
    return results


def _range_issues(item: dict, turns: dict, interview_id: str) -> tuple[list[dict], tuple[int, int] | None]:
    issues = []
    start, end = item.get("turn_start"), item.get("turn_end")
    for field, turn_id in (("turn_start", start), ("turn_end", end)):
        problem = turn_problem(turn_id, turns, interview_id)
        if problem:
            issues.append(make_issue("UNKNOWN_RANGE_TURN" if problem == "UNKNOWN_TURN_ID" else problem,
                                     field=field, turn_id=turn_id))
    if issues:
        return issues, None
    first, last = turns[start]["position"], turns[end]["position"]
    if first > last:
        return [make_issue("INVALID_TURN_RANGE", turn_start=start, turn_end=end)], None
    return [], (first, last)


# Référence de l'enquêté·e à sa propre parole : première personne, ou « (de) dire ça », « dit comme ça ».
_SELF_REFERENCE = re.compile(r"\b(?:je|j|moi|me|m|mon|ma|mes|dire)\b|\bdit comme ca\b")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n+")


def refers_to_own_speech(texts: list[str]) -> bool:
    """Vrai si l'un des textes contient une marque de référence à sa propre parole (accents et casse ignorés)."""
    folded = interpretation_guard.fold(" ".join(texts)).replace("'", " ")
    return bool(_SELF_REFERENCE.search(folded))


def quoted_sentences(quote: str, text: str) -> list[str]:
    """Phrase(s) du tour qui contiennent la citation (la citation seule si elle chevauche plusieurs phrases).

    Une évaluation de sa propre parole peut être citée sans la proposition qui la rattache à la parole
    (« c'est un peu facile », suivi dans la même phrase de « ce que je dis ») : la phrase entière est lue."""
    quote_n, text_n = _nfc(quote), _nfc(text)
    start = text_n.find(quote_n)
    if start < 0:
        return [quote]
    end = start + len(quote_n)
    sentences, offset = [], 0
    for part in _SENTENCE_BREAK.split(text_n):
        begin = text_n.find(part, offset)
        offset = begin + len(part)
        if begin < end and offset > start:
            sentences.append(part)
    return sentences if len(sentences) == 1 else [quote]


def validate_item(item: dict, kind: str, turns: dict, interview_id: str,
                  speaker_warnings: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Valide une pratique (kind='practice_extractor') ou un signal. Renvoie (résultats par citation, anomalies).

    `speaker_warnings` : tours dont l'attribution du locuteur est douteuse (auditeur) ; les citer
    marque l'objet à revoir (information), sans rien changer au locuteur officiel.
    """
    evidence = item.get("evidence", [])
    results = validate_evidence(evidence, turns, interview_id)
    issues = [make_issue(r["code"], evidence_index=i, turn_id=evidence[i].get("turn_id"))
              for i, r in enumerate(results) if r["code"]]
    valid_turns = [evidence[i]["turn_id"] for i, r in enumerate(results) if r["valid"]]
    if not valid_turns:
        issues.append(make_issue("NO_VALID_EVIDENCE"))
    elif all(turns[t]["speaker"] == "enqueteur" and t not in (speaker_warnings or {}) for t in valid_turns):
        # un tour enquêteur signalé par l'audit (réponse probable de l'enquêté·e) reste un appui, à revoir :
        # EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN ci-dessous
        issues.append(make_issue("NO_INTERVIEWEE_EVIDENCE"))

    for turn_id in dict.fromkeys(t for t in valid_turns if t in (speaker_warnings or {})):
        issues.append(make_issue("EVIDENCE_ON_DOUBTFUL_SPEAKER_TURN", turn_id=turn_id))

    if kind == "practice_extractor":
        has_reason = item.get("non_use_reason") is not None
        if "use_status" in item and has_reason != (item["use_status"] in NON_USE_STATUSES):
            issues.append(make_issue("NON_USE_REASON_MISMATCH", use_status=item.get("use_status"),
                                     non_use_reason=item.get("non_use_reason")))
        range_issues, span = _range_issues(item, turns, interview_id)
        issues.extend(range_issues)
        if span:
            for i, r in enumerate(results):
                if r["valid"] and not span[0] <= turns[evidence[i]["turn_id"]]["position"] <= span[1]:
                    issues.append(make_issue("EVIDENCE_OUTSIDE_RANGE", evidence_index=i, turn_id=evidence[i]["turn_id"]))
    else:
        listed = item.get("turn_ids", [])
        for turn_id in listed:
            problem = turn_problem(turn_id, turns, interview_id)
            if problem:
                issues.append(make_issue(problem, field="turn_ids", turn_id=turn_id))
        for i, r in enumerate(results):
            if r["valid"] and evidence[i]["turn_id"] not in listed:
                issues.append(make_issue("EVIDENCE_TURN_NOT_LISTED", evidence_index=i, turn_id=evidence[i]["turn_id"]))
        if item.get("signal_type") == "cross_turn_contradiction" and (
                len(set(listed)) < 2 or len(set(valid_turns)) < 2):
            issues.append(make_issue("CONTRADICTION_SINGLE_TURN"))
        if item.get("signal_type") == "metadiscursive_self_evaluation" and not refers_to_own_speech(
                [sentence for i, r in enumerate(results) if r["valid"]
                 for sentence in quoted_sentences(evidence[i]["quote"], turns[evidence[i]["turn_id"]]["text"])]):
            issues.append(make_issue("METADISCURSIVE_NO_SELF_REFERENCE"))
        affect = item.get("explicit_affect")
        if affect and not interpretation_guard.affect_in_quotes(
                affect, [evidence[i]["quote"] for i, r in enumerate(results) if r["valid"]]):
            issues.append(make_issue("AFFECT_NOT_IN_QUOTES", explicit_affect=affect))
        if item.get("needs_human_review"):
            issues.append(make_issue("AGENT_FLAGGED_REVIEW"))

    for finding in interpretation_guard.scan_item(item, kind):
        issues.append(make_issue("INTERPRETIVE_VOCABULARY", **finding))
    if item.get("explicitness") == "unclear":
        issues.append(make_issue("EXPLICITNESS_UNCLEAR"))
    return results, issues


def validate_agent_output(agent: str, items: list[dict], transcript: dict, id_letter: str,
                          speaker_warnings: dict | None = None) -> dict:
    """Valide toutes les pratiques / tous les signaux d'un agent pour UN entretien.

    Renvoie {"items": objets annotés, "report": synthèse pour evidence_validation.json}.
    Les objets annotés reçoivent un identifiant stable (<INTERVIEW>_P001 / _S001),
    `evidence[i].validation`, `needs_review` et `review_reasons` ; le texte
    produit par l'agent n'est jamais modifié.
    """
    interview_id = transcript["interview_id"]
    turns = index_turns(transcript)
    id_key = "practice_id" if agent == "practice_extractor" else "signal_id"
    annotated, all_issues = [], []
    evidence_count = valid_count = 0
    for number, item in enumerate(items, start=1):
        object_id = f"{interview_id}_{id_letter}{number:03d}"
        results, issues = validate_item(item, agent, turns, interview_id, speaker_warnings)
        evidence = [{**e, "validation": r} for e, r in zip(item.get("evidence", []), results)]
        evidence_count += len(results)
        valid_count += sum(r["valid"] for r in results)
        annotated.append({
            id_key: object_id,
            **{k: v for k, v in item.items() if k != "evidence"},
            "evidence": evidence,
            "needs_review": bool(issues),
            "review_reasons": sorted({i["code"] for i in issues}),
        })
        all_issues.extend({"object_id": object_id, **issue} for issue in issues)

    severities = [i["severity"] for i in all_issues]
    report = {
        "available": True,
        "object_count": len(items),
        "evidence_count": evidence_count,
        "valid_evidence_count": valid_count,
        "invalid_evidence_count": evidence_count - valid_count,
        "error_count": severities.count(ERROR),
        "warning_count": severities.count(WARNING),
        "objects_needing_review": [a[id_key] for a in annotated if a["needs_review"]],
        "issues": all_issues,
    }
    if report["invalid_evidence_count"]:
        logger.warning("%s/%s : %d citation(s) invalide(s) sur %d", interview_id, agent,
                       report["invalid_evidence_count"], evidence_count)
    return {"items": annotated, "report": report}


def has_problems(report: dict) -> bool:
    """Vrai si la validation relève une erreur ou un avertissement (statut SUCCESS_WITH_WARNINGS)."""
    return bool(report.get("error_count") or report.get("warning_count"))
