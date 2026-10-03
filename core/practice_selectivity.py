"""Sélectivité déterministe des pratiques de l'étape 3 (après la réponse du Practice Extractor, avant la validation des
preuves), sur le modèle de core/signal_selectivity.py. Aucun appel au modèle, aucune citation modifiée.

1. Normalisation FORMELLE de `non_use_reason` (schéma : renseigné pour non_use / refusal seulement) :
   - statut use / past_use / hypothetical avec `not_stated` (valeur parasite qui signifie « aucune raison ») → null ;
   - statut non_use / refusal sans raison (null) → `not_stated`.
   Une autre raison (preference, personal_rule…) n'est jamais modifiée : l'incohérence reste signalée par le
   validateur (NON_USE_REASON_MISMATCH).
2. Pertinence : une pratique doit avoir un lien EXPLICITE avec une IAG (usage, non-usage / refus / limite d'usage,
   vérification d'une sortie, délégation, interaction avec l'outil), démontrable à partir des champs déjà produits ou
   du texte cité. Le lien est établi si l'un des éléments suivants est présent :
   - statut non_use ou refusal (par définition, un non-usage ou un refus d'une IAG) ;
   - un champ défini par rapport à l'outil est renseigné : outil nommé (`ai_tool`), ce que l'outil produit
     (`ai_action`), ce que l'enquêté·e fait du résultat (`student_action_after`), contrôles sur le résultat
     (`verification_or_control`) ;
   - la phrase citée nomme une IAG (IA, ChatGPT, Copilot…, ou l'outil déclaré) ou s'adresse à l'outil (« je lui
     demande », « je lui fais traduire », « je l'utilise », « je m'en sers », « il me donne »…), ou, première phrase
     du tour, elle répond à une question de l'enquêteur qui le fait, ou elle reprend explicitement ce qui précède (« aussi », « pareil »,
     « de même », « sauf », « à part »…) : le reste du même tour ou la réponse précédente du même locuteur, liés à
     une IAG. La phrase citée, et non le tour entier : une
     routine citée dans une autre phrase d'un tour qui parle de l'outil n'y est pas rattachée.
   Sinon, la pratique est une routine sans lien démontré avec une IAG (NO_AI_LINK) : elle reste visible dans
   `set_aside_practices` (avec sa raison et la vérification de ses citations) et n'est transmise ni à la validation ni
   aux étapes suivantes. Une pratique dont une citation est invalide n'est jamais écartée : le validateur la signale.
"""

from __future__ import annotations

import re

from agents.practice_extractor import NON_USE_STATUSES
from core import evidence_validator, interpretation_guard
from core.schemas import SPEAKER_INTERVIEWER

PRACTICE_SELECTIVITY_VERSION = "1.0"
# Champs définis par rapport à l'outil (prompts/practice_extractor.md) : l'outil nommé, ce qu'il produit, ce que
# l'enquêté·e fait ENSUITE du résultat, les contrôles effectués SUR le résultat.
AI_FIELDS = ("ai_tool", "ai_action", "student_action_after", "verification_or_control")

REASON_NO_AI_LINK = "NO_AI_LINK"
SET_ASIDE_REASONS = {
    REASON_NO_AI_LINK: "Aucun lien explicite avec une IAG : ni non-usage / refus, ni champ défini par rapport à l'outil "
                       "(outil, action de l'outil, suite donnée au résultat, contrôle du résultat), ni tour cité qui "
                       "nomme une IAG ou s'adresse à l'outil (routine sans lien démontré).",
}

# Désignations d'une IAG dans le texte replié (minuscules, sans accents, apostrophes remplacées par des espaces).
_AI_TERMS = re.compile(
    r"\b(?:ia|iag|i a|ias|chat ?gpt|gpt\w*|openai|copilot|gemini|bard|mistral|perplexity|deepseek|deepl|quillbot|"
    r"grammarly|chatbots?|midjourney|dall e|llm|intelligences? artificielles?)\b")
# Formulations adressées à l'outil (texte replié) : « je lui demande », « je l'utilise », « je m'en sers »…
_ADDRESSED = re.compile(
    r"\b(?:lui (?:\w+ ){0,3}?(?:demand|fai[st]|fait|pos|donn|envoi|envoy|coll|soum|di[st]|ecri|tradu|"
    r"corrig|redig|reformul|resum|cherch|genere)\w*|l (?:\w+ ){0,2}?utilis\w*|m en (?:sers|servais|servir|suis servi\w*)|"
    r"me sers d\w*|(?:il|elle|ca) m(?:e|) (?:donn|propos|sort|gener|fai|ecri|corrig|repon|redig|resum|conseill|aid|"
    r"expliqu|sugger|recommand|montr|indiqu|tradu|reformul)\w*)\b")


def _fold(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", interpretation_guard.fold(text or "")).split())


def names_ai(text: str, tools: list[str] | None = None) -> bool:
    """Vrai si le texte nomme une IAG (terme générique) ou l'un des outils déclarés."""
    folded = f" {_fold(text)} "
    if _AI_TERMS.search(folded):
        return True
    return any(len(name) >= 3 and f" {name} " in folded for name in (_fold(t) for t in tools or []))


def normalize_non_use_reason(practice: dict) -> tuple[dict, dict | None]:
    """Normalisation formelle de non_use_reason. → (pratique, normalisation appliquée ou None)."""
    status, reason = practice.get("use_status"), practice.get("non_use_reason")
    if status in NON_USE_STATUSES and reason is None:
        return {**practice, "non_use_reason": "not_stated"}, {"use_status": status, "from": None, "to": "not_stated"}
    if status not in NON_USE_STATUSES and reason == "not_stated":
        return {**practice, "non_use_reason": None}, {"use_status": status, "from": "not_stated", "to": None}
    return practice, None


def addresses_ai(text: str) -> bool:
    """Vrai si le texte s'adresse à un outil (« je lui demande », « je l'utilise », « il me donne »…)."""
    return bool(_ADDRESSED.search(f" {_fold(text)} "))


# Reprise ou exception qui renvoie à ce qui précède : « aussi », « pareil », « de même », « sauf », « à part »…
_CONTINUATION = re.compile(r"\b(?:aussi|pareil|pareille|de meme|meme chose|egalement|idem|sauf|a part|excepte|"
                           r"sinon)\b")


def turn_linked_to_ai(position: int, ordered: list[dict], tools: list[str] | None = None, depth: int = 0,
                      text: str | None = None) -> bool:
    """Le passage (`text`, par défaut le tour entier) nomme une IAG ou s'adresse à l'outil ; ou, première phrase du
    tour, il répond à une question de l'enquêteur qui le fait ; ou il prolonge EXPLICITEMENT (« aussi », « pareil », « de même »…) la
    réponse précédente du même locuteur, elle-même liée à une IAG (au plus 3 reprises)."""
    turn = ordered[position]
    text = turn["text"] if text is None else text
    if names_ai(text, tools) or addresses_ai(text):
        return True
    previous = ordered[position - 1] if position > 0 else None
    direct_answer = turn["text"].lstrip().startswith(text.lstrip()[:40])  # première phrase : réponse à la question
    if previous and direct_answer and previous["speaker"] == SPEAKER_INTERVIEWER and (
            names_ai(previous["text"], tools) or addresses_ai(previous["text"])):
        return True
    if depth < 3 and _CONTINUATION.search(f" {_fold(text)} "):
        if text != turn["text"] and (names_ai(turn["text"], tools) or addresses_ai(turn["text"])):
            return True  # la reprise renvoie au reste du même tour, qui nomme l'outil ou s'adresse à lui
        earlier = next((i for i in (position - 1, position - 2)
                        if i >= 0 and ordered[i]["speaker"] == turn["speaker"]), None)
        return earlier is not None and turn_linked_to_ai(earlier, ordered, tools, depth + 1)
    return False


def relevance_problem(practice: dict, turns: dict, ordered: list[dict]) -> str | None:
    """Raison d'écarter la pratique, ou None si son lien avec une IAG est établi. Le passage examiné est la PHRASE
    citée (une routine citée dans une autre phrase d'un tour qui parle de l'outil n'y est pas rattachée)."""
    if practice.get("use_status") in NON_USE_STATUSES or any(practice.get(field) for field in AI_FIELDS):
        return None
    for evidence in practice.get("evidence", []):
        turn = turns.get(evidence.get("turn_id"))
        if turn is None:
            continue
        # la phrase du tour qui contient la citation (comme le validateur pour la référence à sa propre parole)
        sentence = " ".join(evidence_validator.quoted_sentences(evidence.get("quote") or "", turn["text"]))
        if turn_linked_to_ai(turn["position"], ordered, practice.get("ai_tool"), text=sentence):
            return None
    return REASON_NO_AI_LINK


def apply(practices: list[dict], transcript: dict) -> dict:
    """Normalisation puis sélectivité. → {"practices", "set_aside", "summary"} (ordre conservé)."""
    turns = evidence_validator.index_turns(transcript)
    ordered = transcript["turns"]
    interview_id = transcript["interview_id"]
    kept, set_aside, normalized = [], [], []
    for index, practice in enumerate(practices):
        practice, change = normalize_non_use_reason(practice)
        if change:
            normalized.append({"index_received": index, **change})
        results = evidence_validator.validate_evidence(practice.get("evidence", []), turns, interview_id)
        reason = relevance_problem(practice, turns, ordered) if results and all(r["valid"] for r in results) else None
        if reason is None:
            kept.append(practice)
        else:
            set_aside.append({**practice, "evidence": [{**e, "validation": v} for e, v in
                                                       zip(practice.get("evidence", []), results)],
                              "set_aside_reason": reason})
    by_reason = {code: sum(p["set_aside_reason"] == code for p in set_aside) for code in SET_ASIDE_REASONS
                 if any(p["set_aside_reason"] == code for p in set_aside)}
    return {"practices": kept, "set_aside": set_aside, "summary": {
        "practice_selectivity_version": PRACTICE_SELECTIVITY_VERSION, "practices_received": len(practices),
        "practices_kept": len(kept), "practices_set_aside": len(set_aside), "set_aside_by_reason": by_reason,
        "non_use_reason_normalized": normalized}}
