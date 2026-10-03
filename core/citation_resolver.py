"""CitationResolver : correction DÉTERMINISTE et conservatrice des citations de l'étape 3, sans modèle.

Pour une citation invalide (tour inconnu, citation absente ou non littérale), seule une correspondance TEXTUELLE est
cherchée, dans les tours du matériau envoyé à l'agent (bloc ou sélection) :

1. le tour annoncé : la citation y figure-t-elle à la forme près (espaces, apostrophes et guillemets typographiques,
   casse, ponctuation) ? → la citation est remplacée par le passage littéral du tour ;
2. sinon, les autres tours autorisés : la citation (au moins MIN_WORDS_ELSEWHERE mots) figure-t-elle, à la forme près,
   dans UN SEUL tour ? → turn_id de ce tour et passage littéral ;
3. sinon : aucune correction. Plusieurs tours possibles (ambiguë), aucun tour (introuvable), citation vide ou trop
   courte : la citation reste telle quelle et le validateur habituel la signale.

Jamais de paraphrase, de reconstruction sémantique, de citation partielle ou recousue, ni de tour hors du matériau
autorisé. Seuls les MOTS comptent, dans le même ordre, sans aucun mot ajouté ou retiré : la normalisation ne sert qu'à
la recherche, et la citation écrite est toujours un extrait exact du transcript. Chaque correction est journalisée.
"""

from __future__ import annotations

import unicodedata

RESOLVER_VERSION = "1.0"
MIN_WORDS_ELSEWHERE = 4      # citation cherchée hors du tour annoncé : au moins 4 mots (pas de coïncidence courte)
MIN_WORDS_ANNOUNCED = 1

FIXED_IN_TURN = "fixed_in_announced_turn"     # passage littéral restauré dans le tour annoncé
FIXED_OTHER_TURN = "fixed_in_other_turn"      # citation trouvée dans un seul autre tour autorisé
AMBIGUOUS = "ambiguous"                       # plusieurs tours (ou passages) possibles : aucun choix arbitraire
NOT_FOUND = "not_found"                       # aucune correspondance textuelle dans le matériau autorisé
TOO_SHORT = "too_short"                       # citation vide ou trop courte pour une recherche sûre

_TYPOGRAPHIC = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "«": '"', "»": '"', "“": '"',
                              "”": '"', "„": '"', " ": " ", " ": " ", " ": " "})


def _normalize(text: str) -> tuple[str, list[int]]:
    """Forme de RECHERCHE : NFC, casse ignorée, apostrophes / guillemets / espaces unifiés, ponctuation retirée ; les
    mots sont séparés par une espace et la chaîne est bordée d'espaces. Renvoie aussi, pour chaque caractère de la
    forme normalisée, sa position dans le texte d'origine (pour extraire le passage littéral)."""
    chars, positions = [" "], [-1]
    for index, char in enumerate(text):
        for folded in char.translate(_TYPOGRAPHIC).casefold():
            if folded.isalnum():
                chars.append(folded)
                positions.append(index)
            elif chars[-1] != " ":
                chars.append(" ")
                positions.append(index)
    if chars[-1] != " ":
        chars.append(" ")
        positions.append(len(text))
    return "".join(chars), positions


def _edge_punctuation(quote: str) -> tuple[str, str]:
    """Ponctuation (hors lettres et chiffres) en tête et en fin de citation, en forme typographique unifiée."""
    stripped = quote.strip().translate(_TYPOGRAPHIC)
    lead = ""
    for char in stripped:
        if char.isalnum():
            break
        lead += char
    trail = ""
    for char in reversed(stripped):
        if char.isalnum():
            break
        trail = char + trail
    return lead, trail


def _spans(needle: str, text: str, quote: str = "") -> list[str]:
    """Passages LITTÉRAUX du tour qui correspondent à la citation (à la forme près), sans doublon. La ponctuation
    que la citation porte à ses bords est incluse si le tour la porte au même endroit (« rapide. »)."""
    norm_text, positions = _normalize(text)
    lead, trail = _edge_punctuation(quote)
    found, start = [], norm_text.find(needle)
    while start >= 0:
        first = positions[start + 1]                   # premier caractère du premier mot
        last = positions[start + len(needle) - 2]      # dernier caractère du dernier mot
        while last + 1 < len(text) and not text[last + 1].isalnum() and not text[last + 1].isspace() \
                and text[last + 1].translate(_TYPOGRAPHIC) in trail:
            last += 1
        while first > 0 and not text[first - 1].isalnum() and not text[first - 1].isspace() \
                and text[first - 1].translate(_TYPOGRAPHIC) in lead:
            first -= 1
        found.append(text[first:last + 1])
        start = norm_text.find(needle, start + 1)
    return list(dict.fromkeys(found))


def resolve(quote: str, announced_turn_id: str | None, turns: list[dict], allowed: set[str]) -> dict:
    """Décision pour UNE citation invalide. `turns` : tours de l'entretien (turn_id, text) ; `allowed` : turn_id du
    matériau envoyé à l'agent. → {"decision", "turn_id"?, "quote"?, "candidates"?}."""
    text_of = {t["turn_id"]: unicodedata.normalize("NFC", t["text"]) for t in turns}
    needle, _ = _normalize(unicodedata.normalize("NFC", quote or ""))
    words = len(needle.split())
    if words < MIN_WORDS_ANNOUNCED:
        return {"decision": TOO_SHORT}
    if announced_turn_id in allowed and announced_turn_id in text_of:
        spans = _spans(needle, text_of[announced_turn_id], quote)
        if len(spans) == 1:
            return {"decision": FIXED_IN_TURN, "turn_id": announced_turn_id, "quote": spans[0]}
        if len(spans) > 1:
            return {"decision": AMBIGUOUS, "candidates": [announced_turn_id]}
    if words < MIN_WORDS_ELSEWHERE:
        return {"decision": TOO_SHORT}
    matches = {}
    for turn in turns:
        turn_id = turn["turn_id"]
        if turn_id in allowed and turn_id != announced_turn_id:
            spans = _spans(needle, text_of[turn_id], quote)
            if spans:
                matches[turn_id] = spans
    if not matches:
        return {"decision": NOT_FOUND}
    if len(matches) > 1 or len(next(iter(matches.values()))) > 1:
        return {"decision": AMBIGUOUS, "candidates": sorted(matches)}
    [(turn_id, [span])] = matches.items()
    return {"decision": FIXED_OTHER_TURN, "turn_id": turn_id, "quote": span}


def resolve_item(item: dict, invalid_indices: list[int], turns: list[dict], allowed: set[str],
                 signal: bool) -> tuple[dict, list[dict]]:
    """Corrige les citations invalides d'un objet quand une correspondance sûre existe. → (objet, journal).

    Signal : un turn_id INEXISTANT remplacé dans une citation l'est aussi dans `turn_ids` (même tour) ; un tour
    ajouté est complété par la mise en forme habituelle (signal_selectivity.complete_turn_ids)."""
    item = {**item, "evidence": [dict(e) for e in item.get("evidence") or []]}
    known = {t["turn_id"] for t in turns}
    log = []
    for index in invalid_indices:
        evidence = item["evidence"][index]
        before = {"turn_id": evidence.get("turn_id"), "quote": evidence.get("quote")}
        result = resolve(evidence.get("quote") or "", evidence.get("turn_id"), turns, allowed)
        entry = {"evidence_index": index, "decision": result["decision"], "before": before}
        if result["decision"] in (FIXED_IN_TURN, FIXED_OTHER_TURN):
            evidence.update(turn_id=result["turn_id"], quote=result["quote"])
            entry["after"] = {"turn_id": result["turn_id"], "quote": result["quote"]}
            old = before["turn_id"]
            if signal and old != result["turn_id"] and old not in known and old in (item.get("turn_ids") or []):
                item["turn_ids"] = list(dict.fromkeys(result["turn_id"] if t == old else t for t in item["turn_ids"]))
        elif result.get("candidates"):
            entry["candidates"] = result["candidates"]
        log.append(entry)
    return item, log
