"""Garde-fou déterministe contre le vocabulaire interprétatif dans les sorties des agents.

Les consignes des agents interdisent les jugements moraux, les diagnostics
psychologiques et les concepts théoriques (réservés à une étape ultérieure).
Ce module vérifie APRÈS coup les champs rédigés par l'agent (jamais les
citations) et signale — sans rien modifier — les termes suspects.

Un terme n'est pas signalé s'il figure dans les citations du même objet :
l'enquêté·e l'a alors employé lui-même ou elle-même.

Les listes ne figurent volontairement PAS dans les prompts : les nommer dans
les consignes introduirait le cadre théorique que l'on veut tenir à distance.
"""

from __future__ import annotations

import re
import unicodedata

GUARD_VERSION = "1.0"

# Concepts théoriques réservés à l'étape interprétative ultérieure (les deux agents).
THEORETICAL_TERMS = {
    "accountability": r"accountab\w*",
    "garfinkel": r"garfinkel\w*",
    "breach / breaching": r"breach\w*",
    "réparation": r"reparation\w*",
    "ethnométhodologie": r"ethnomethodolog\w*",
    "métier d'étudiant": r"metier d.etudiant\w*",
    "régime d'…": r"regimes? d.\w*",
    "identité": r"identit\w*",
    "image de soi": r"image de soi",
}

# Jugements moraux et diagnostics psychologiques (les deux agents).
JUDGMENT_TERMS = {
    "triche": r"trich\w*",
    "fraude": r"fraud\w*",
    "malhonnêteté": r"malhonnet\w*",
    "paresse": r"paress\w*",
    "dépendance": r"dependan\w*",
    "addiction": r"addict\w*",
    "culpabilité": r"culpabil\w*",
    "honte": r"hont\w*",
    "peur": r"peur\w*",
    "anxiété": r"anxi\w*",
    "angoisse": r"angoiss\w*",
    "éviter l'effort": r"evit\w* (?:l.)?effort\w*",
}

# Attribution de fonctions ou d'intentions à une formulation (Interaction Signal Reader).
FUNCTION_TERMS = {
    "stratégie": r"strateg\w*",
    "défensif": r"defensi\w*",
    "se défendre": r"se defend\w*",
    "se justifier / justification": r"justifi\w*",
    "se protéger": r"se proteg\w*",
    "rationalisation": r"rationalis\w*",
    "hypocrisie": r"hypocri\w*",
    "mensonge": r"mensong\w*",
    "dissimulation": r"dissimul\w*",
    "gêné (rire gêné…)": r"gene(?:e|s|es)?\b",
}

AGENT_TERM_SETS = {
    "practice_extractor": (THEORETICAL_TERMS, JUDGMENT_TERMS),
    "interaction_signal_reader": (THEORETICAL_TERMS, JUDGMENT_TERMS, FUNCTION_TERMS),
}

# Champs rédigés par l'agent (les citations sont exclues).
AUTHORED_FIELDS = {
    "practice_extractor": (
        "summary", "academic_task", "discipline", "context", "ai_tool", "student_action_before",
        "ai_action", "student_action_after", "stated_reason", "explicit_constraints",
        "verification_or_control", "stated_frequency", "other_actors", "uncertainty_note",
    ),
    "interaction_signal_reader": (
        "surface_form", "description", "topic", "explicit_affect", "cross_turn_reference",
    ),
}


def fold(text: str) -> str:
    """Minuscules, sans accents, apostrophes typographiques unifiées."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("’", "'").replace("ʼ", "'")


def _compile(term_sets) -> list[tuple[str, re.Pattern]]:
    return [(label, re.compile(r"\b" + pattern)) for terms in term_sets for label, pattern in terms.items()]


_PATTERNS = {agent: _compile(sets) for agent, sets in AGENT_TERM_SETS.items()}


def _authored_text(item: dict, agent: str) -> dict[str, str]:
    texts = {}
    for name in AUTHORED_FIELDS[agent]:
        value = item.get(name)
        if isinstance(value, list):
            value = " | ".join(v for v in value if isinstance(v, str))
        if isinstance(value, str) and value:
            texts[name] = value
    return texts


def scan_item(item: dict, agent: str) -> list[dict]:
    """Termes interprétatifs trouvés dans les champs rédigés d'un objet (pratique ou signal).

    Renvoie une liste {term, field} ; un terme présent dans les citations de
    l'objet n'est pas signalé.
    """
    quotes = fold(" ".join(e.get("quote", "") for e in item.get("evidence", [])))
    findings, seen = [], set()
    for name, text in _authored_text(item, agent).items():
        folded = fold(text)
        for label, pattern in _PATTERNS[agent]:
            if label in seen or not pattern.search(folded):
                continue
            if pattern.search(quotes):
                continue  # mot employé par l'enquêté·e
            seen.add(label)
            findings.append({"term": label, "field": name})
    return findings


def affect_in_quotes(affect: str, quotes: list[str]) -> bool:
    """Vérifie qu'un affect « explicite » est bien présent dans les citations (accents et casse ignorés).

    Tolère une variation de fin de mot (singulier / pluriel, genre) : « scrupule » ↔ « scrupules ».
    """
    words = [w for w in re.findall(r"\w+", fold(affect)) if len(w) >= 3]
    if not words:
        return False
    haystack = fold(" ".join(quotes))
    return all(re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), haystack) for w in words)
