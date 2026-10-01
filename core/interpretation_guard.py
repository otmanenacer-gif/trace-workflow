"""Garde-fou déterministe contre le vocabulaire interprétatif dans les sorties des agents.

Les consignes des agents interdisent les jugements moraux, les diagnostics
psychologiques et les concepts théoriques (réservés à une étape ultérieure).
Ce module vérifie APRÈS coup les champs rédigés par l'agent (jamais les
citations) et signale — sans rien modifier — les termes suspects.

Un terme n'est pas signalé s'il figure dans les citations du même objet :
l'enquêté·e l'a alors employé lui-même ou elle-même.

Certains termes sont ambigus : « réparation » désigne aussi bien la réparation
d'un lave-vaisselle (sens ordinaire, matériel) qu'un concept théorique
(« réparation discursive »). Pour ces termes, chaque occurrence est examinée
dans son contexte immédiat (règles lexicales simples, sans IA) et seul un
emploi interprétatif est signalé (voir `reparation_uses`).

Les listes ne figurent volontairement PAS dans les prompts : les nommer dans
les consignes introduirait le cadre théorique que l'on veut tenir à distance.
"""

from __future__ import annotations

import re
import unicodedata

GUARD_VERSION = "1.3"
# 1.2 : « réparation » examinée en contexte (sens matériel non signalé), auditeur des locuteurs.
# 1.3 : « légitimation », « normalisation » (deux agents), « défense » (Interaction Reader) ; affect comparé par radical.

# Concepts théoriques réservés à l'étape interprétative ultérieure (les deux agents).
THEORETICAL_TERMS = {
    "accountability": r"accountab\w*",
    "garfinkel": r"garfinkel\w*",
    "breach / breaching": r"breach\w*",
    "réparation": r"repar\w*",  # terme contextuel : voir CONTEXTUAL_CHECKS
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
    "défense": r"defenses?\b",
}

# Catégories d'analyse qui qualifient l'opération de l'enquêté·e (les deux agents descriptifs, pas l'auditeur).
ANALYTIC_TERMS = {
    "légitimation": r"legitim\w*",
    "normalisation": r"normalis\w*",
}

# Lecture de la personne à partir d'une formulation, p. ex. d'une préférence énoncée (Interaction Signal Reader).
PERSON_READING_TERMS = {
    "autonomie": r"autonom\w*",
    "résistance": r"resistan\w*",
    "position morale": r"positions? morales?",
}

AGENT_TERM_SETS = {
    "practice_extractor": (THEORETICAL_TERMS, JUDGMENT_TERMS, ANALYTIC_TERMS),
    "interaction_signal_reader": (THEORETICAL_TERMS, JUDGMENT_TERMS, FUNCTION_TERMS, PERSON_READING_TERMS,
                                  ANALYTIC_TERMS),
    # Auditeur des locuteurs : sa justification (« reason ») ne doit contenir ni concept ni jugement.
    "speaker_attribution_auditor": (THEORETICAL_TERMS, JUDGMENT_TERMS),
}

# Champs rédigés par l'agent (les citations sont exclues).
AUTHORED_FIELDS = {
    "practice_extractor": (
        "summary", "academic_task", "discipline", "context", "ai_tool", "student_action_before",
        "ai_action", "student_action_after", "stated_reason", "explicit_constraints",
        "verification_or_control", "stated_frequency", "scope_qualifier", "other_actors", "uncertainty_note",
    ),
    "interaction_signal_reader": (
        "surface_form", "description", "topic", "explicit_affect", "cross_turn_reference",
    ),
    "speaker_attribution_auditor": ("reason",),
}


def fold(text: str) -> str:
    """Minuscules, sans accents, apostrophes typographiques unifiées."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("’", "'").replace("ʼ", "'")


def _compile(term_sets) -> list[tuple[str, re.Pattern]]:
    return [(label, re.compile(r"\b" + pattern)) for terms in term_sets for label, pattern in terms.items()]


_PATTERNS = {agent: _compile(sets) for agent, sets in AGENT_TERM_SETS.items()}


# --- Termes contextuels : « réparation » -----------------------------------------------------
#
# Chaque occurrence de réparation / réparer / réparateur… est classée d'après les mots qui
# l'entourent dans la même proposition (fenêtre de 4 mots avant, 7 après) :
#   1. marqueur interprétatif fort (« identité », « conduite », « discursive »…)  → interprétatif ;
#   2. objet ou cadre matériel (« lave-vaisselle », « téléphone », « coût de »…)  → ordinaire ;
#   3. nom d'analyse devant (« travail de », « opération de », « stratégie de »…) → interprétatif ;
#   4. objet matériel ailleurs dans le même objet (pratique ou signal)             → ordinaire ;
#   5. sinon : le nom (« une réparation ») reste signalé, comme avant ; le verbe non.
# Les listes sont volontairement courtes et lisibles ; elles ne figurent dans aucun prompt.

_REPAIR_FORM = re.compile(
    r"^repar(?:ations?|ateurs?|atrices?|ables?|er|e|es|ee|ees|ent|ait|aient|ais|ant|ons|ez|ions|iez"
    r"|erai|eras|era|erons|erez|eront|erait|eraient)$")
_REPAIR_NOUN = re.compile(r"^reparations?$")
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_CLAUSE_BREAK = re.compile(r"[.;,:!?|()\[\]\n«»\"]")

REPAIR_INTERPRETIVE_MARKERS = frozenset("""
    identite identites identitaire identitaires soi image conduite conduites comportement comportements
    accountability accountable morale moral moraux normatif normative normatifs symbolique symboliques
    discursive discursif discursives discursifs interactionnelle interactionnel interactionnelles
    conversationnelle conversationnel narrative narratif biographique rituelle rituel relationnelle
    relationnel reputation honneur dignite credibilite legitimite statut role offense faute transgression
    rupture breach ordre interaction interactions situation presentation recit discours propos parole
    lien liens relation relations confiance ethnomethodologique face
""".split())
REPAIR_ANALYTIC_NOUNS = frozenset("""
    travail operation operations strategie strategies procede procedes mecanisme mecanismes logique
    forme formes geste gestes tentative tentatives sequence sequences dispositif demarche effort efforts
    enjeu enjeux fonction dimension visee registre acte actes processus mouvement ressource ressources
    rhetorique pratique pratiques
""".split())
REPAIR_MATERIAL_MARKERS = frozenset("""
    appareil appareils machine machines lave-vaisselle lave-linge seche-linge frigo refrigerateur
    congelateur four micro-ondes plaque plaques cuisiniere chaudiere chauffe-eau radiateur radiateurs
    electromenager menager menagers menagere domestique domestiques ordinateur ordinateurs ordi ordis pc
    mac portable portables telephone telephones smartphone smartphones iphone tablette ecran ecrans
    clavier souris imprimante console televiseur tele tv voiture voitures auto automobile vehicule moteur
    velo velos scooter moto trottinette pneu pneus frein freins robinet evier toilettes wc plomberie
    canalisation canalisations tuyau tuyaux fuite eau gaz electricite electrique prise cable cables
    chargeur batterie meuble meubles porte fenetre volet toit toiture mur murs maison appartement
    logement objet objets outil outils outillage piece pieces panne pannes bricolage bricoler garage
    garagiste reparateur depanneur depannage technicien plombier electricien montre chaussure chaussures
    vetement vetements lunettes materiel materielle mecanique informatique cout couts frais prix devis
    facture atelier service kit tuto tutoriel tutoriels video videos notice manuel magasin boutique sav
    garantie
""".split())
_LEFT, _RIGHT = 4, 7


def _strong_marker(window: list[str]) -> bool:
    for i, token in enumerate(window):
        if token not in REPAIR_INTERPRETIVE_MARKERS:
            continue
        nxt = window[i + 1] if i + 1 < len(window) else ""
        prev = window[i - 1] if i else ""
        if token == "soi" and nxt == "meme":                      # « réparer soi même »
            continue
        if token == "face" and prev not in ("la", "sa", "leur"):  # « face à… »
            continue
        if token in ("conduite", "conduites") and {"eau", "gaz"} & set(window):  # conduite d'eau
            continue
        return True
    return False


def reparation_uses(text: str, context: str = "") -> list[str]:
    """Classe chaque occurrence de « réparation / réparer… » : 'interpretive' ou 'ordinary'.

    `text` et `context` sont attendus repliés (voir `fold`). `context` (le reste de
    l'objet : autres champs, citations) ne sert qu'à reconnaître un cadre matériel.
    """
    material_context = bool(REPAIR_MATERIAL_MARKERS & set(_TOKEN.findall(context)))
    uses = []
    for clause in _CLAUSE_BREAK.split(text):
        tokens = _TOKEN.findall(clause.replace("'", " "))
        for i, token in enumerate(tokens):
            if not _REPAIR_FORM.match(token):
                continue
            left, right = tokens[max(0, i - _LEFT):i], tokens[i + 1:i + 1 + _RIGHT]
            if _strong_marker(left + [""] + right):
                uses.append("interpretive")
            elif REPAIR_MATERIAL_MARKERS & set(left + right):
                uses.append("ordinary")
            elif REPAIR_ANALYTIC_NOUNS & set(left):
                uses.append("interpretive")
            elif material_context or not _REPAIR_NOUN.match(token):
                uses.append("ordinary")
            else:
                uses.append("interpretive")
    return uses


def reparation_is_interpretive(text: str, context: str = "") -> bool:
    return "interpretive" in reparation_uses(text, context)


# Termes ambigus : signalés seulement si l'emploi est interprétatif (le motif ne fait que présélectionner).
CONTEXTUAL_CHECKS = {"réparation": reparation_is_interpretive}


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
    """Termes interprétatifs trouvés dans les champs rédigés d'un objet (pratique, signal, évaluation).

    Renvoie une liste {term, field} ; un terme présent dans les citations de
    l'objet n'est pas signalé. Un terme contextuel (« réparation ») n'est
    signalé que dans un emploi interprétatif, et n'est exempté que si les
    citations l'emploient elles-mêmes dans ce sens.
    """
    quotes = fold(" ".join(e.get("quote", "") for e in item.get("evidence", [])))
    authored = {name: fold(text) for name, text in _authored_text(item, agent).items()}
    context = " | ".join([*authored.values(), quotes])
    findings, seen = [], set()
    for name, folded in authored.items():
        for label, pattern in _PATTERNS[agent]:
            if label in seen or not pattern.search(folded):
                continue
            check = CONTEXTUAL_CHECKS.get(label)
            if check is not None:
                if not check(folded, context) or check(quotes, quotes):
                    continue
            elif pattern.search(quotes):
                continue  # mot employé par l'enquêté·e
            seen.add(label)
            findings.append({"term": label, "field": name})
    return findings


# Mots-outils et auxiliaires ignorés quand un affect est écrit comme une expression (« j'avais peur »).
_AFFECT_FUNCTION_WORDS = frozenset("""
    pas que qui les des une est suis etait etais avais avait avoir etre ete fait tres trop plus moins bien tout
    cette avec pour dans sur mon mes ton son ses leur elle lui nous vous ils car mais donc
""".split())


def _affect_stem(word: str) -> str:
    """Radical d'un mot d'affect : 6 lettres au plus, la dernière ôtée au-delà de 4 (« scrupule(s) »,
    « stressé(e) », « énerve / énervement », « angoissé / angoisse »)."""
    return word[:6] if len(word) > 6 else (word[:-1] if len(word) > 4 else word)


def affect_in_quotes(affect: str, quotes: list[str]) -> bool:
    """Vérifie qu'un affect « explicite » est bien présent dans les citations (accents et casse ignorés).

    Chaque mot porteur (hors mots-outils) doit apparaître dans les citations, à une variation de fin de
    mot près (singulier / pluriel, genre, dérivation proche) : « scrupule » ↔ « scrupules », « j'avais peur »
    ↔ « j'ai peur ». Un affect absent des citations (« honte » pour « Euh… oui ») reste signalé.
    """
    words = [w for w in re.findall(r"\w+", fold(affect)) if len(w) >= 3 and w not in _AFFECT_FUNCTION_WORDS]
    if not words:
        return False
    haystack = fold(" ".join(quotes))
    return all(re.search(r"\b" + re.escape(_affect_stem(w)), haystack) for w in words)
