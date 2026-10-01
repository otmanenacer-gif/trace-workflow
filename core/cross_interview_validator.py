"""Étape 6 — validation DÉTERMINISTE de la comparaison inter-entretiens (aucun LLM).

Pour chaque affirmation inter-entretiens (`cross_case_claims`, identifiants entiers rétablis) :
- appuis : chaque interview_id existe dans le corpus importé et exploitable (un entretien exclu n'appuie rien) ;
  chaque claim_id / criterion_id existe, est utilisable à l'étape 5 et appartient à CET entretien ; un appui sans
  aucun identifiant est un appui inventé ; chaque tour cité existe dans l'entretien (sinon erreur) et figure
  parmi les tours des éléments cités (sinon avertissement) ;
- comptes : `n_supporting_interviews` RECALCULÉ sur les interview_id distincts (plusieurs éléments ou entrées
  d'un même entretien ne gonflent jamais le compte) ; une valeur différente du modèle est signalée et corrigée ;
- contre-exemples : mêmes contrôles ; un contre-exemple sans élément de l'étape 5 ne fait pas un refus explicite :
  l'entretien reste « non observé » (le contre-exemple est conservé, à revoir) ;
- positions : pour chaque entretien exploitable, `explicit_presence`, `explicit_refusal`, `contrary_case`,
  `divergent_variant` ou `not_observed` — l'absence d'une mention n'est jamais une absence du critère ;
- types : une régularité (`recurring_*`, `*_pattern`, `contextual_association`) sur un seul entretien est
  REQUALIFIÉE en `minority_configuration` ; un `temporal_pattern` exige des `explicit_temporal_change` validés
  par l'étape 5 (sinon requalifié en `contextual_association`) ; un pattern de critère doit citer des critères ;
- éléments à revoir : propagation des review_reasons de l'étape 5 (par entrée et pour l'affirmation) ; une
  affirmation appuyée UNIQUEMENT sur des éléments à revoir est marquée à revoir et sa confiance limitée à « low » ;
  une régularité dont moins de deux entretiens ont un appui propre est signalée et limitée de même ;
- confiance : limitée à « medium » en mode exploratoire (2 entretiens), à « low » sur un seul entretien ;
- texte : citation « » absente du matériau de l'étape 5 des entretiens cités (erreur), typologie de personnes et
  attribution psychologique / sociale non documentée (erreur), vocabulaire psychologisant, causalité, généralisation
  au-delà du N observé, comptes d'occurrences, comptes chiffrés incohérents, non-observation présentée comme
  absence, entretien mentionné sans appui (avertissements).

Cas négatifs : chaque contre-exemple documenté et chaque affirmation `negative_case` est conservé dans
`negative_cases`, même si l'affirmation qui le porte est rejetée pour une autre raison ; seuls les identifiants
inventés n'y entrent pas. Rien n'est réécrit : TRACE ajoute statut, positions, comptes et raisons ; une
requalification garde `model_claim_type`, une confiance limitée garde `model_confidence`.
"""

from __future__ import annotations

import re

from agents.cross_interview_comparator import COUNTER_RELATIONS, CROSS_CLAIM_TYPES
from core import interpretation_guard
from core import trajectory_validator as stage5_validator

VALIDATOR_VERSION = "1.0"
ERROR, WARNING, INFO = "error", "warning", "info"

ISSUE_CODES = {
    # appuis
    "UNKNOWN_INTERVIEW": (ERROR, "Entretien absent du corpus importé."),
    "EXCLUDED_INTERVIEW": (ERROR, "Entretien exclu du corpus (import refusé) : il n'appuie rien."),
    "UNKNOWN_OBJECT_ID": (ERROR, "claim_id / criterion_id inexistant dans les sorties de l'étape 5 de cet entretien."),
    "FOREIGN_OBJECT_ID": (ERROR, "claim_id / criterion_id appartenant à un autre entretien que celui de l'appui."),
    "STAGE5_REJECTED_OBJECT": (ERROR, "Élément rejeté par l'étape 5 (usable_for_next_stages = false) : il "
                                      "n'appuie rien."),
    "EMPTY_SUPPORT_ENTRY": (ERROR, "Appui sans aucun élément de l'étape 5 : appui inventé."),
    "NO_VALID_SUPPORT": (ERROR, "Aucun entretien n'appuie valablement cette affirmation."),
    "UNKNOWN_EVIDENCE_TURN": (ERROR, "Tour cité inexistant dans le matériau de l'étape 5 de cet entretien."),
    "EVIDENCE_TURN_OUTSIDE_SUPPORT": (WARNING, "Tour cité absent des éléments de l'étape 5 invoqués."),
    "DUPLICATE_SUPPORT_INTERVIEW": (INFO, "Plusieurs entrées pour un même entretien : fusionnées, comptées une fois."),
    "COUNT_MISMATCH": (WARNING, "n_supporting_interviews du modèle différent du nombre d'entretiens distincts : "
                                "corrigé."),
    # contre-exemples
    "COUNTEREXAMPLE_UNKNOWN": (ERROR, "Contre-exemple : entretien ou identifiant inexistant (cas inventé)."),
    "COUNTEREXAMPLE_WITHOUT_SUPPORT": (WARNING, "Contre-exemple sans élément de l'étape 5 : l'entretien reste « non "
                                                "observé » (une non-mention n'est pas un refus)."),
    "SUPPORT_AND_COUNTEREXAMPLE": (WARNING, "Un même entretien appuie l'affirmation et la contredit."),
    # types
    "RECURRENCE_SINGLE_INTERVIEW": (WARNING, "Régularité appuyée sur un seul entretien : requalifiée en configuration "
                                             "minoritaire."),
    "CONTRAST_SINGLE_INTERVIEW": (WARNING, "Contraste ou divergence documenté dans un seul entretien."),
    "CRITERION_PATTERN_WITHOUT_CRITERIA": (WARNING, "Pattern de critère du métier d'étudiant sans critère validé de "
                                                    "l'étape 5 (RC…) pour au moins un entretien."),
    "TEMPORAL_SUPPORT_NOT_VALIDATED": (WARNING, "Appui d'un pattern temporel sans changement temporel validé par "
                                                "l'étape 5."),
    "TEMPORAL_PATTERN_REQUALIFIED": (WARNING, "Pattern temporel sans aucun changement temporel validé : requalifié en "
                                              "association contextuelle."),
    "NEGATIVE_CASE_UNRELATED": (WARNING, "Cas négatif sans régularité désignée (related_claim_numbers)."),
    # éléments à revoir
    "SUPPORTED_ONLY_BY_REVIEW_ITEMS": (INFO, "Affirmation appuyée uniquement sur des éléments à revoir : marquée à "
                                             "revoir, confiance limitée."),
    "STRONG_PATTERN_ON_REVIEW_ITEMS": (WARNING, "Confiance « high » alors que tous les appuis sont à revoir."),
    "RECURRENCE_RESTS_ON_REVIEW_ITEMS": (WARNING, "Régularité dont moins de deux entretiens ont un appui sans "
                                                  "élément à revoir : elle ne peut pas être forte."),
    "REVIEW_ITEMS_PROPAGATED": (INFO, "Certains appuis sont des éléments à revoir (review_reasons propagées)."),
    "MODEL_FLAGGED_REVIEW": (INFO, "Le modèle demande une vérification humaine."),
    "EXPLORATORY_CONFIDENCE_CAPPED": (INFO, "Corpus exploratoire (2 entretiens) : confiance limitée à « medium »."),
    "SINGLE_INTERVIEW_CONFIDENCE_CAPPED": (INFO, "Un seul entretien d'appui : confiance limitée à « low »."),
    # texte
    "QUOTE_NOT_IN_STAGE5": (ERROR, "Citation « » absente du matériau de l'étape 5 des entretiens cités."),
    "TYPOLOGY_OF_PERSONS": (ERROR, "Typologie ou catégorisation de personnes (« type d'étudiant », « profil »…)."),
    "PERSON_OR_SOCIAL_ATTRIBUTION": (ERROR, "Explication par la personnalité, la morale, l'intelligence, la "
                                            "paresse, le milieu social, le genre ou l'origine, absente des données."),
    "PSYCHOLOGICAL_VOCABULARY": (WARNING, "Vocabulaire psychologisant, d'intention ou de récit de conversion."),
    "CAUSAL_CLAIM": (WARNING, "Formulation causale (« s'explique par », « est dû à »…) : décrire une association."),
    "GENERALIZATION_BEYOND_N": (WARNING, "Généralisation au-delà des entretiens observés (« en général », « la "
                                         "plupart des étudiants »…)."),
    "OCCURRENCE_COUNT": (WARNING, "Compte d'occurrences ou d'épisodes : les comptes portent sur des entretiens."),
    "COUNT_IN_TEXT_MISMATCH": (WARNING, "Compte d'entretiens écrit dans le texte incohérent avec les appuis."),
    "NOT_OBSERVED_AS_ABSENCE": (WARNING, "Non-observation présentée comme absence ou désintérêt."),
    "INTERVIEW_MENTION_NOT_SUPPORTED": (WARNING, "Entretien nommé dans le texte sans figurer dans les appuis ni les "
                                                 "contre-exemples."),
    # document
    "SUMMARY_VOCABULARY": (WARNING, "Synthèse : typologie, attribution aux personnes ou vocabulaire psychologisant."),
    "SUMMARY_CAUSAL": (WARNING, "Synthèse : formulation causale."),
    "SUMMARY_GENERALIZATION": (WARNING, "Synthèse : généralisation au-delà du N observé ou comptes d'occurrences."),
    "SUMMARY_QUOTE_NOT_IN_STAGE5": (WARNING, "Synthèse : citation « » absente du matériau de l'étape 5."),
    "SUMMARY_UNKNOWN_INTERVIEW": (WARNING, "Synthèse : entretien nommé absent du corpus exploitable."),
    "PAYLOAD_OVER_THRESHOLD": (WARNING, "Représentation envoyée au-delà du seuil d'un appel unique."),
    "EXPLORATORY_CORPUS": (INFO, "Deux entretiens exploitables seulement : comparaison exploratoire."),
}

REGULARITY_TYPES = frozenset({"recurring_boundary", "recurring_accounting_move", "recurring_student_role_criterion",
                              "ordinary_zone_pattern", "exception_pattern", "contextual_association",
                              "temporal_pattern"})
CONTRAST_TYPES = frozenset({"divergent_boundary", "divergent_accounting_move", "divergent_student_role_criterion",
                            "unresolved_cross_case_contrast"})
CRITERION_TYPES = frozenset({"recurring_student_role_criterion", "divergent_student_role_criterion"})
LIST_KEYS = {
    "recurring_boundary": "recurring_boundaries", "divergent_boundary": "divergent_boundaries",
    "recurring_accounting_move": "recurring_accounting_moves",
    "divergent_accounting_move": "divergent_accounting_moves",
    "recurring_student_role_criterion": "recurring_student_role_criteria",
    "divergent_student_role_criterion": "divergent_student_role_criteria",
    "ordinary_zone_pattern": "ordinary_zone_patterns", "exception_pattern": "exception_patterns",
    "contextual_association": "contextual_associations", "temporal_pattern": "temporal_patterns",
    "unresolved_cross_case_contrast": "unresolved_cross_case_contrasts",
    "minority_configuration": "minority_configurations", "negative_case": "negative_case_claims",
}
POSITIONS = ("explicit_presence", "explicit_refusal", "contrary_case", "divergent_variant", "not_observed")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

# --- Vocabulaire (texte replié : minuscules, sans accents) ---------------------------------------------------
_PSYCH = [(label, re.compile(r"\b(?:" + p + ")")) for label, p in stage5_validator.FORBIDDEN_TERMS.items()]
_TYPOLOGY = re.compile(
    r"\b(?:types? d.(?:etudiant|usager|utilisateur)\w*|profils? (?:d.etudiant|d.usager|type|psycholog)\w*|"
    r"categories? d.etudiant\w*|typologi\w*|famille d.etudiant\w*|etudiant\w* (?:de|du) type\b|"
    r"(?:les |des )?(?:bons|mauvais|vrais) etudiant\w*|etudiant\w* (?:consciencieu|paresseu|serieu|tricheu|vertueu|"
    r"honnete|malhonnete|responsable|irresponsable)\w*)")
_PERSON_SOCIAL = re.compile(
    r"\b(?:personnalit\w*|tempérament|temperament\w*|caractere (?:de|des|du) |paress\w*|"
    r"intelligen(?:t|te|ts|tes)\b|(?:son|leur|leurs|sa) intelligence\b|moralit\w*|immoral\w*|"
    r"(?:sens|valeurs?) mora\w*|milieu (?:social|familial|d.origine)|origine (?:sociale|etrangere|ethnique|"
    r"geographique|culturelle)|classes? (?:sociale|populaire|moyenne|superieure)s?|capital (?:culturel|economique)|"
    r"(?:selon|par|du|de leur|leur) (?:genre|sexe)\b|les (?:filles|garcons)\b|boursier\w*|socio-?economique\w*|"
    r"csp\b)")
_CAUSAL = re.compile(
    r"\b(?:s.expliqu\w* par|expliqu\w* (?:la|les|ces|cette|cet) (?:difference|variation|ecart|divergence|contraste)\w*|"
    r"(?:est|sont) (?:du|due|dus|dues) (?:a|au|aux)\b|a cause (?:de|du|des)\b|en raison (?:de|du|des)\b|"
    r"(?:l.)?effet (?:de la|du|des) (?:discipline|filiere|matiere|personnalite|genre|milieu)|"
    r"(?:la|leur|sa) (?:discipline|filiere|matiere) (?:determine|explique|conduit|pousse|entraine|cause|produit)\w*|"
    r"(?:determine|cause|provoque|entraine)(?:nt|e|es|s)? (?:par|chez)\b|consequence (?:de|du|des)\b|"
    r"parce qu.(?:ils|elles) sont|du fait de leur)")
_GENERALIZATION = re.compile(
    r"\b(?:en general\b|generalement|la plupart des|la majorite des|majoritairement|tous les etudiant\w*|"
    r"toutes les etudiant\w*|chez les etudiant\w*|typiquement|universel\w*|"
    r"les etudiant\w* (?:\w+ ){0,3}(?:font|utilisent|"
    r"considerent|pensent|estiment|ont|sont|deleguent|associent|refusent|preferent|ne)\b|"
    r"les etudiant\w* en general|on observe que les etudiant\w*)")
_OCCURRENCE = re.compile(r"\b\d+ (?:fois|occurrences?|episodes?|mentions?|affirmations?|passages?)\b")
_ABSENCE = re.compile(
    r"\b(?:ne possede\w* pas|n.(?:a|ont) pas (?:ce|de|le) critere|absence (?:du|de ce|de|d.un) critere|depourvu\w*|"
    r"ignor(?:e|ent) (?:ce|le) critere|sans (?:ce|aucun) critere|indifferen\w*|n.accord\w* (?:aucune|pas d.) "
    r"importance|ne (?:se )?soucie\w* pas|desinteress\w*)")
_QUOTES = re.compile(r"«\s*(.+?)\s*»|“(.+?)”")
_NUMBER_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
                 "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
                 "seize": 16, "dix-sept": 17, "dix-huit": 18, "dix-neuf": 19, "vingt": 20}
_NUM = r"(\d+|" + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True)) + r")"
_RATIO = re.compile(r"\b" + _NUM + r" (?:des |sur (?:les )?)" + _NUM + r" entretiens\b")
_COUNT = re.compile(r"\b" + _NUM + r" (?:seul )?entretiens?\b")
_ALIAS = re.compile(r"\bI\d{2}\b")
_NOT_OBSERVED_WORDS = re.compile(r"\b(?:non observ\w*|pas observ\w*|sans mention|"
                                 r"ne (?:le |la |les )?(?:mentionn|evoqu|formul|abord)\w*|"
                                 r"n.(?:en )?(?:mentionn|evoqu|abord)\w*)")


def make_issue(code: str, **context) -> dict:
    severity, message = ISSUE_CODES[code]
    issue = {"code": code, "severity": severity, "message": message}
    issue.update({k: v for k, v in context.items() if v is not None})
    return issue


def _number(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


def _normalize(text: str) -> str:
    return " ".join(interpretation_guard.fold(text or "").replace(" ", " ").replace("\xa0", " ").split())


def _min_confidence(value: str | None, cap: str) -> str | None:
    if value not in CONFIDENCE_ORDER:
        return value
    return cap if CONFIDENCE_ORDER[value] > CONFIDENCE_ORDER[cap] else value


class _Context:
    def __init__(self, material: dict, excluded_ids: set[str]):
        self.material = material
        self.usable = set(material["interview_ids"])
        self.excluded = excluded_ids
        self.aliases = material["aliases"]
        self.turns = {iid: set(t) for iid, t in material["turns_by_interview"].items()}
        self.texts = {iid: _normalize(t) for iid, t in material["texts_by_interview"].items()}
        self.rejected = {o for info in material["interviews"].values() for o in info["rejected_object_ids"]}
        self.mentions = {**{iid: iid for iid in [*self.usable, *excluded_ids]},
                         **{alias: iid for iid, alias in self.aliases.items()}}

    def obj(self, object_id: str) -> dict | None:
        return self.material["claims"].get(object_id) or self.material["criteria"].get(object_id)

    def resolve(self, entry: dict, issues: list[dict], *, counter: bool) -> dict | None:
        """Entrée d'appui ou de contre-exemple → objets de l'étape 5 vérifiés. None si l'entretien est inconnu."""
        iid = entry.get("interview_id")
        unknown = "COUNTEREXAMPLE_UNKNOWN" if counter else None
        if iid not in self.usable:
            code = unknown or ("EXCLUDED_INTERVIEW" if iid in self.excluded else "UNKNOWN_INTERVIEW")
            issues.append(make_issue(code, interview_id=iid))
            return None
        objects, bad = [], False
        for key in ("claim_ids", "criterion_ids"):
            for object_id in dict.fromkeys(entry.get(key) or []):
                obj = self.obj(object_id)
                if obj is not None and obj["interview_id"] == iid:
                    objects.append(obj)
                    continue
                bad = True
                if obj is not None:
                    code = unknown or "FOREIGN_OBJECT_ID"
                elif object_id in self.rejected:
                    code = unknown or "STAGE5_REJECTED_OBJECT"
                else:
                    code = unknown or "UNKNOWN_OBJECT_ID"
                issues.append(make_issue(code, interview_id=iid, stage5_id=object_id))
        cited = {t for o in objects for t in o["evidence_turn_ids"]}
        cited |= {a["turn_id"] for o in objects for a in o.get("anchors", [])}
        turns = []
        for turn_id in dict.fromkeys(entry.get("evidence_turn_ids") or []):
            if turn_id not in self.turns.get(iid, set()):
                bad = True
                issues.append(make_issue(unknown or "UNKNOWN_EVIDENCE_TURN", interview_id=iid, turn_id=turn_id))
            else:
                turns.append(turn_id)
                if objects and turn_id not in cited:
                    issues.append(make_issue("EVIDENCE_TURN_OUTSIDE_SUPPORT", interview_id=iid, turn_id=turn_id))
        if not objects and not bad and not counter:
            issues.append(make_issue("EMPTY_SUPPORT_ENTRY", interview_id=iid))
        return {"interview_id": iid, "objects": objects, "turns": turns, "invalid": bad}


def _entry(resolved: dict, ctx: _Context) -> dict:
    objects = resolved["objects"]
    claims = ctx.material["claims"]
    reasons = sorted({r for o in objects for r in o["review_reasons"]})
    return {"interview_id": resolved["interview_id"],
            "claim_ids": [o["id"] for o in objects if o["id"] in claims],
            "criterion_ids": [o["id"] for o in objects if o["id"] not in claims],
            "evidence_turn_ids": resolved["turns"] or sorted({t for o in objects for t in o["evidence_turn_ids"]}),
            "needs_review": bool(objects) and all(o["needs_review"] for o in objects),
            "stage5_review_reasons": reasons}


def _merge_support(entries: list[dict], ctx: _Context, issues: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    seen: dict[str, int] = {}
    for entry in entries:
        resolved = ctx.resolve(entry, issues, counter=False)
        if resolved is None:
            continue
        iid = resolved["interview_id"]
        seen[iid] = seen.get(iid, 0) + 1
        if iid in merged:
            merged[iid]["objects"] += [o for o in resolved["objects"] if o not in merged[iid]["objects"]]
            merged[iid]["turns"] += [t for t in resolved["turns"] if t not in merged[iid]["turns"]]
            merged[iid]["invalid"] |= resolved["invalid"]
        else:
            merged[iid] = resolved
    for iid, count in seen.items():
        if count > 1:
            issues.append(make_issue("DUPLICATE_SUPPORT_INTERVIEW", interview_id=iid, entries=count))
    return {iid: r for iid, r in merged.items() if r["objects"]}


def _text_checks(texts: list[str], cited: set[str], ctx: _Context, claim: dict, counts: dict,
                 issues: list[dict]) -> None:
    authored = _normalize(" ".join(t for t in texts if t))
    material = " ".join(ctx.texts.get(iid, "") for iid in sorted(cited))
    for raw in texts:
        for match in _QUOTES.finditer(raw or ""):
            quote = _normalize(match.group(1) or match.group(2)).strip(" .,;:!?")
            if len(quote) >= 3 and quote not in material:
                issues.append(make_issue("QUOTE_NOT_IN_STAGE5", quote=(match.group(1) or match.group(2))[:120]))
    if _TYPOLOGY.search(authored) and not _TYPOLOGY.search(material):
        issues.append(make_issue("TYPOLOGY_OF_PERSONS", term=_TYPOLOGY.search(authored).group(0)))
    if _PERSON_SOCIAL.search(authored) and not _PERSON_SOCIAL.search(material):
        issues.append(make_issue("PERSON_OR_SOCIAL_ATTRIBUTION", term=_PERSON_SOCIAL.search(authored).group(0)))
    terms = [label for label, regex in _PSYCH if regex.search(authored) and not regex.search(material)]
    if terms:
        issues.append(make_issue("PSYCHOLOGICAL_VOCABULARY", terms=terms))
    if _CAUSAL.search(authored) and not _CAUSAL.search(material):
        issues.append(make_issue("CAUSAL_CLAIM", term=_CAUSAL.search(authored).group(0)))
    if _GENERALIZATION.search(authored) and not _GENERALIZATION.search(material):
        issues.append(make_issue("GENERALIZATION_BEYOND_N", term=_GENERALIZATION.search(authored).group(0)))
    if _OCCURRENCE.search(authored):
        issues.append(make_issue("OCCURRENCE_COUNT", term=_OCCURRENCE.search(authored).group(0)))
    if _ABSENCE.search(authored) and not _ABSENCE.search(material):
        issues.append(make_issue("NOT_OBSERVED_AS_ABSENCE", term=_ABSENCE.search(authored).group(0)))
    denominators = {ctx.material["corpus_n_usable"], ctx.material["corpus_n_total"]}

    def allowed(match) -> set[int]:
        """Comptes que le texte peut écrire ; celui des non-observés seulement s'il est dit comme tel."""
        values = {v for k, v in counts.items() if k != "not_observed"}
        window = authored[max(0, match.start() - 60):match.end() + 60]
        return values | ({counts["not_observed"]} if _NOT_OBSERVED_WORDS.search(window) else set())

    ratios = list(_RATIO.finditer(authored))
    for match in ratios:
        if _number(match.group(1)) not in allowed(match) or _number(match.group(2)) not in denominators:
            issues.append(make_issue("COUNT_IN_TEXT_MISMATCH", text=match.group(0)))
    spans = [m.span() for m in ratios]
    for match in _COUNT.finditer(authored):
        if any(s[0] <= match.start() < s[1] for s in spans):
            continue
        if _number(match.group(1)) not in allowed(match) | denominators:
            issues.append(make_issue("COUNT_IN_TEXT_MISMATCH", text=match.group(0)))
    raw = " ".join(t for t in texts if t)
    for token in dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", raw)):
        iid = ctx.mentions.get(token)
        if iid and iid not in cited:
            issues.append(make_issue("INTERVIEW_MENTION_NOT_SUPPORTED", interview_id=iid))


def _display(text: str, ctx: _Context) -> str:
    by_alias = {alias: iid for iid, alias in ctx.aliases.items()}
    return _ALIAS.sub(lambda m: by_alias.get(m.group(0), m.group(0)), text or "")


def validate_claim(claim: dict, claim_id: str, number: int, total: int, ctx: _Context) -> tuple[dict, list[dict]]:
    issues: list[dict] = []
    support = _merge_support(claim.get("support") or [], ctx, issues)
    if not support:
        issues.append(make_issue("NO_VALID_SUPPORT"))
    counters, positions = [], {iid: "not_observed" for iid in sorted(ctx.usable)}
    for iid in support:
        positions[iid] = "explicit_presence"
    for index, counter in enumerate(claim.get("counterexamples") or []):
        resolved = ctx.resolve(counter, issues, counter=True)
        relation = counter.get("relation") if counter.get("relation") in COUNTER_RELATIONS else "divergent_variant"
        if resolved is None or resolved["invalid"]:
            continue  # identifiant inventé : erreur déjà enregistrée, jamais un cas négatif
        entry = {**_entry(resolved, ctx), "relation": relation, "description": counter.get("description") or "",
                 "counter_index": index}
        if resolved["objects"]:
            entry["validation_status"] = "supported"
            if resolved["interview_id"] in support:
                issues.append(make_issue("SUPPORT_AND_COUNTEREXAMPLE", interview_id=resolved["interview_id"]))
            else:
                positions[resolved["interview_id"]] = relation
        else:
            entry.update(validation_status="unsupported", needs_review=True)
            issues.append(make_issue("COUNTEREXAMPLE_WITHOUT_SUPPORT", interview_id=resolved["interview_id"]))
        counters.append(entry)

    n_support = len(support)
    model_type = claim.get("claim_type")
    final_type = model_type
    model_n = claim.get("n_supporting_interviews")
    if support and model_n != n_support:
        issues.append(make_issue("COUNT_MISMATCH", model_value=model_n, recomputed=n_support))
    entries = {iid: _entry(r, ctx) for iid, r in support.items()}

    if final_type == "temporal_pattern" and support:
        temporal = [iid for iid, r in support.items()
                    if any(o.get("claim_type") == "explicit_temporal_change" for o in r["objects"])]
        missing = [iid for iid in support if iid not in temporal]
        if not temporal:
            final_type = "contextual_association"
            issues.append(make_issue("TEMPORAL_PATTERN_REQUALIFIED"))
        elif missing:
            issues.append(make_issue("TEMPORAL_SUPPORT_NOT_VALIDATED", interview_ids=missing))
    if final_type in REGULARITY_TYPES and n_support == 1:
        final_type = "minority_configuration"
        issues.append(make_issue("RECURRENCE_SINGLE_INTERVIEW"))
    contrasted = {c["interview_id"] for c in counters if c["validation_status"] == "supported"}
    if final_type in CONTRAST_TYPES and len(set(support) | contrasted) < 2:
        issues.append(make_issue("CONTRAST_SINGLE_INTERVIEW"))
    if final_type in CRITERION_TYPES:
        without = [iid for iid, e in entries.items() if not e["criterion_ids"]]
        if without:
            issues.append(make_issue("CRITERION_PATTERN_WITHOUT_CRITERIA", interview_ids=without))
    related = [n for n in claim.get("related_claim_numbers") or [] if isinstance(n, int) and 1 <= n <= total
               and n != number]
    if final_type == "negative_case" and not related:
        issues.append(make_issue("NEGATIVE_CASE_UNRELATED"))

    forced = []
    review_only = sorted(iid for iid, e in entries.items() if e["needs_review"])
    clean = n_support - len(review_only)
    confidence = claim.get("confidence")
    if entries and clean == 0:
        forced.append("SUPPORTED_ONLY_BY_REVIEW_ITEMS")
        issues.append(make_issue("SUPPORTED_ONLY_BY_REVIEW_ITEMS", interview_ids=review_only))
        if confidence == "high":
            issues.append(make_issue("STRONG_PATTERN_ON_REVIEW_ITEMS"))
        confidence = _min_confidence(confidence, "low")
    elif final_type in REGULARITY_TYPES and clean < 2:
        issues.append(make_issue("RECURRENCE_RESTS_ON_REVIEW_ITEMS", clean_interviews=clean,
                                 review_only_interview_ids=review_only))
        confidence = _min_confidence(confidence, "low")
    elif review_only or any(e["stage5_review_reasons"] for e in entries.values()):
        issues.append(make_issue("REVIEW_ITEMS_PROPAGATED", interview_ids=review_only or None))
    if n_support == 1 and confidence != _min_confidence(confidence, "low"):
        confidence = _min_confidence(confidence, "low")
        issues.append(make_issue("SINGLE_INTERVIEW_CONFIDENCE_CAPPED"))
    if ctx.material["mode"] == "exploratory" and confidence != _min_confidence(confidence, "medium"):
        confidence = _min_confidence(confidence, "medium")
        issues.append(make_issue("EXPLORATORY_CONFIDENCE_CAPPED"))
    if claim.get("needs_review"):
        forced.append("MODEL_FLAGGED_REVIEW")
        issues.append(make_issue("MODEL_FLAGGED_REVIEW"))

    supported_counters = [c for c in counters if c["validation_status"] == "supported"]
    cited = set(support) | {c["interview_id"] for c in counters}
    counts = {"support": n_support, "counter": len({c["interview_id"] for c in supported_counters}),
              "review_only": len(review_only), "clean": clean,
              "not_observed": sum(p == "not_observed" for p in positions.values()),
              "support_and_counter": n_support + len({c["interview_id"] for c in supported_counters})}
    texts = [claim.get("description") or "", *(claim.get("contexts") or []), claim.get("criterion_label") or "",
             *(c["description"] for c in counters)]
    _text_checks(texts, cited, ctx, claim, counts, issues)

    severities = {i["severity"] for i in issues}
    status = "rejected" if ERROR in severities else "needs_review" if WARNING in severities else "valid"
    annotated = {
        "cross_claim_id": claim_id, "claim_type": final_type,
        **({"model_claim_type": model_type} if final_type != model_type else {}),
        "description": claim.get("description") or "", "description_display": _display(claim.get("description"), ctx),
        "criterion_label": claim.get("criterion_label"),
        "interview_ids": sorted(support), "support": [entries[iid] for iid in sorted(entries)],
        "counterexamples": counters, "contexts": list(claim.get("contexts") or []),
        "corpus_n_total": ctx.material["corpus_n_total"], "corpus_n_usable": ctx.material["corpus_n_usable"],
        "n_supporting_interviews": n_support, "model_n_supporting_interviews": model_n,
        "n_review_only_interviews": len(review_only), "review_only_interview_ids": review_only,
        "n_counterexample_interviews": counts["counter"],
        "n_not_observed_interviews": counts["not_observed"],
        "interview_positions": positions,
        "not_observed_interview_ids": [iid for iid, p in positions.items() if p == "not_observed"],
        "related_claim_numbers": related,
        "stage5_review_reasons": sorted({r for e in entries.values() for r in e["stage5_review_reasons"]}),
        "confidence": confidence,
        **({"model_confidence": claim.get("confidence")} if confidence != claim.get("confidence") else {}),
        "model_needs_review": bool(claim.get("needs_review")),
        "needs_review": bool(forced) or status != "valid",
        "review_reasons": sorted({i["code"] for i in issues if i["severity"] != INFO} | set(forced)),
        "validation_status": status, "usable_for_next_stages": status != "rejected",
    }
    return annotated, issues


def _negative_cases(claims: list[dict]) -> list[dict]:
    """Tous les cas négatifs documentés, dédoublonnés par (entretien, régularité compliquée). Jamais supprimés."""
    ids = [c["cross_claim_id"] for c in claims]
    cases: dict[tuple, dict] = {}

    def add(iid, related, relation, description, support_ids, source, needs_review, status):
        key = (iid, tuple(sorted(related)))
        case = cases.setdefault(key, {"interview_id": iid, "related_cross_claim_ids": sorted(related),
                                      "relations": [], "descriptions": [], "support_ids": [], "sources": [],
                                      "needs_review": False, "validation_status": status})
        for field, value in (("relations", relation), ("descriptions", description), ("sources", source)):
            if value and value not in case[field]:
                case[field].append(value)
        case["support_ids"] += [s for s in support_ids if s not in case["support_ids"]]
        case["needs_review"] |= needs_review
        if status == "supported":
            case["validation_status"] = "supported"

    for claim in claims:
        rejected = not claim["usable_for_next_stages"]
        for counter in claim["counterexamples"]:
            add(counter["interview_id"], [claim["cross_claim_id"]], counter["relation"], counter["description"],
                counter["claim_ids"] + counter["criterion_ids"], f"{claim['cross_claim_id']}.counterexamples",
                counter["needs_review"] or counter["validation_status"] != "supported" or rejected,
                counter["validation_status"])
        if claim["claim_type"] == "negative_case" or claim.get("model_claim_type") == "negative_case":
            related = [ids[n - 1] for n in claim["related_claim_numbers"]]
            for entry in claim["support"]:
                add(entry["interview_id"], related, "negative_case", claim["description"],
                    entry["claim_ids"] + entry["criterion_ids"], claim["cross_claim_id"],
                    entry["needs_review"] or claim["needs_review"] or rejected, "supported")
    result = []
    for number, case in enumerate(cases.values(), start=1):
        result.append({"negative_case_id": f"NC{number:03d}", **case})
    return result


def configuration_distribution(material: dict) -> dict:
    """Configurations de l'étape 5, comptées en entretiens (aucun type d'étudiant)."""
    distribution: dict[str, dict] = {}
    for iid in material["interview_ids"]:
        kind = material["interviews"][iid]["configuration_type"] or "unknown"
        entry = distribution.setdefault(kind, {"n_interviews": 0, "interview_ids": []})
        entry["n_interviews"] += 1
        entry["interview_ids"].append(iid)
    n = material["corpus_n_usable"]
    for entry in distribution.values():
        entry["share"] = f"{entry['n_interviews']} entretien(s) sur {n} exploitable(s)"
    return dict(sorted(distribution.items()))


def criterion_patterns(claims: list[dict]) -> list[dict]:
    """student_role_criterion_patterns : vue des patterns de critères (identifiants, positions, contrastes)."""
    patterns = []
    for c in claims:
        if not c["usable_for_next_stages"] or c["claim_type"] not in CRITERION_TYPES:
            continue
        positions = {p: [iid for iid, v in c["interview_positions"].items() if v == p] for p in POSITIONS}
        patterns.append({
            "cross_claim_id": c["cross_claim_id"], "claim_type": c["claim_type"],
            "criterion_label": c["criterion_label"], "description": c["description_display"],
            "interview_ids": c["interview_ids"],
            "criterion_ids": {e["interview_id"]: e["criterion_ids"] for e in c["support"]},
            "evidence": {e["interview_id"]: e["evidence_turn_ids"] for e in c["support"]},
            "positions": positions,
            "contrasts": [{"interview_id": x["interview_id"], "relation": x["relation"],
                           "support_ids": x["claim_ids"] + x["criterion_ids"], "description": x["description"]}
                          for x in c["counterexamples"]],
            "n_supporting_interviews": c["n_supporting_interviews"], "corpus_n_usable": c["corpus_n_usable"],
            "confidence": c["confidence"], "needs_review": c["needs_review"], "review_reasons": c["review_reasons"]})
    return patterns


def validate_comparison(output: dict, material: dict, excluded_ids: set[str] | None = None) -> dict:
    """Valide la sortie (identifiants entiers) du Comparator. Renvoie {"document": champs, "report": bilan}."""
    ctx = _Context(material, set(excluded_ids or ()))
    drafts = output.get("cross_case_claims") or []
    all_issues: list[dict] = []
    claims = []
    for number, draft in enumerate(drafts, start=1):
        claim_id = f"CC{number:03d}"
        annotated, issues = validate_claim(draft, claim_id, number, len(drafts), ctx)
        claims.append(annotated)
        all_issues.extend({"object_id": claim_id, **i} for i in issues)
    kept = [c for c in claims if c["usable_for_next_stages"]]
    lists = {key: [c["cross_claim_id"] for c in kept if c["claim_type"] == t] for t, key in LIST_KEYS.items()}

    doc_issues = []
    if material["mode"] == "exploratory":
        doc_issues.append(make_issue("EXPLORATORY_CORPUS"))
    summary = output.get("cross_case_summary") or ""
    folded = _normalize(summary)
    every = " ".join(ctx.texts.values())
    if (_TYPOLOGY.search(folded) and not _TYPOLOGY.search(every)) or (
            _PERSON_SOCIAL.search(folded) and not _PERSON_SOCIAL.search(every)) or any(
            r.search(folded) and not r.search(every) for _, r in _PSYCH):
        doc_issues.append(make_issue("SUMMARY_VOCABULARY"))
    if _CAUSAL.search(folded) and not _CAUSAL.search(every):
        doc_issues.append(make_issue("SUMMARY_CAUSAL"))
    if (_GENERALIZATION.search(folded) and not _GENERALIZATION.search(every)) or _OCCURRENCE.search(folded):
        doc_issues.append(make_issue("SUMMARY_GENERALIZATION"))
    for match in _QUOTES.finditer(summary):
        quote = _normalize(match.group(1) or match.group(2)).strip(" .,;:!?")
        if len(quote) >= 3 and quote not in every:
            doc_issues.append(make_issue("SUMMARY_QUOTE_NOT_IN_STAGE5", quote=quote[:120]))
    for token in dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", summary)):
        if _ALIAS.fullmatch(token) and token not in ctx.mentions:
            doc_issues.append(make_issue("SUMMARY_UNKNOWN_INTERVIEW", interview_id=token))
        elif ctx.mentions.get(token) in ctx.excluded:
            doc_issues.append(make_issue("SUMMARY_UNKNOWN_INTERVIEW", interview_id=token))
    all_issues.extend({"object_id": "corpus", **i} for i in doc_issues)
    if output.get("needs_review"):
        all_issues.append({"object_id": "corpus", **make_issue("MODEL_FLAGGED_REVIEW")})

    severities = [i["severity"] for i in all_issues]
    model_confidence = output.get("confidence")
    confidence = _min_confidence(model_confidence, "medium") if material["mode"] == "exploratory" \
        else model_confidence
    negative = _negative_cases(claims)
    document = {
        "configuration_distribution": configuration_distribution(material),
        "cross_case_claims": claims,
        **lists,
        "student_role_criterion_patterns": criterion_patterns(claims),
        "negative_cases": negative,
        "cross_case_summary": summary,
        "cross_case_summary_display": _display(summary, ctx),
        "confidence": confidence,
        **({"model_confidence": model_confidence} if confidence != model_confidence else {}),
        "model_needs_review": bool(output.get("needs_review")),
        "needs_review": bool(output.get("needs_review")) or ERROR in severities or WARNING in severities,
        "comparator_notes": output.get("comparator_notes"),
    }
    report = {
        "validator_version": VALIDATOR_VERSION,
        "cross_claim_count": len(claims), "kept_cross_claim_count": len(kept),
        "rejected_cross_claim_ids": [c["cross_claim_id"] for c in claims if not c["usable_for_next_stages"]],
        "requalified_cross_claim_ids": [c["cross_claim_id"] for c in claims if "model_claim_type" in c],
        "cross_claims_needing_review": [c["cross_claim_id"] for c in claims if c["needs_review"]],
        "negative_case_count": len(negative),
        "error_count": severities.count(ERROR), "warning_count": severities.count(WARNING),
        "info_count": severities.count(INFO), "issues": all_issues,
    }
    return {"document": document, "report": report}


def has_problems(report: dict) -> bool:
    return bool(report.get("error_count") or report.get("warning_count"))


assert set(LIST_KEYS) == set(CROSS_CLAIM_TYPES)
