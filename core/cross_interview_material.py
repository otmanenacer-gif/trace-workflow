"""Étape 6 — préparation DÉTERMINISTE du matériau inter-entretiens (aucun LLM).

Entrée : les triplets de l'étape 5 reconnus exploitables (core/cross_interview_corpus.py). Rien d'autre : ni
transcription, ni sorties des étapes 3 et 4. Les seules citations disponibles sont celles que l'étape 5 a
conservées (ancrages temporels validés : copies exactes) ; les tours cités sont repris par identifiant.

Pour chaque entretien sont retenus : configuration, affirmations UTILISABLES (`usable_for_next_stages`) avec
leur type, description, contextes, confiance, `needs_review` et `review_reasons`, opérations, ancrages validés,
tours cités, requalification éventuelle ; critères du métier d'étudiant utilisables avec les mêmes champs. Les
affirmations rejetées par l'étape 5 sont comptées, jamais transmises.

Index (identifiants groupés, AUCUNE conclusion) :
A. affirmations par famille (type d'affirmation de l'étape 5), avec, pour chaque famille, les entretiens où
   elle est explicitement présente et ceux où elle n'est PAS OBSERVÉE (jamais « absente ») ;
B. critères du métier d'étudiant par entretien ;
C. opérations (accounting moves) par entretien, et entretiens par opération ;
D. contextes / tâches ;
E-H. frontières stables, exceptions, tensions, zones ordinaires (et variations, changements temporels,
   opérations récurrentes) par entretien.

Regroupements lexicaux (contextes, libellés de critères) : seulement à formulation IDENTIQUE après repliement
(casse, accents, ponctuation). Deux formulations proches ne sont jamais déclarées équivalentes : c'est au
Comparator de proposer un rapprochement, que le validateur contrôle sur les identifiants.

Représentation envoyée (`build_payload`) : NORMALISÉE — entretiens abrégés (`I01`…, table de correspondance
incluse), identifiants locaux de l'étape 5 (`TC003`, `RC001`, `T0040`), un entretien par ligne, valeurs par
défaut omises (confiance « medium », listes vides), JSON compact. `expand_ids` rétablit les identifiants entiers.
"""

from __future__ import annotations

import json
import re

from core import interpretation_guard
from core import trajectory_validator as stage5_validator
from core.interaction_chunking import estimate_tokens

PREPROCESSOR_VERSION = "1.0"
PAYLOAD_FORMAT_VERSION = "1"

PRESENT = "explicit_presence"
NOT_OBSERVED = "not_observed"
FAMILY_LISTS = stage5_validator.LIST_KEYS  # type d'affirmation de l'étape 5 → nom de liste


def fold_key(text: str | None) -> str:
    """Clé de regroupement à formulation identique (aucune équivalence sémantique)."""
    return " ".join(re.findall(r"[a-z0-9]+", interpretation_guard.fold(text or "")))


def _local(full_id: str, interview_id: str) -> str:
    prefix = f"{interview_id}_"
    return full_id[len(prefix):] if full_id.startswith(prefix) else full_id


def build_material(usable: dict[str, dict], n_total: int, mode: str) -> dict:
    """Matériau complet (identifiants entiers). `usable` : {interview_id: document student_trajectory}."""
    ids = sorted(usable)
    aliases = {iid: f"I{n:02d}" for n, iid in enumerate(ids, start=1)}
    interviews, claims, criteria = {}, {}, {}
    turns: dict[str, set[str]] = {}
    texts: dict[str, list[str]] = {}
    for iid in ids:
        doc = usable[iid]
        kept = [c for c in doc["trajectory_claims"] if c.get("usable_for_next_stages")]
        kept_criteria = [c for c in doc["student_role_criteria"] if c.get("usable_for_next_stages")]
        turns[iid] = set()
        texts[iid] = []
        for c in kept:
            anchors = [{"text": a["text"], "turn_id": a["turn_id"], "kind": a.get("kind")}
                       for a in c.get("validated_temporal_anchors") or []]
            claims[c["claim_id"]] = {
                "id": c["claim_id"], "local": _local(c["claim_id"], iid), "interview_id": iid,
                "claim_type": c["claim_type"], "description": c.get("description") or "",
                "contexts": list(c.get("contexts") or []), "confidence": c.get("confidence"),
                "needs_review": bool(c.get("needs_review")), "review_reasons": list(c.get("review_reasons") or []),
                "move_types": list(c.get("accounting_move_types") or []), "anchors": anchors,
                "evidence_turn_ids": list(c.get("evidence_turn_ids") or []),
                "model_claim_type": c.get("model_claim_type")}
            turns[iid].update(c.get("evidence_turn_ids") or [])
            turns[iid].update(a["turn_id"] for a in anchors)
            texts[iid] += [c.get("description") or "", *(c.get("contexts") or []), *(a["text"] for a in anchors)]
        for c in kept_criteria:
            criteria[c["criterion_id"]] = {
                "id": c["criterion_id"], "local": _local(c["criterion_id"], iid), "interview_id": iid,
                "criterion": c.get("criterion") or "", "description": c.get("description") or "",
                "confidence": c.get("confidence"), "needs_review": bool(c.get("needs_review")),
                "review_reasons": list(c.get("review_reasons") or []),
                "evidence_turn_ids": list(c.get("evidence_turn_ids") or [])}
            turns[iid].update(c.get("evidence_turn_ids") or [])
            texts[iid] += [c.get("criterion") or "", c.get("description") or ""]
        interviews[iid] = {
            "interview_id": iid, "alias": aliases[iid], "configuration_type": doc.get("configuration_type"),
            "model_configuration_type": doc.get("model_configuration_type"), "status": doc.get("status"),
            "confidence": doc.get("confidence"), "needs_review": bool(doc.get("needs_review")),
            "claim_ids": [c["claim_id"] for c in kept], "criterion_ids": [c["criterion_id"] for c in kept_criteria],
            "rejected_claim_count": len(doc["trajectory_claims"]) - len(kept),
            "rejected_criterion_count": len(doc["student_role_criteria"]) - len(kept_criteria),
            "rejected_object_ids": (
                [c["claim_id"] for c in doc["trajectory_claims"] if c not in kept]
                + [c["criterion_id"] for c in doc["student_role_criteria"] if c not in kept_criteria]),
            "review_object_ids": (
                [c["claim_id"] for c in kept if c.get("needs_review")]
                + [c["criterion_id"] for c in kept_criteria if c.get("needs_review")])}
    material = {
        "preprocessor_version": PREPROCESSOR_VERSION,
        "corpus_n_total": n_total, "corpus_n_usable": len(ids), "mode": mode, "interview_ids": ids,
        "aliases": aliases, "interviews": interviews, "claims": claims, "criteria": criteria,
        "turns_by_interview": {iid: sorted(t) for iid, t in turns.items()},
        "texts_by_interview": {iid: interpretation_guard.fold(" ".join(t)) for iid, t in texts.items()},
    }
    material["indexes"] = build_indexes(material)
    material["summary"] = {
        "corpus_n_total": n_total, "corpus_n_usable": len(ids), "mode": mode,
        "claim_count": len(claims), "criterion_count": len(criteria),
        "review_claim_count": sum(c["needs_review"] for c in claims.values()),
        "review_criterion_count": sum(c["needs_review"] for c in criteria.values()),
        "rejected_stage5_object_count": sum(i["rejected_claim_count"] + i["rejected_criterion_count"]
                                            for i in interviews.values()),
    }
    return material


def _groups(entries: list[tuple[str, str, str]]) -> list[dict]:
    """[(texte, interview_id, object_id)] → groupes à formulation identique, triés (plus d'entretiens d'abord)."""
    groups: dict[str, dict] = {}
    for text, iid, object_id in entries:
        key = fold_key(text)
        if not key:
            continue
        group = groups.setdefault(key, {"key": key, "labels": [], "interview_ids": [], "object_ids": []})
        for field, value in (("labels", text), ("interview_ids", iid), ("object_ids", object_id)):
            if value not in group[field]:
                group[field].append(value)
    return sorted(groups.values(), key=lambda g: (-len(g["interview_ids"]), g["key"]))


def build_indexes(material: dict) -> dict:
    ids = material["interview_ids"]
    claims, criteria = material["claims"].values(), material["criteria"].values()
    by_family = {t: [c["id"] for c in claims if c["claim_type"] == t] for t in FAMILY_LISTS}
    family_presence = {}
    for t, claim_ids in by_family.items():
        present = sorted({material["claims"][c]["interview_id"] for c in claim_ids})
        family_presence[t] = {PRESENT: present, NOT_OBSERVED: [i for i in ids if i not in present]}
    moves_by_interview: dict[str, dict[str, list[str]]] = {iid: {} for iid in ids}
    for c in claims:
        for move in c["move_types"]:
            moves_by_interview[c["interview_id"]].setdefault(move, []).append(c["id"])
    interviews_by_move: dict[str, list[str]] = {}
    for iid, moves in moves_by_interview.items():
        for move in moves:
            interviews_by_move.setdefault(move, []).append(iid)
    per_family = {key: {iid: [c for c in by_family[t] if material["claims"][c]["interview_id"] == iid] for iid in ids}
                  for t, key in FAMILY_LISTS.items()}
    return {
        "claims_by_family": by_family,                                                     # A
        "family_presence": family_presence,
        "criteria_by_interview": {iid: material["interviews"][iid]["criterion_ids"] for iid in ids},  # B
        "accounting_moves_by_interview": moves_by_interview,                               # C
        "interviews_by_move": dict(sorted(interviews_by_move.items())),
        "contexts": _groups([(ctx, c["interview_id"], c["id"]) for c in claims for ctx in c["contexts"]]),  # D
        "criterion_labels": _groups([(c["criterion"], c["interview_id"], c["id"]) for c in criteria]),
        "boundaries": per_family["stable_boundaries"],                                     # E
        "exceptions": per_family["exceptions"],                                            # F
        "tensions": per_family["unresolved_tensions"],                                     # G
        "ordinary_zones": per_family["ordinary_zones"],                                    # H
        "contextual_variations": per_family["contextual_variations"],
        "temporal_changes": per_family["explicit_temporal_changes"],
        "recurring_accounting_moves": per_family["recurring_accounting_moves"],
    }


def public(material: dict) -> dict:
    """Ce que le fichier de sortie garde de la préparation : comptes et index (identifiants seulement)."""
    indexes = material["indexes"]
    return {"preprocessor_version": PREPROCESSOR_VERSION, **material["summary"],
            "aliases": material["aliases"],
            "family_presence": indexes["family_presence"],
            "criteria_by_interview": indexes["criteria_by_interview"],
            "interviews_by_move": indexes["interviews_by_move"],
            "shared_contexts": [g for g in indexes["contexts"] if len(g["interview_ids"]) >= 2],
            "shared_criterion_labels": [g for g in indexes["criterion_labels"] if len(g["interview_ids"]) >= 2],
            "grouping_basis": "identical_wording"}


# --- Représentation envoyée (normalisée, identifiants abrégés) -------------------------------------------

def _compact(data: dict) -> dict:
    return {k: v for k, v in data.items() if v not in (None, [], {}, "")}


def _ref(material: dict, object_id: str) -> str:
    obj = material["claims"].get(object_id) or material["criteria"][object_id]
    return f"{material['aliases'][obj['interview_id']]}:{obj['local']}"


def build_payload(material: dict, interview_ids: list[str] | None = None) -> dict:
    """Représentation compacte des entretiens choisis (tous par défaut)."""
    selected = interview_ids or material["interview_ids"]
    aliases = material["aliases"]

    def short_turn(turn_id: str, iid: str) -> str:
        return _local(turn_id, iid)

    interviews = []
    for iid in selected:
        info = material["interviews"][iid]
        claims = {}
        for cid in info["claim_ids"]:
            c = material["claims"][cid]
            claims[c["local"]] = _compact({
                "type": c["claim_type"], "text": c["description"], "contexts": c["contexts"],
                "confidence": None if c["confidence"] == "medium" else c["confidence"], "moves": c["move_types"],
                "anchors": [[a["text"], short_turn(a["turn_id"], iid)] for a in c["anchors"]],
                "turns": [short_turn(t, iid) for t in c["evidence_turn_ids"]],
                "review": c["review_reasons"] if c["needs_review"] else [],
                "requalified_from": c["model_claim_type"]})
        criteria = {}
        for rid in info["criterion_ids"]:
            c = material["criteria"][rid]
            criteria[c["local"]] = _compact({
                "criterion": c["criterion"], "text": c["description"],
                "confidence": None if c["confidence"] == "medium" else c["confidence"],
                "turns": [short_turn(t, iid) for t in c["evidence_turn_ids"]],
                "review": c["review_reasons"] if c["needs_review"] else []})
        interviews.append(_compact({
            "id": aliases[iid], "interview_id": iid, "configuration": info["configuration_type"],
            "claims": claims, "criteria": criteria}))
    wanted = set(selected)
    indexes = material["indexes"]
    families = {t: _compact({"present": [aliases[i] for i in p[PRESENT] if i in wanted],
                             "not_observed": [aliases[i] for i in p[NOT_OBSERVED] if i in wanted]})
                for t, p in indexes["family_presence"].items()}

    def shared(groups: list[dict]) -> list[dict]:
        out = []
        for g in groups:
            refs = [_ref(material, o) for o in g["object_ids"]
                    if (material["claims"].get(o) or material["criteria"].get(o))["interview_id"] in wanted]
            if len({r.split(":")[0] for r in refs}) >= 2:
                out.append({"labels": g["labels"], "refs": refs})
        return out

    moves = {}
    for iid in selected:
        for move, claim_ids in indexes["accounting_moves_by_interview"][iid].items():
            moves.setdefault(move, []).extend(_ref(material, c) for c in claim_ids)
    return {
        "corpus": {"n_total": material["corpus_n_total"], "n_usable": material["corpus_n_usable"],
                   "n_in_this_call": len(selected), "mode": material["mode"]},
        "interviews": interviews,
        "indexes": _compact({"families": families, "moves": dict(sorted(moves.items())),
                             "shared_contexts": shared(indexes["contexts"]),
                             "shared_criterion_labels": shared(indexes["criterion_labels"])}),
    }


def serialize_payload(payload: dict) -> str:
    """JSON compact et déterministe ; un entretien par ligne. « </ » échappé (le texte ne ferme pas la balise)."""
    head = {k: v for k, v in payload.items() if k != "interviews"}
    lines = [json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for x in payload["interviews"]]
    text = json.dumps(head, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:-1]
    return (text + ',"interviews":[\n' + ",\n".join(lines) + "\n]}").replace("</", "<\\/")


def payload_estimate(text: str) -> int:
    return estimate_tokens(text)


def raw_size(usable_files: dict[str, dict]) -> int:
    """Taille des triplets bruts de l'étape 5 (pour comparaison avec la représentation envoyée)."""
    return sum(len(data) for files in usable_files.values() for _, data, _ in files.values())


# --- Réponse du modèle : identifiants entiers ---------------------------------------------------------------

_LOCAL = re.compile(r"^(?:(?P<alias>I\d+)[:/.])?(?P<local>(?:TC|RC|T)\d+)$")


def expand_ids(output: dict, material: dict) -> dict:
    """Rétablit les identifiants entiers (entretiens, affirmations, critères, tours) dans la réponse du modèle.
    Une valeur inconnue est laissée telle quelle : le validateur la signale (jamais corrigée en silence)."""
    by_alias = {alias: iid for iid, alias in material["aliases"].items()}

    def interview(value):
        if value in by_alias:
            return by_alias[value]
        return value

    def full(value, iid):
        match = _LOCAL.match(value) if isinstance(value, str) else None
        if not match:
            return value
        owner = by_alias.get(match.group("alias"), iid) if match.group("alias") else iid
        return f"{owner}_{match.group('local')}" if owner else value

    result = json.loads(json.dumps(output))
    for claim in result.get("cross_case_claims", []):
        for entry in [*claim.get("support", []), *claim.get("counterexamples", [])]:
            entry["interview_id"] = interview(entry.get("interview_id"))
            iid = entry["interview_id"] if entry["interview_id"] in material["interviews"] else None
            for key in ("claim_ids", "criterion_ids", "evidence_turn_ids"):
                entry[key] = [full(v, iid) for v in entry.get(key, [])]
    return result
