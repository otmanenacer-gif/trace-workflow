"""Étape 6 — constitution du corpus : import et contrôle des sorties de l'étape 5 de plusieurs entretiens.

L'étape 6 ne reçoit QUE les triplets de l'étape 5, jamais les transcriptions ni les sorties des étapes 3 et 4 :

    student_trajectory.json, student_trajectory_validation.json, student_trajectory_manifest.json

Ils viennent de fichiers téléchargés (le stockage local de l'application est éphémère) ou des sorties de
l'étape 5 du run courant (`run_stage5_uploads`). Aucun appel LLM, aucune étape 3, 4 ou 5 n'est relancée.

Reconnaissance par le CONTENU (jamais par le seul nom de fichier, préfixé ou non), puis regroupement des
fichiers par `interview_id`. Un même fichier importé deux fois à l'identique est dédoublonné ; deux versions
différentes du même fichier pour un même entretien forment un DOUBLON AMBIGU : l'entretien est exclu.

Contrôles par entretien (tout manquement exclut l'entretien, avec sa raison ; les autres restent exploitables) :
- les 3 fichiers présents, même interview_id, identifiants des affirmations / critères préfixés par cet entretien ;
- versions compatibles : agent Trajectory Mapper, schéma et validateur de l'étape 5 identiques à ceux de TRACE
  (une autre version du prompt, de l'agent ou du préparateur est seulement signalée) ;
- statut exploitable (SUCCESS, SUCCESS_WITH_WARNINGS, CACHED), `analysis_complete: true` partout ;
- `validation_error_count = 0` (document, manifest et fichier de validation) ;
- empreintes : `source_hashes` identiques dans le document et le manifest ; le manifest décrit ce document
  (statut, configuration, comptes, fichier de sortie) ;
- aucune altération : comptes, listes par type d'affirmation, statuts et `needs_review` / `review_reasons`
  RECALCULÉS depuis les affirmations et les anomalies du fichier de validation, et comparés aux valeurs déclarées.

Seuil : moins de 2 entretiens exploitables → étape 6 bloquée ; 2 → comparaison exploratoire ; 3 et plus →
comparaison normale. N total et N exploitable sont toujours affichés.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents import trajectory_mapper as mapper
from core import config, trajectory
from core import trajectory_validator as stage5_validator
from core.analysis import STATUS_CACHED, STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, eligible_files

CORPUS_CHECK_VERSION = "1.0"
KIND_TRAJECTORY = config.STUDENT_TRAJECTORY_FILENAME
KIND_VALIDATION = config.STUDENT_TRAJECTORY_VALIDATION_FILENAME
KIND_MANIFEST = config.STUDENT_TRAJECTORY_MANIFEST_FILENAME
KINDS = {KIND_TRAJECTORY: "configuration (student_trajectory)", KIND_VALIDATION: "validation",
         KIND_MANIFEST: "manifest"}
USABLE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)

MODE_BLOCKED = "blocked"
MODE_EXPLORATORY = "exploratory"
MODE_COMPARATIVE = "comparative"
MIN_USABLE = 2
EXPLORATORY_MAX = 2

STATUS_USABLE = "exploitable"
STATUS_EXCLUDED = "exclu"

REPORT_KEYS = ("claim_count", "kept_claim_count", "rejected_claim_ids", "requalified_claim_ids",
               "claims_needing_review", "criterion_count", "rejected_criterion_ids")
MATERIAL_KEYS = ("usable_episode_count", "excluded_episode_count", "unmarked_practice_count", "temporal_anchor_count")


def corpus_mode(n_usable: int) -> str:
    if n_usable < MIN_USABLE:
        return MODE_BLOCKED
    return MODE_EXPLORATORY if n_usable <= EXPLORATORY_MAX else MODE_COMPARATIVE


def _kind(document) -> str | None:
    """Type d'un fichier d'après son CONTENU."""
    if not isinstance(document, dict) or not document.get("interview_id"):
        return None
    if document.get("agent") == mapper.AGENT_NAME:
        if "trajectory_claims" in document:
            return KIND_TRAJECTORY
        if "manifest_version" in document:
            return KIND_MANIFEST
        return None
    if "agent" not in document and {"validator_version", "stage4", "available"} <= set(document):
        return KIND_VALIDATION
    return None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _group(uploads: list[tuple[str, bytes]]) -> tuple[dict, list[dict]]:
    """{interview_id: {kind: [(nom, octets, document)]}} et fichiers non reconnus."""
    groups: dict[str, dict[str, list]] = {}
    unrecognized = []
    for name, data in uploads:
        try:
            document = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError):
            unrecognized.append({"file": name, "reason": "JSON invalide."})
            continue
        kind = _kind(document)
        if kind is None:
            unrecognized.append({"file": name, "reason": "Fichier non reconnu (ni configuration, ni validation, ni "
                                                         "manifest de l'étape 5, ou interview_id absent)."})
            continue
        groups.setdefault(str(document["interview_id"]), {}).setdefault(kind, []).append((name, data, document))
    return groups, unrecognized


def _counts_problems(doc: dict) -> list[str]:
    """Comptes et listes RECALCULÉS depuis les affirmations (document non altéré)."""
    problems = []
    try:
        counts = trajectory._counts(doc)
    except (KeyError, TypeError):
        return ["Configuration illisible (structure inattendue)."]
    declared = {k: doc.get(k) for k in counts}
    if declared != counts:
        problems.append("Comptes incohérents avec les affirmations : " + ", ".join(
            f"{k} {declared[k]} ≠ {v}" for k, v in counts.items() if declared[k] != v) + ".")
    kept = [c for c in doc["trajectory_claims"] if c.get("usable_for_next_stages")]
    for claim_type, key in stage5_validator.LIST_KEYS.items():
        expected = [c["claim_id"] for c in kept if c.get("claim_type") == claim_type]
        if doc.get(key) != expected:
            problems.append(f"Liste {key} incohérente avec les affirmations retenues.")
    return problems


def _object_problems(doc: dict, validation: dict) -> list[str]:
    """Statut, needs_review et review_reasons de chaque objet cohérents avec les anomalies du fichier de validation."""
    problems = []
    iid = doc["interview_id"]
    issues: dict[str, list[dict]] = {}
    for issue in validation.get("issues") or []:
        issues.setdefault(issue.get("object_id"), []).append(issue)
    objects = [("claim_id", c) for c in doc["trajectory_claims"]] + [("criterion_id", c)
                                                                     for c in doc["student_role_criteria"]]
    ids = [obj.get(key) for key, obj in objects]
    if len(set(ids)) != len(ids):
        problems.append("Identifiants d'affirmations ou de critères en double.")
    for key, obj in objects:
        object_id = obj.get(key) or ""
        if not object_id.startswith(f"{iid}_"):
            problems.append(f"{object_id or '(sans identifiant)'} : identifiant étranger à l'entretien {iid}.")
            continue
        own = issues.get(object_id, [])
        severities = {i.get("severity") for i in own}
        status = ("rejected" if stage5_validator.ERROR in severities else
                  "needs_review" if stage5_validator.WARNING in severities else "valid")
        reasons = set(obj.get("review_reasons") or [])
        non_info = {i.get("code") for i in own if i.get("severity") != stage5_validator.INFO}
        if obj.get("validation_status") != status or obj.get("usable_for_next_stages") != (status != "rejected"):
            problems.append(f"{object_id} : statut de validation incohérent avec les anomalies enregistrées.")
        if not non_info <= reasons <= {i.get("code") for i in own}:
            problems.append(f"{object_id} : review_reasons incohérentes avec les anomalies enregistrées.")
        if bool(obj.get("needs_review")) != bool(reasons):
            problems.append(f"{object_id} : needs_review incohérent avec ses review_reasons.")
    return problems


def check_triplet(interview_id: str, files: dict[str, tuple[str, bytes, dict]]) -> dict:
    """Contrôle un triplet {kind: (nom, octets, document)}. Renvoie {"reasons": [...], "warnings": [...]}."""
    reasons, warnings = [], []
    missing = [k for k in KINDS if k not in files]
    if missing:
        return {"reasons": [f"Fichier manquant : {k} ({KINDS[k]})." for k in missing], "warnings": []}
    doc, validation, manifest = (files[k][2] for k in (KIND_TRAJECTORY, KIND_VALIDATION, KIND_MANIFEST))

    expected = {"schema_version": mapper.TRAJECTORY_SCHEMA_VERSION,
                "validator_version": stage5_validator.VALIDATOR_VERSION}
    for key, value in expected.items():
        if doc.get(key) != value:
            reasons.append(f"Version incompatible : {key} = {doc.get(key)} (attendu {value}).")
    for name, other in (("validation", validation), ("manifest", manifest)):
        if other.get("validator_version") != stage5_validator.VALIDATOR_VERSION:
            reasons.append(f"Version incompatible ({name}) : validator_version = {other.get('validator_version')}.")
    if manifest.get("schema_version") != mapper.TRAJECTORY_SCHEMA_VERSION:
        reasons.append(f"Version incompatible (manifest) : schema_version = {manifest.get('schema_version')}.")
    if doc.get("agent_version") != mapper.TRAJECTORY_MAPPER_VERSION:
        warnings.append(f"Trajectory Mapper version {doc.get('agent_version')} (actuelle "
                        f"{mapper.TRAJECTORY_MAPPER_VERSION}).")
    if doc.get("prompt_sha256") != mapper.SPEC.prompt_sha256:
        warnings.append("Configuration produite avec d'autres consignes que celles de l'étape 5 actuelle.")

    status = doc.get("status")
    if status not in USABLE_STATUSES:
        reasons.append(f"Statut {status} non exploitable (attendu : {', '.join(USABLE_STATUSES)}).")
    if doc.get("analysis_complete") is not True or manifest.get("analysis_complete") is not True \
            or validation.get("analysis_complete") is not True:
        reasons.append("analysis_complete = false : étape 5 incomplète (PARTIAL) ou non disponible.")
    if validation.get("available") is not True:
        reasons.append("Fichier de validation : résultat non disponible.")
    errors = (doc.get("validation_error_count"), manifest.get("validation_error_count"), validation.get("error_count"))
    if errors != (0, 0, 0):
        reasons.append(f"validation_error_count = {errors[0]} (manifest {errors[1]}, validation {errors[2]}) : une "
                       "étape 5 avec des erreurs de validation n'est pas importée.")

    if doc.get("source_hashes") != manifest.get("source_hashes") or not doc.get("source_hashes"):
        reasons.append("Empreintes (source_hashes) différentes entre le document et le manifest.")
    if manifest.get("output_file") != KIND_TRAJECTORY or manifest.get("status") != status \
            or manifest.get("configuration_type") != doc.get("configuration_type") \
            or manifest.get("interview_id") != interview_id:
        reasons.append("Le manifest ne décrit pas ce document (statut, configuration, fichier ou entretien).")
    if validation.get("status") != status or validation.get("interview_id") != interview_id:
        reasons.append("Fichier de validation incohérent avec le document (statut ou entretien).")
    if reasons:
        return {"reasons": reasons, "warnings": warnings}

    reasons += _counts_problems(doc)
    if reasons:
        return {"reasons": reasons, "warnings": warnings}
    counts = trajectory._counts(doc)
    if any(manifest.get(k) != v for k, v in counts.items()):
        reasons.append("Comptes du manifest différents de ceux du document.")
    for key in ("validation_warning_count",):
        if manifest.get(key) != doc.get(key) or validation.get("warning_count") != doc.get(key):
            reasons.append("Nombre d'avertissements différent entre document, manifest et validation.")
    material = doc.get("material") or {}
    if any(manifest.get(k) != material.get(k) for k in MATERIAL_KEYS):
        reasons.append("Bilan de la préparation (matériau) différent entre document et manifest.")
    claims, criteria = doc["trajectory_claims"], doc["student_role_criteria"]
    recomputed = {
        "claim_count": len(claims), "kept_claim_count": sum(bool(c.get("usable_for_next_stages")) for c in claims),
        "rejected_claim_ids": [c["claim_id"] for c in claims if not c.get("usable_for_next_stages")],
        "requalified_claim_ids": [c["claim_id"] for c in claims if "model_claim_type" in c],
        "claims_needing_review": [c["claim_id"] for c in claims if c.get("needs_review")],
        "criterion_count": len(criteria),
        "rejected_criterion_ids": [c["criterion_id"] for c in criteria if not c.get("usable_for_next_stages")]}
    if any(validation.get(k) != v for k, v in recomputed.items()):
        reasons.append("Fichier de validation incohérent avec les affirmations (comptes, rejets, revues).")
    reasons += _object_problems(doc, validation)
    if (validation.get("warning_count") or validation.get("error_count")) and not doc.get("needs_review"):
        reasons.append("needs_review = false malgré des avertissements de validation.")
    return {"reasons": reasons, "warnings": warnings}


def _review_count(doc: dict) -> int:
    return sum(bool(c.get("needs_review")) for c in doc.get("trajectory_claims", []) + doc.get(
        "student_role_criteria", []) if c.get("usable_for_next_stages"))


def check_corpus(uploads: list[tuple[str, bytes]]) -> dict:
    """Contrôle les fichiers importés. Ne modifie rien, n'appelle rien.

    Renvoie {"rows": tableau par entretien, "usable": {interview_id: triplet}, "unrecognized": [...],
    "n_total", "n_usable", "mode", "notes"}.
    """
    groups, unrecognized = _group(uploads)
    rows, usable, notes = [], {}, []
    for interview_id in sorted(groups):
        group = groups[interview_id]
        reasons, chosen = [], {}
        for kind, entries in group.items():
            distinct = {_sha(data): (name, data, doc) for name, data, doc in entries}
            if len(distinct) > 1:
                reasons.append(f"Doublon ambigu : {len(distinct)} versions différentes de {kind} "
                               f"({', '.join(n for n, _, _ in entries)}).")
            elif len(entries) > 1:
                notes.append(f"{interview_id} : {kind} importé {len(entries)} fois à l'identique (dédoublonné).")
            chosen[kind] = next(iter(distinct.values()))
        checked = {"reasons": reasons, "warnings": []} if reasons else check_triplet(interview_id, chosen)
        doc = chosen.get(KIND_TRAJECTORY, (None, None, {}))[2]
        row = {
            "interview_id": interview_id,
            "status": STATUS_EXCLUDED if checked["reasons"] else STATUS_USABLE,
            "reasons": checked["reasons"], "warnings": checked["warnings"],
            "claims": doc.get("kept_claim_count"), "needs_review": _review_count(doc) if doc else None,
            "configuration": doc.get("configuration_type"),
            "files": {kind: {"imported_name": name, "sha256": _sha(data), "bytes": len(data)}
                      for kind, (name, data, _) in sorted(chosen.items())},
        }
        rows.append(row)
        if not checked["reasons"]:
            usable[interview_id] = {kind: chosen[kind] for kind in KINDS}
    n_usable = len(usable)
    return {"check_version": CORPUS_CHECK_VERSION, "rows": rows, "usable": usable, "unrecognized": unrecognized,
            "n_total": len(rows), "n_usable": n_usable, "mode": corpus_mode(n_usable), "notes": notes}


def corpus_id(checked: dict) -> str:
    """Identifiant du corpus : empreinte des fichiers exploitables et des exclusions (mêmes fichiers → même dossier)."""
    basis = [(r["interview_id"], r["status"], sorted((k, f["sha256"]) for k, f in r["files"].items()))
             for r in checked["rows"]]
    return "corpus_" + hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:12]


def public(checked: dict) -> dict:
    """Bilan de l'import, sans contenu (enregistré avec les sorties de l'étape 6)."""
    return {"check_version": checked["check_version"], "n_total": checked["n_total"], "n_usable": checked["n_usable"],
            "mode": checked["mode"], "interviews": checked["rows"], "unrecognized": checked["unrecognized"],
            "notes": checked["notes"]}


def install(checked: dict, corpus_dir: Path) -> None:
    """Recopie OCTET POUR OCTET les triplets exploitables dans <corpus>/stage5/<interview_id>/ (traçabilité)."""
    for interview_id, files in checked["usable"].items():
        target = corpus_dir / "stage5" / interview_id
        target.mkdir(parents=True, exist_ok=True)
        for kind, (_, data, _) in files.items():
            (target / kind).write_bytes(data)


def run_stage5_uploads(metadata: dict | None) -> list[tuple[str, bytes]]:
    """Triplets de l'étape 5 présents dans le run courant (lecture seule, aucun recalcul)."""
    uploads = []
    for info in eligible_files(metadata or {"files": []}):
        analysis_dir = Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
        if not (analysis_dir / KIND_TRAJECTORY).is_file():
            continue
        iid = info["ingestion"]["interview_id"]
        for kind in KINDS:
            path = analysis_dir / kind
            if path.is_file():
                uploads.append((f"{iid}_{kind}", path.read_bytes()))
    return uploads
