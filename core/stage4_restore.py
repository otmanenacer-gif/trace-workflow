"""Restauration de sorties de l'étape 4 déjà calculées (importées depuis des fichiers téléchargés).

Pourquoi : le stockage local de l'application (data/outputs, data/cache) n'est pas durable ; un
redéploiement l'efface. Après restauration de l'étape 3 (core/stage3_restore.py), l'étape 5 trouverait
une étape 4 absente et proposerait de la repayer. Les deux JSON téléchargés depuis l'interface suffisent :

    accountability_episodes.json, accountability_episode_validation.json

`check_restore_stage4` les valide sans rien modifier ; `restore_stage4` les recopie OCTET POUR OCTET dans le
dossier analysis/ de l'entretien (là où l'étape 4 les aurait écrits) avec un manifest de restauration, puis
met à jour metadata.json. Aucun appel LLM, aucune étape 4 : l'étape 5 lit ensuite ces fichiers exactement
comme des sorties normales (core/trajectory.py).

Contrôles (tous bloquants) :
- les deux fichiers sont présents, une seule fois chacun, en JSON valide, reconnus par leur CONTENU ;
- même interview_id dans les deux fichiers, égal à celui de l'entretien choisi ;
- étape 3 de cet entretien disponible et complète (calculée ou restaurée) ;
- versions compatibles : schéma de l'Accountability Episode Builder, constructeur de candidats et validateur
  identiques à ceux de TRACE (le recalcul ci-dessous en dépend) ; une autre version du prompt est signalée ;
- statut exploitable (SUCCESS, SUCCESS_WITH_WARNINGS, CACHED), `analysis_complete: true`, étape 3 COMPLETE ;
- `validation_error_count = 0` (et 0 erreur dans le fichier de validation) ;
- empreintes : les `source_hashes` enregistrés par l'étape 4 sont ceux des sorties de l'étape 3 actuellement
  installées (même transcription, mêmes pratiques, mêmes signaux, même audit des locuteurs) ;
- fichiers non altérés : les candidats sont RECALCULÉS (sans IA) depuis l'étape 3 installée, les épisodes
  RE-VALIDÉS par le validateur de l'étape 4 (citations revérifiées sur la transcription actuelle) ; candidats,
  épisodes annotés, pratiques sans marqueur, comptes et fichier de validation doivent coïncider exactement.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from agents import accountability_episode_builder as builder
from core import accountability, config, evidence_validator, trajectory
from core import accountability_candidates as candidates_mod
from core import accountability_episode_validator as episode_validator
from core import speaker_attribution_auditor as speaker_audit
from core.analysis import AUDITOR, INTERACTION, PRACTICE, STATUS_CACHED, STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS
from core.analysis import eligible_files
from core.analysis_cache import write_json_atomic
from core.run_manager import save_metadata
from core.stage3_restore import RestoreError

RESTORE_VERSION = "1.0"
RESTORE_MANIFEST_FILENAME = "stage4_restore_manifest.json"
EXPECTED = {
    config.ACCOUNTABILITY_EPISODES_FILENAME: "épisodes d'accountability",
    config.ACCOUNTABILITY_VALIDATION_FILENAME: "validation des épisodes",
}
USABLE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)
RESTORED_NOTICE = "Stage 4 restauré depuis fichiers — 0 appel API"
ANNOTATIONS = ("episode_id", "model_needs_review", "speaker_warnings", "validation_status", "usable_for_next_stages",
               "review_reasons")


def _kind(document) -> str | None:
    """Type d'un fichier d'après son CONTENU (jamais d'après son seul nom)."""
    if not isinstance(document, dict):
        return None
    if document.get("agent") == builder.AGENT_NAME and "episodes" in document:
        return config.ACCOUNTABILITY_EPISODES_FILENAME
    if "agent" not in document and {"issues", "stage3", "validator_version"} <= set(document):
        return config.ACCOUNTABILITY_VALIDATION_FILENAME
    return None


def _ingestion(metadata: dict, interview_id: str) -> dict:
    for info in eligible_files(metadata):
        if info["ingestion"]["interview_id"] == interview_id:
            return info
    raise RestoreError([f"Entretien {interview_id} introuvable ou non analysable dans ce run."])


def _normalized(data):
    return json.loads(json.dumps(data, ensure_ascii=False, sort_keys=True))


def _raw_episode(episode: dict) -> dict:
    """Épisode tel que le modèle l'a rendu (identifiants entiers), avant les annotations de TRACE."""
    raw = {k: v for k, v in episode.items() if k not in ANNOTATIONS and k not in ("evidence", "needs_review")}
    raw["evidence"] = [{k: v for k, v in e.items() if k != "validation"} for e in episode.get("evidence", [])]
    raw["needs_review"] = bool(episode.get("model_needs_review"))
    return raw


def _parse(uploads: list[tuple[str, bytes]]) -> tuple[dict, dict]:
    problems, files, documents = [], {}, {}
    for name, data in uploads:
        try:
            document = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError):
            problems.append(f"{name} : JSON invalide.")
            continue
        kind = _kind(document)
        if kind is None:
            problems.append(f"{name} : fichier non reconnu (ni épisodes d'accountability, ni validation des épisodes).")
        elif kind in files:
            problems.append(f"{EXPECTED[kind]} fourni deux fois ({files[kind][0]}, {name}).")
        else:
            files[kind], documents[kind] = (name, data), document
    for kind, label in EXPECTED.items():
        if kind not in files:
            problems.append(f"Fichier manquant : {kind} ({label}).")
    if problems:
        raise RestoreError(problems)
    return files, documents


def check_restore_stage4(metadata: dict, interview_id: str, uploads: list[tuple[str, bytes]]) -> dict:
    """Valide les fichiers importés pour un entretien. Ne modifie rien.

    Renvoie {"files": {nom attendu: (nom importé, octets)}, "documents": {...}, "warnings": [...]} ;
    lève RestoreError(problèmes) au moindre problème bloquant.
    """
    files, documents = _parse(uploads)
    episodes_doc = documents[config.ACCOUNTABILITY_EPISODES_FILENAME]
    validation_doc = documents[config.ACCOUNTABILITY_VALIDATION_FILENAME]
    ids = {kind: doc.get("interview_id") for kind, doc in documents.items()}
    if len(set(ids.values())) != 1:
        raise RestoreError(["Fichiers provenant de plusieurs entretiens : "
                            + ", ".join(f"{k} → {v}" for k, v in ids.items()) + "."])
    if episodes_doc.get("interview_id") != interview_id:
        raise RestoreError([f"Les fichiers concernent {episodes_doc.get('interview_id')}, pas l'entretien choisi "
                            f"({interview_id})."])

    info = _ingestion(metadata, interview_id)
    interview_dir = Path(info["ingestion"]["output_dir"])
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    stage3 = accountability.stage3_state(analysis_dir)
    if stage3["status"] != accountability.STAGE3_COMPLETE:
        raise RestoreError([f"Étape 3 de cet entretien {stage3['status']} : restaurez (ou relancez) d'abord l'étape 3 "
                            "complète, puis l'étape 4."])

    problems, warnings = [], []
    expected_versions = {"schema_version": builder.ACCOUNTABILITY_SCHEMA_VERSION,
                         "candidate_builder_version": candidates_mod.CANDIDATE_BUILDER_VERSION,
                         "payload_format": candidates_mod.PAYLOAD_FORMAT_VERSION,
                         "validator_version": episode_validator.VALIDATOR_VERSION}
    for key, expected in expected_versions.items():
        if episodes_doc.get(key) != expected:
            problems.append(f"Version incompatible : {key} = {episodes_doc.get(key)} (attendu {expected}).")
    if validation_doc.get("validator_version") != episode_validator.VALIDATOR_VERSION:
        problems.append(f"Validation : validator_version = {validation_doc.get('validator_version')} "
                        f"(attendu {episode_validator.VALIDATOR_VERSION}).")
    if episodes_doc.get("agent_version") != builder.ACCOUNTABILITY_BUILDER_VERSION:
        warnings.append(f"Accountability Episode Builder : version {episodes_doc.get('agent_version')} (version "
                        f"actuelle {builder.ACCOUNTABILITY_BUILDER_VERSION}).")
    if episodes_doc.get("prompt_sha256") != builder.SPEC.prompt_sha256:
        warnings.append("Épisodes construits avec d'autres consignes que les consignes actuelles de l'étape 4.")
    status = episodes_doc.get("status")
    if status not in USABLE_STATUSES:
        problems.append(f"Statut {status} non exploitable (attendu : {', '.join(USABLE_STATUSES)}).")
    if episodes_doc.get("analysis_complete") is not True:
        problems.append("analysis_complete = false (étape 4 incomplète).")
    if episodes_doc.get("stage3_status") != accountability.STAGE3_COMPLETE:
        problems.append(f"Étape 4 calculée sur une étape 3 {episodes_doc.get('stage3_status')} (attendu COMPLETE).")
    if validation_doc.get("status") != status or validation_doc.get("analysis_complete") is not True \
            or validation_doc.get("available") is not True:
        problems.append("Fichier de validation incohérent avec les épisodes (statut, analyse complète, disponibilité).")
    if episodes_doc.get("validation_error_count") != 0 or validation_doc.get("error_count") != 0:
        problems.append(f"validation_error_count = {episodes_doc.get('validation_error_count')} / "
                        f"{validation_doc.get('error_count')} : une étape 4 avec des erreurs de validation n'est pas "
                        "restaurable.")

    current = trajectory.stage3_hashes(interview_dir)
    recorded = episodes_doc.get("source_hashes") or {}
    changed = sorted(k for k, v in current.items() if recorded.get(k) != v)
    if changed:
        problems.append("Empreintes différentes de l'étape 3 actuellement installée (" + ", ".join(changed) + ") : "
                        "ces épisodes ont été calculés sur d'autres sorties de l'étape 3 ou une autre transcription.")
    if problems:
        raise RestoreError(problems)

    # Recalcul déterministe : candidats, puis re-validation des épisodes sur la transcription actuelle
    transcript = json.loads((interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
    practices = json.loads((analysis_dir / PRACTICE.output_filename).read_text(encoding="utf-8")).get("practices", [])
    signals = json.loads((analysis_dir / INTERACTION.output_filename).read_text(encoding="utf-8")).get("signals", [])
    audit_path = analysis_dir / AUDITOR.output_filename
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    speaker_warnings = speaker_audit.agent_warnings(audit, transcript) if audit and "items" in audit else {}
    built = candidates_mod.build_candidates(transcript, practices, signals, speaker_warnings)
    if _normalized(candidates_mod.public(built)) != _normalized(episodes_doc.get("candidates")):
        problems.append("Candidats différents de ceux que TRACE recalcule depuis l'étape 3 installée : fichier modifié "
                        "ou issu d'une autre version.")
    episodes = episodes_doc.get("episodes") or []
    turns = evidence_validator.index_turns(transcript)
    stale = sum(bool((e.get("validation") or {}).get("valid")) and not r["valid"]
                for ep in episodes for e, r in zip(ep.get("evidence", []), evidence_validator.validate_evidence(
                    ep.get("evidence", []), turns, interview_id)))
    if stale:
        problems.append(f"{stale} citation(s) déclarée(s) valide(s) ne correspondent plus au texte de l'entretien.")
    try:
        revalidated = episode_validator.validate_episodes([_raw_episode(e) for e in episodes], transcript, practices,
                                                          signals, built["candidates"], speaker_warnings)
    except (KeyError, TypeError, ValueError, AttributeError):
        raise RestoreError(problems + ["Épisodes illisibles (structure inattendue)."]) from None
    if _normalized(revalidated["episodes"]) != _normalized(episodes):
        problems.append("Épisodes différents de leur re-validation par TRACE (identifiants, citations, statuts ou "
                        "raisons de revue) : fichier modifié.")
    report = revalidated["report"]
    counts = accountability._counts(revalidated["episodes"])
    declared = {k: episodes_doc.get(k) for k in counts}
    if declared != counts:
        problems.append("Comptes du fichier d'épisodes incohérents avec les épisodes : "
                        + ", ".join(f"{k} {declared[k]} ≠ {v}" for k, v in counts.items() if declared[k] != v) + ".")
    for key in ("episode_count", "rejected_episode_ids", "episodes_needing_review", "unaddressed_candidate_ids"):
        if _normalized(validation_doc.get(key)) != _normalized(report[key]):
            problems.append(f"Fichier de validation incohérent avec les épisodes ({key}).")
    episode_issues = [i for i in validation_doc.get("issues", []) if i.get("code") not in (
        "PAYLOAD_OVER_THRESHOLD", "STAGE3_INCOMPLETE")]
    if _normalized(episode_issues) != _normalized(report["issues"]):
        problems.append("Anomalies du fichier de validation différentes de celles que TRACE recalcule.")
    unmarked_ids = [u.get("practice_id") for u in episodes_doc.get("unmarked_practices", [])]
    if unmarked_ids != built["unmarked_practice_ids"] or episodes_doc.get("unmarked_practice_count") != len(unmarked_ids):
        problems.append("Pratiques sans marqueur différentes de celles que TRACE recalcule.")
    if episodes_doc.get("candidate_count") != len(built["candidates"]):
        problems.append("candidate_count incohérent avec les candidats recalculés.")
    if problems:
        raise RestoreError(problems)
    return {"files": files, "documents": documents, "warnings": warnings, "counts": counts,
            "evidence_count": sum(len(e.get("evidence", [])) for e in episodes)}


def restore_stage4(metadata: dict, interview_id: str, uploads: list[tuple[str, bytes]]) -> dict:
    """Valide puis installe les sorties de l'étape 4 d'un entretien. 0 appel. Met à jour metadata.json."""
    checked = check_restore_stage4(metadata, interview_id, uploads)
    info = _ingestion(metadata, interview_id)
    analysis_dir = Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    restored_at = datetime.now().isoformat(timespec="seconds")
    document = checked["documents"][config.ACCOUNTABILITY_EPISODES_FILENAME]

    # Sorties périmées de l'étape 5 : elles ne décrivent pas ces épisodes
    for name in (config.STUDENT_TRAJECTORY_FILENAME, config.STUDENT_TRAJECTORY_MANIFEST_FILENAME,
                 config.STUDENT_TRAJECTORY_VALIDATION_FILENAME):
        (analysis_dir / name).unlink(missing_ok=True)
    for kind, (_, data) in checked["files"].items():
        (analysis_dir / kind).write_bytes(data)  # octet pour octet
    imported = {kind: {"imported_name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for kind, (name, data) in checked["files"].items()}
    manifest = {
        "manifest_version": "1.0", "interview_id": interview_id, "agent": builder.AGENT_NAME,
        "agent_version": document.get("agent_version"), "schema_version": document.get("schema_version"),
        "prompt_sha256": document.get("prompt_sha256"), "model": document.get("model"),
        "status": document.get("status"), "analysis_complete": True, "stage3_status": document.get("stage3_status"),
        "source_hashes": document.get("source_hashes"),
        "restored_from_file": imported[config.ACCOUNTABILITY_EPISODES_FILENAME]["imported_name"],
        "restored_sha256": imported[config.ACCOUNTABILITY_EPISODES_FILENAME]["sha256"], "restored_at": restored_at,
        "cache_hit": False, "llm_called": False, "api_calls": 0, "billed_this_run": False, "usage": None,
        "candidate_count": document.get("candidate_count"), **checked["counts"],
        "unmarked_practice_count": document.get("unmarked_practice_count"),
        "validation_error_count": document.get("validation_error_count"),
        "validation_warning_count": document.get("validation_warning_count"),
        "has_warnings": bool(document.get("validation_warning_count")),
        "output_file": config.ACCOUNTABILITY_EPISODES_FILENAME, "error": None,
    }
    write_json_atomic(analysis_dir / config.ACCOUNTABILITY_MANIFEST_FILENAME, manifest)
    write_json_atomic(analysis_dir / RESTORE_MANIFEST_FILENAME, {
        "restore_version": RESTORE_VERSION, "interview_id": interview_id, "restored_at": restored_at,
        "notice": RESTORED_NOTICE, "api_calls": 0, "files": imported,
        "source_hashes": document.get("source_hashes"), "evidence_count": checked["evidence_count"],
        "warnings": checked["warnings"]})

    info.pop("trajectory", None)  # résumé périmé de l'étape 5
    info["accountability"] = {
        "interview_id": interview_id, "analysis_dir": str(analysis_dir), "run_at": restored_at,
        **{k: manifest.get(k) for k in accountability.SUMMARY_KEYS},
        "estimated_input_tokens": 0, "over_single_call_threshold": False, "chunk_count": document.get("chunk_count"),
        "duration_seconds": None, "restored": True, "restored_at": restored_at,
        "restore_warnings": checked["warnings"],
    }
    metadata["pipeline"][config.ACCOUNTABILITY_STEP] = accountability._step_state(metadata)
    metadata["pipeline"][config.TRAJECTORY_STEP] = trajectory._step_state(metadata)
    metadata["last_stage4_restore"] = {"interview_id": interview_id, "restored_at": restored_at, "api_calls": 0,
                                       "notice": RESTORED_NOTICE}
    save_metadata(metadata, Path(metadata["output_dir"]))
    return metadata
