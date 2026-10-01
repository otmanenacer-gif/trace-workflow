"""Restauration de sorties de l'étape 3 déjà calculées (importées depuis des fichiers téléchargés).

Pourquoi : le stockage local de l'application (data/outputs, data/cache) n'est pas durable ; sur
Streamlit Cloud, un redéploiement l'efface. Un entretien réimporté crée alors un nouveau run sans
sortie de l'étape 3, et TRACE proposerait de repayer tous les appels. Les quatre JSON téléchargés
depuis l'interface suffisent pourtant à l'étape 4 :

    practice_extractor.json, interaction_signals.json, evidence_validation.json,
    speaker_attribution_audit.json

`check_restore` les valide sans rien modifier ; `restore_stage3` les recopie OCTET POUR OCTET dans
le dossier analysis/ de l'entretien (là où l'étape 3 les aurait écrits) avec des manifests de
restauration, puis met à jour metadata.json. Aucun appel LLM, ni étape 3, ni audit des locuteurs :
l'étape 4 lit ensuite ces fichiers exactement comme des sorties normales (core/accountability.py).

Contrôles (tous bloquants) :
- les quatre fichiers sont présents, une seule fois chacun, en JSON valide, reconnus par leur contenu
  (le nom de fichier peut porter un préfixe, ex. « OTMANE_NACER_practice_extractor.json ») ;
- même interview_id dans les quatre fichiers, égal à celui de l'entretien choisi ;
- Practice Extractor et Interaction Reader : agent et version de schéma attendus, statut exploitable
  (SUCCESS, SUCCESS_WITH_WARNINGS, CACHED), jamais `analysis_complete: false` ni PARTIAL / FAILED ;
- transcription : l'empreinte du structured_transcript.json enregistrée par l'audit des locuteurs doit
  être celle de l'entretien choisi (même fichier source, même segmentation) ;
- citations : chaque citation est revérifiée sur la transcription ACTUELLE ; un tour inexistant ou une
  citation déclarée valide qui ne l'est plus est une incohérence critique ;
- evidence_validation.json : aucune anomalie critique (tour inexistant ou d'un autre entretien), au plus
  MAX_INVALID_EVIDENCE_RATIO de citations invalides, totaux cohérents avec les deux sorties.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from core import accountability, config, evidence_validator
from core.analysis import (AGENTS, AUDITOR, INTERACTION, PRACTICE, STATUS_CACHED, STATUS_SUCCESS,
                           STATUS_SUCCESS_WITH_WARNINGS, _agent_summary, _step_state, eligible_files)
from core.analysis_cache import write_json_atomic
from core.run_manager import save_metadata

RESTORE_VERSION = "1.0"
RESTORE_MANIFEST_FILENAME = "stage3_restore_manifest.json"
EXPECTED = {
    PRACTICE.output_filename: "Practice Extractor",
    INTERACTION.output_filename: "Interaction Signal Reader",
    config.EVIDENCE_VALIDATION_FILENAME: "validation des preuves",
    AUDITOR.output_filename: "audit des locuteurs",
}
USABLE_STATUSES = (STATUS_SUCCESS, STATUS_SUCCESS_WITH_WARNINGS, STATUS_CACHED)
CRITICAL_ISSUE_CODES = frozenset({"UNKNOWN_TURN_ID", "FOREIGN_INTERVIEW_TURN", "UNKNOWN_RANGE_TURN"})
MAX_INVALID_EVIDENCE_RATIO = 0.10
RESTORED_NOTICE = "Stage 3 restauré depuis fichiers — 0 appel API"


class RestoreError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(" ; ".join(problems))


def _kind(document) -> str | None:
    """Type d'un fichier d'après son CONTENU (jamais d'après son seul nom)."""
    if not isinstance(document, dict):
        return None
    if document.get("agent") == PRACTICE.name and "practices" in document:
        return PRACTICE.output_filename
    if document.get("agent") == INTERACTION.name and "signals" in document:
        return INTERACTION.output_filename
    if "agents" in document and "total_invalid_evidence" in document:
        return config.EVIDENCE_VALIDATION_FILENAME
    if "items" in document and "candidate_count" in document and "review_count" in document:
        return AUDITOR.output_filename
    return None


def _ingestion(metadata: dict, interview_id: str) -> dict:
    for info in eligible_files(metadata):
        if info["ingestion"]["interview_id"] == interview_id:
            return info
    raise RestoreError([f"Entretien {interview_id} introuvable ou non analysable dans ce run."])


def check_restore(metadata: dict, interview_id: str, uploads: list[tuple[str, bytes]]) -> dict:
    """Valide les fichiers importés pour un entretien. Ne modifie rien.

    Renvoie {"files": {nom attendu: (nom importé, octets)}, "documents": {...}, "warnings": [...]} ;
    lève RestoreError(problèmes) au moindre problème bloquant.
    """
    problems, warnings = [], []
    files, documents = {}, {}
    for name, data in uploads:
        try:
            document = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError):
            problems.append(f"{name} : JSON invalide.")
            continue
        kind = _kind(document)
        if kind is None:
            problems.append(f"{name} : fichier non reconnu (ni pratiques, ni signaux, ni validation, ni audit).")
        elif kind in files:
            problems.append(f"{EXPECTED[kind]} fourni deux fois ({files[kind][0]}, {name}).")
        else:
            files[kind], documents[kind] = (name, data), document
    for kind, label in EXPECTED.items():
        if kind not in files:
            problems.append(f"Fichier manquant : {kind} ({label}).")
    if problems:
        raise RestoreError(problems)

    ids = {kind: doc.get("interview_id") for kind, doc in documents.items()}
    if len(set(ids.values())) != 1:
        raise RestoreError(["Fichiers provenant de plusieurs entretiens : "
                            + ", ".join(f"{k} → {v}" for k, v in ids.items()) + "."])
    if ids[PRACTICE.output_filename] != interview_id:
        raise RestoreError([f"Les fichiers concernent {ids[PRACTICE.output_filename]}, pas l'entretien choisi "
                            f"({interview_id})."])

    info = _ingestion(metadata, interview_id)
    transcript_path = Path(info["ingestion"]["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    current_sha = hashlib.sha256(transcript_path.read_bytes()).hexdigest()

    for spec in (PRACTICE, INTERACTION):
        doc = documents[spec.output_filename]
        if doc.get("schema_version") != spec.schema_version:
            problems.append(f"{spec.label} : schéma {doc.get('schema_version')} incompatible "
                            f"(attendu {spec.schema_version}).")
        if doc.get("agent_version") != spec.version:
            warnings.append(f"{spec.label} : version {doc.get('agent_version')} (version actuelle {spec.version}).")
        if doc.get("status") not in USABLE_STATUSES:
            problems.append(f"{spec.label} : statut {doc.get('status')} non exploitable "
                            f"(attendu : {', '.join(USABLE_STATUSES)}).")
        if doc.get("analysis_complete", True) is not True:
            problems.append(f"{spec.label} : analysis_complete = false (analyse incomplète).")

    audit = documents[AUDITOR.output_filename]
    recorded = audit.get("structured_transcript_sha256")
    if recorded is None:
        warnings.append("Audit des locuteurs sans empreinte de transcription : cohérence vérifiée par les seules citations.")
    elif recorded != current_sha:
        problems.append("Les sorties ont été calculées sur une autre transcription structurée que celle de l'entretien "
                        "choisi (autre fichier ou autre segmentation des tours) : relancez l'étape 3 ou importez les "
                        "sorties correspondant à ce fichier.")

    # Citations revérifiées sur la transcription actuelle (jamais corrigées)
    turns = evidence_validator.index_turns(transcript)
    critical = changed = total = invalid = 0
    for spec in (PRACTICE, INTERACTION):
        for item in documents[spec.output_filename][spec.items_key]:
            evidence = item.get("evidence", [])
            for e, result in zip(evidence, evidence_validator.validate_evidence(evidence, turns, interview_id)):
                total += 1
                invalid += not result["valid"]
                critical += result["code"] in CRITICAL_ISSUE_CODES
                changed += bool((e.get("validation") or {}).get("valid")) and not result["valid"]
    if critical:
        problems.append(f"{critical} citation(s) renvoient à des tours absents de cet entretien.")
    if changed:
        problems.append(f"{changed} citation(s) déclarée(s) valide(s) ne correspondent plus au texte de l'entretien.")

    validation = documents[config.EVIDENCE_VALIDATION_FILENAME]
    agents = validation.get("agents") or {}
    for spec in AGENTS:
        report = agents.get(spec.name) or {}
        if report.get("available") is False or not report:
            problems.append(f"evidence_validation.json : pas de validation pour {spec.label}.")
        bad = [i["code"] for i in report.get("issues", []) if i.get("code") in CRITICAL_ISSUE_CODES]
        if bad:
            problems.append(f"evidence_validation.json : {len(bad)} anomalie(s) critique(s) pour {spec.label} "
                            f"({', '.join(sorted(set(bad)))}).")
    declared = validation.get("total_invalid_evidence") or 0
    documented = sum(documents[s.output_filename].get("invalid_evidence_count") or 0 for s in AGENTS)
    if declared != documented:
        problems.append(f"evidence_validation.json ({declared} citation(s) invalide(s)) ne correspond pas aux sorties "
                        f"importées ({documented}).")
    if total and max(invalid, declared) / total > MAX_INVALID_EVIDENCE_RATIO:
        problems.append(f"Trop de citations invalides ({max(invalid, declared)}/{total}) pour une analyse exploitable.")
    if problems:
        raise RestoreError(problems)
    return {"files": files, "documents": documents, "warnings": warnings, "transcript_sha256": current_sha,
            "evidence_count": total, "invalid_evidence_count": invalid}


def _restore_manifest(spec, document: dict, imported_name: str, data: bytes, info: dict) -> dict:
    return {
        "manifest_version": "1.0", "interview_id": document["interview_id"],
        "agent": spec.name, "agent_version": document.get("agent_version"),
        "schema_version": document.get("schema_version"), "model": document.get("model"),
        "status": document.get("status"), "analysis_complete": True,
        "restored_from_file": imported_name, "restored_sha256": hashlib.sha256(data).hexdigest(),
        "restored_at": info["restored_at"], "cache_hit": False, "api_calls": 0, "billed_this_run": False,
        "usage": None, "item_count": document.get("item_count"),
        "invalid_evidence_count": document.get("invalid_evidence_count"),
        "needs_review_count": document.get("needs_review_count"),
        "validation_issue_count": document.get("validation_issue_count"),
        "has_warnings": bool(document.get("validation_issue_count")), "output_file": spec.output_filename,
        "chunking": document.get("chunking"), "selectivity": document.get("selectivity"),
        "error": None,
    }


def restore_stage3(metadata: dict, interview_id: str, uploads: list[tuple[str, bytes]]) -> dict:
    """Valide puis installe les sorties de l'étape 3 d'un entretien. 0 appel. Met à jour metadata.json."""
    checked = check_restore(metadata, interview_id, uploads)
    info = _ingestion(metadata, interview_id)
    analysis_dir = Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    restored_at = datetime.now().isoformat(timespec="seconds")
    context = {"restored_at": restored_at}

    # Sorties périmées de l'étape 4 : elles ne décrivent pas ces fichiers
    for name in (config.ACCOUNTABILITY_EPISODES_FILENAME, config.ACCOUNTABILITY_MANIFEST_FILENAME,
                 config.ACCOUNTABILITY_VALIDATION_FILENAME):
        (analysis_dir / name).unlink(missing_ok=True)
    # Les quatre fichiers, octet pour octet
    for kind, (_, data) in checked["files"].items():
        (analysis_dir / kind).write_bytes(data)
    docs = checked["documents"]
    manifests = {}
    for spec in AGENTS:
        imported_name, data = checked["files"][spec.output_filename]
        manifests[spec.name] = _restore_manifest(spec, docs[spec.output_filename], imported_name, data, context)
        write_json_atomic(analysis_dir / spec.manifest_filename, manifests[spec.name])
    audit = docs[AUDITOR.output_filename]
    write_json_atomic(analysis_dir / AUDITOR.manifest_filename, {
        "manifest_version": "1.0", "interview_id": interview_id, "agent": AUDITOR.name, "status": audit.get("status"),
        "restored_from_file": checked["files"][AUDITOR.output_filename][0], "restored_at": restored_at,
        "llm_called": False, "api_calls": 0, "billed_this_run": False, "usage": None,
        "candidate_count": audit.get("candidate_count"), "review_count": audit.get("review_count"), "error": None})
    write_json_atomic(analysis_dir / RESTORE_MANIFEST_FILENAME, {
        "restore_version": RESTORE_VERSION, "interview_id": interview_id, "restored_at": restored_at,
        "notice": RESTORED_NOTICE, "api_calls": 0,
        "structured_transcript_sha256": checked["transcript_sha256"],
        "files": {kind: {"imported_name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for kind, (name, data) in checked["files"].items()},
        "evidence_recheck": {"evidence_count": checked["evidence_count"],
                             "invalid_evidence_count": checked["invalid_evidence_count"]},
        "warnings": checked["warnings"]})

    agents = {}
    for spec in AGENTS:
        summary = _agent_summary(manifests[spec.name])
        cues = docs[spec.output_filename].get("non_use_cues")
        if cues:
            summary["non_use_cues"] = {"cue_turn_count": cues.get("cue_turn_count"),
                                       "uncovered_count": len(cues.get("uncovered_turn_ids") or [])}
        agents[spec.name] = summary
    info.pop("accountability", None)  # résumé périmé de l'étape 4
    info["analysis"] = {
        "interview_id": interview_id, "analysis_dir": str(analysis_dir), "analyzed_at": restored_at,
        "restored": True, "restored_at": restored_at, "restore_warnings": checked["warnings"],
        "invalid_evidence_count": docs[config.EVIDENCE_VALIDATION_FILENAME].get("total_invalid_evidence") or 0,
        "speaker_audit": {"status": audit.get("status"), "audit_mode": audit.get("audit_mode"), "llm_called": False,
                          "cache_hit": False, "candidate_count": audit.get("candidate_count"),
                          "review_count": audit.get("review_count"), "api_calls": 0, "billed_this_run": False,
                          "usage": None, "error": None,
                          "has_warnings": bool(audit.get("error_count") or audit.get("warning_count"))},
        "agents": agents,
    }
    for spec in AGENTS:
        metadata["pipeline"][spec.pipeline_step] = _step_state(metadata, spec.name)
    metadata["pipeline"][config.ACCOUNTABILITY_STEP] = accountability._step_state(metadata)
    metadata["last_restore"] = {"interview_id": interview_id, "restored_at": restored_at, "api_calls": 0,
                                "notice": RESTORED_NOTICE}
    save_metadata(metadata, Path(metadata["output_dir"]))
    return metadata
