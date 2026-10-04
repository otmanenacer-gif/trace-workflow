"""Restauration de sorties de l'étape 3 déjà calculées (importées depuis des fichiers téléchargés).

Pourquoi : le stockage local de l'application (data/outputs, data/cache) n'est pas durable ; sur
Streamlit Cloud, un redéploiement l'efface. Un entretien réimporté crée alors un nouveau run sans
sortie de l'étape 3, et TRACE proposerait de refaire jouer toutes ses tâches d'agent. Les quatre JSON téléchargés
depuis l'interface suffisent pourtant à l'étape 4 :

    practice_extractor.json, interaction_signals.json, evidence_validation.json,
    speaker_attribution_audit.json

`check_restore` les valide sans rien modifier ; `restore_stage3` les installe dans le dossier analysis/ de
l'entretien (là où l'étape 3 les aurait écrits) avec des manifests de restauration, puis met à jour metadata.json.
Aucun appel LLM, ni étape 3, ni audit des locuteurs : l'étape 4 lit ensuite ces fichiers exactement comme des
sorties normales (core/accountability.py).

Version canonique : avant installation, la sélectivité déterministe ACTUELLE de l'étape 3 (celle que réapplique
`trace_local.py refilter` : core/practice_selectivity.py, core/signal_selectivity.py) et la validation des preuves
sont réappliquées aux objets importés (`reselect`). Des fichiers téléchargés AVANT un `refilter` ne peuvent donc
jamais réinstaller des pratiques ou des signaux que la sélectivité écarte : ils donnent les mêmes objets que le
run refiltré. Des fichiers déjà à jour sont installés octet pour octet.

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
from core import speaker_attribution_auditor as speaker_audit
from core.analysis import (AGENTS, AUDITOR, INTERACTION, PRACTICE, STATUS_CACHED, STATUS_SUCCESS,
                           STATUS_SUCCESS_WITH_WARNINGS, _agent_summary, _issue_count, _step_state, eligible_files,
                           postprocess_for, prepare_interview, with_speaker_warnings)
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


# --- Version canonique : sélectivité actuelle réappliquée (comme `refilter`) --------------------------------

SELECTIVITY_KEYS = {PRACTICE.name: ("practice_id", "practices", "set_aside_practices", "practice_selectivity"),
                    INTERACTION.name: ("signal_id", "signals", "set_aside_signals", "selectivity")}


def _unvalidated(item: dict, id_key: str) -> dict:
    """Objet tel que l'agent l'a produit : sans les annotations du validateur (identifiant, validation, revue)."""
    item = {k: v for k, v in item.items() if k not in (id_key, "needs_review", "review_reasons")}
    item["evidence"] = [{k: v for k, v in e.items() if k != "validation"} for e in item.get("evidence", [])]
    return item


def reselect(spec, document: dict, prepared) -> tuple[dict, dict] | None:
    """Réapplique au document importé d'un agent la sélectivité déterministe actuelle puis la validation des preuves,
    exactement comme `refilter` (core/analysis.postprocess_for). Renvoie (document canonique, rapport de validation),
    ou None si le document est déjà à jour (aucun objet écarté, objets inchangés)."""
    id_key, noun, aside_key, summary_key = SELECTIVITY_KEYS[spec.name]
    items = [_unvalidated(i, id_key) for i in document[spec.items_key]]
    kept, doc_extra, _ = postprocess_for(spec, prepared)(items)
    validated = evidence_validator.validate_agent_output(spec.name, kept, prepared.transcript, spec.id_letter,
                                                         prepared.speaker_warnings)
    if not doc_extra[aside_key] and validated["items"] == document[spec.items_key]:
        return None
    aside = [*(document.get(aside_key) or []), *doc_extra[aside_key]]  # écartés à l'import + nouvellement écartés
    summary = dict(doc_extra[summary_key])
    summary.update({f"{noun}_received": (document.get(summary_key) or {}).get(f"{noun}_received", len(items)),
                    f"{noun}_set_aside": len(aside)})
    reasons = [a.get("set_aside_reason") for a in aside]
    summary["set_aside_by_reason"] = {r: reasons.count(r) for r in dict.fromkeys(reasons)}
    report = validated["report"]
    canonical = {**document, **doc_extra, aside_key: aside, summary_key: summary,
                 "item_count": len(validated["items"]), "needs_review_count": len(report["objects_needing_review"]),
                 "invalid_evidence_count": report["invalid_evidence_count"],
                 "validation_issue_count": _issue_count(report)}
    canonical.pop(spec.items_key)
    canonical[spec.items_key] = validated["items"]  # les objets en dernier, comme une sortie normale
    return canonical, report


def canonical_stage3(info: dict, documents: dict) -> dict:
    """Documents de l'étape 3 tels qu'ils doivent être installés : {nom attendu: document canonique} pour ceux que la
    sélectivité actuelle modifie, evidence_validation.json recalculé en conséquence. Vide si tout est à jour."""
    prepared = prepare_interview(info["ingestion"])
    audit = documents[AUDITOR.output_filename]
    warnings = speaker_audit.agent_warnings(audit, prepared.transcript) if "items" in audit else {}
    prepared = with_speaker_warnings(prepared, warnings)
    changed, reports = {}, {}
    for spec in AGENTS:
        result = reselect(spec, documents[spec.output_filename], prepared)
        if result is not None:
            changed[spec.output_filename], reports[spec.name] = result
    if changed:
        validation = json.loads(json.dumps(documents[config.EVIDENCE_VALIDATION_FILENAME]))
        for name, report in reports.items():
            validation["agents"][name] = {**validation["agents"].get(name, {}), **report}
        available = [a for a in validation["agents"].values() if a.get("available")]
        validation.update(total_evidence=sum(a["evidence_count"] for a in available),
                          total_invalid_evidence=sum(a["invalid_evidence_count"] for a in available),
                          total_objects_needing_review=sum(len(a["objects_needing_review"]) for a in available))
        changed[config.EVIDENCE_VALIDATION_FILENAME] = validation
    return changed


def _restore_manifest(spec, document: dict, imported_name: str, data: bytes, info: dict) -> dict:
    return {
        "manifest_version": "1.0", "interview_id": document["interview_id"],
        "agent": spec.name, "agent_version": document.get("agent_version"),
        "schema_version": document.get("schema_version"), "model": document.get("model"),
        "status": document.get("status"), "analysis_complete": True,
        "restored_from_file": imported_name, "restored_sha256": hashlib.sha256(data).hexdigest(),
        "reselected": spec.output_filename in info["reselected"], "restored_at": info["restored_at"],
        "cache_hit": False, "api_calls": 0, "billed_this_run": False,
        "usage": None, "item_count": document.get("item_count"),
        "invalid_evidence_count": document.get("invalid_evidence_count"),
        "needs_review_count": document.get("needs_review_count"),
        "validation_issue_count": document.get("validation_issue_count"),
        "has_warnings": bool(document.get("validation_issue_count")), "output_file": spec.output_filename,
        "chunking": document.get("chunking"), "selectivity": document.get("selectivity"),
        "error": None,
    }


def restore_stage3(metadata: dict, interview_id: str, uploads: list[tuple[str, bytes]]) -> dict:
    """Valide puis installe les sorties de l'étape 3 d'un entretien, dans leur version canonique (sélectivité actuelle
    réappliquée, voir `reselect`). 0 appel. Met à jour metadata.json."""
    checked = check_restore(metadata, interview_id, uploads)
    info = _ingestion(metadata, interview_id)
    analysis_dir = Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    restored_at = datetime.now().isoformat(timespec="seconds")
    imported = dict(checked["files"])
    canonical = canonical_stage3(info, checked["documents"])
    for kind, document in canonical.items():  # sérialisés comme une sortie normale (write_json_atomic)
        checked["documents"][kind] = document
        checked["files"][kind] = (imported[kind][0], json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"))
    context = {"restored_at": restored_at, "reselected": set(canonical)}

    # Sorties périmées de l'étape 4 : elles ne décrivent pas ces fichiers
    for name in (config.ACCOUNTABILITY_EPISODES_FILENAME, config.ACCOUNTABILITY_MANIFEST_FILENAME,
                 config.ACCOUNTABILITY_VALIDATION_FILENAME):
        (analysis_dir / name).unlink(missing_ok=True)
    # Les quatre fichiers : octet pour octet s'ils sont à jour, sinon leur version canonique
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
        "files": {kind: {"imported_name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                         "imported_sha256": hashlib.sha256(imported[kind][1]).hexdigest(),
                         "reselected": kind in canonical}
                  for kind, (name, data) in checked["files"].items()},
        "reselection": {kind: doc.get("practice_selectivity") or doc.get("selectivity")
                        for kind, doc in canonical.items() if kind != config.EVIDENCE_VALIDATION_FILENAME},
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
        "reselected": sorted(canonical),
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
