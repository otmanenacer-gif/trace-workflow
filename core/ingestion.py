"""Orchestration de l'ingestion : extraction -> structuration -> contrôle qualité.

Pour chaque entretien d'un run, écrit dans
data/outputs/<run_id>/interviews/<interview_id>/ :
- raw_text.txt              texte brut extrait (lignes = numéros cités par les tours) ;
- structured_transcript.json  tours de parole ;
- ingestion_report.json      rapport de qualité.

Un fichier en échec n'empêche pas le traitement des autres.
Le fichier source n'est ouvert qu'en lecture ; son SHA-256 est revérifié à la fin.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from core import config
from core.document_extractor import MIME_TYPES, ExtractionError, extract_document
from core.ingestion_validator import build_report
from core.run_manager import get_extension, save_metadata
from core.schemas import SCHEMA_VERSION, STATUS_FAIL, make_warning
from core.transcript_structurer import make_interview_id, structure_transcript

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_interview_id(filename: str, taken: set[str]) -> str:
    """Identifiant dérivé du nom de fichier, suffixé (_2, _3…) s'il est déjà pris."""
    base = make_interview_id(filename)
    candidate, counter = base, 2
    while candidate in taken:
        candidate = f"{base}_{counter}"
        counter += 1
    taken.add(candidate)
    return candidate


def ingest_file(source_path: Path, interview_id: str, output_dir: Path, original_name: str | None = None) -> dict:
    """Ingère un entretien et écrit ses trois fichiers de sortie.

    Renvoie un résumé : interview_id, status, turn_count, warning_count, output_dir.
    Ne lève pas d'exception pour un fichier illisible : le statut devient FAIL.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = get_extension(source_path.name)
    sha_before = sha256_file(source_path)
    source = {
        "filename": original_name or source_path.name,
        "stored_name": source_path.name,
        "format": fmt,
        "mime_type": MIME_TYPES.get(fmt),
        "size_bytes": source_path.stat().st_size,
        "sha256": sha_before,
    }
    processed_at = datetime.now().isoformat(timespec="seconds")

    warnings, turns, lines, extraction = [], [], [], None
    try:
        extraction = extract_document(source_path)
        warnings.extend(extraction["warnings"])
        structured = structure_transcript(extraction["pages"], interview_id, source_path.name)
        turns, lines = structured["turns"], structured["lines"]
        warnings.extend(structured["warnings"])
    except ExtractionError as exc:
        warnings.append(make_warning(exc.code, exc.message))
    except Exception as exc:  # erreur inattendue : isolée à ce fichier
        logger.exception("Erreur inattendue pendant l'ingestion de %s", source_path.name)
        warnings.append(make_warning("EXTRACTION_FAILED", f"Erreur inattendue : {type(exc).__name__}: {exc}"))

    raw_text = "\n".join(text for _, text in lines)
    source["sha256_verified_after_processing"] = sha256_file(source_path) == sha_before
    if not source["sha256_verified_after_processing"]:
        warnings.append(make_warning("SOURCE_MODIFIED"))

    report = build_report(
        interview_id=interview_id, source=source, processed_at=processed_at,
        extraction=extraction, raw_text=raw_text, lines=lines, turns=turns, warnings=warnings,
    )
    transcript = {
        "schema_version": SCHEMA_VERSION,
        "interview_id": interview_id,
        "source": {
            "filename": source["filename"],
            "stored_name": source["stored_name"],
            "sha256": sha_before,
            "format": fmt,
            "page_count": extraction["page_count"] if extraction else None,
        },
        "raw_text_file": config.RAW_TEXT_FILENAME,
        "status": report["status"],
        "turn_count": len(turns),
        "turns": turns,
        "warnings": report["warnings"],
    }

    (output_dir / config.RAW_TEXT_FILENAME).write_text(raw_text, encoding="utf-8")
    _write_json(output_dir / config.STRUCTURED_TRANSCRIPT_FILENAME, transcript)
    _write_json(output_dir / config.INGESTION_REPORT_FILENAME, report)
    logger.info("Ingestion %s : %s, %d tours, %d avertissements",
                interview_id, report["status"], len(turns), report["warning_count"])
    return {
        "interview_id": interview_id,
        "status": report["status"],
        "turn_count": len(turns),
        "warning_count": report["warning_count"],
        "output_dir": str(output_dir),
    }


def ingest_run(metadata: dict, progress_callback=None) -> dict:
    """Ingère tous les fichiers d'un run initialisé par run_manager.init_run.

    Met à jour et réécrit metadata.json : résumé d'ingestion par fichier et
    état de l'étape « Structuration des entretiens ». Les autres étapes
    restent inactives. `progress_callback(index, total, filename)` est
    appelé avant chaque fichier (utilisé par l'interface).
    """
    input_dir, output_dir = Path(metadata["input_dir"]), Path(metadata["output_dir"])
    taken: set[str] = set()
    files = metadata["files"]
    for index, info in enumerate(files):
        if progress_callback:
            progress_callback(index, len(files), info["stored_name"])
        interview_id = unique_interview_id(info["stored_name"], taken)
        info["ingestion"] = ingest_file(
            input_dir / info["stored_name"],
            interview_id,
            output_dir / config.INTERVIEWS_SUBDIR / interview_id,
            original_name=Path(info["original_name"]).name,
        )
        info["status"] = info["ingestion"]["status"]

    failed = sum(1 for f in files if f["ingestion"]["status"] == STATUS_FAIL)
    metadata["pipeline"][config.INGESTION_STEP] = "terminé" if not failed else f"terminé ({failed} échec(s))"
    metadata["status"] = "structuration terminée"
    metadata["ingested_at"] = datetime.now().isoformat(timespec="seconds")
    save_metadata(metadata, output_dir)
    return metadata
