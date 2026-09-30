"""Gestion des runs d'analyse : identifiants, dossiers, fichiers et métadonnées.

Un run correspond à une exécution du pipeline. Il possède :
- un dossier d'entrée  : data/inputs/<run_id>/   (copies des fichiers importés)
- un dossier de sortie : data/outputs/<run_id>/  (problématique, métadonnées,
  et plus tard les sorties intermédiaires de chaque agent)

Les fichiers originaux importés ne sont jamais modifiés : on en écrit une copie.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from core import config

logger = logging.getLogger(__name__)


class TraceError(Exception):
    """Erreur métier avec un message compréhensible par l'utilisateur."""


def generate_run_id(now: datetime | None = None) -> str:
    """Crée un identifiant de run unique et lisible.

    Format : run_AAAAMMJJ_HHMMSS_xxxxxxxx (horodatage + 8 caractères aléatoires).
    La partie aléatoire évite les collisions entre deux runs lancés
    dans la même seconde.
    """
    now = now or datetime.now()
    return f"run_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"


def get_extension(filename: str) -> str:
    """Renvoie l'extension d'un nom de fichier, en minuscules et sans le point."""
    return Path(filename).suffix.lower().lstrip(".")


def validate_extension(filename: str) -> str:
    """Vérifie que le fichier a une extension autorisée et la renvoie.

    Lève TraceError sinon.
    """
    extension = get_extension(filename)
    if extension not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(e.upper() for e in config.ALLOWED_EXTENSIONS)
        raise TraceError(
            f"Le fichier « {filename} » n'est pas dans un format accepté. "
            f"Formats autorisés : {allowed}."
        )
    return extension


def create_run_dirs(
    run_id: str,
    inputs_root: Path = config.INPUTS_DIR,
    outputs_root: Path = config.OUTPUTS_DIR,
) -> tuple[Path, Path]:
    """Crée les dossiers d'entrée et de sortie d'un run.

    Refuse d'écraser un run existant (protection contre les collisions).
    Renvoie (dossier_entrees, dossier_sorties).
    """
    input_dir = Path(inputs_root) / run_id
    output_dir = Path(outputs_root) / run_id
    if input_dir.exists() or output_dir.exists():
        raise TraceError(f"Le run « {run_id} » existe déjà : création annulée.")
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    logger.info("Dossiers créés pour %s", run_id)
    return input_dir, output_dir


def save_input_file(filename: str, content: bytes, input_dir: Path) -> dict:
    """Enregistre une copie d'un fichier importé dans le dossier du run.

    Le nom est réduit à sa partie finale (aucun chemin n'est accepté).
    Si un fichier du même nom existe déjà, un suffixe numérique est ajouté.
    Renvoie les informations du fichier pour les métadonnées.
    """
    safe_name = Path(filename).name
    if not safe_name:
        raise TraceError("Nom de fichier vide ou invalide.")
    extension = validate_extension(safe_name)

    target = Path(input_dir) / safe_name
    counter = 1
    while target.exists():
        target = target.with_name(f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}")
        counter += 1

    target.write_bytes(content)
    return {
        "original_name": filename,
        "stored_name": target.name,
        "format": extension.upper(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "status": "chargé",
    }


def save_problematique(text: str, path: Path = config.PROBLEMATIQUE_FILE) -> Path:
    """Sauvegarde le texte de la problématique (peut être vide)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")
    return path


def load_problematique(path: Path = config.PROBLEMATIQUE_FILE) -> str:
    """Charge la problématique sauvegardée, ou une chaîne vide si absente."""
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def save_metadata(metadata: dict, output_dir: Path) -> Path:
    """Écrit les métadonnées du run au format JSON (UTF-8, lisible)."""
    path = Path(output_dir) / config.METADATA_FILENAME
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_metadata(output_dir: Path) -> dict:
    """Relit les métadonnées JSON d'un run."""
    path = Path(output_dir) / config.METADATA_FILENAME
    if not path.exists():
        raise TraceError(f"Aucune métadonnée trouvée dans {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def init_run(
    files: list[tuple[str, bytes]],
    problematique: str,
    inputs_root: Path = config.INPUTS_DIR,
    outputs_root: Path = config.OUTPUTS_DIR,
) -> dict:
    """Initialise un run complet à partir des fichiers importés.

    - vérifie qu'au moins un fichier est fourni et que tous sont autorisés ;
    - crée les dossiers du run ;
    - copie les fichiers et enregistre la problématique ;
    - écrit et renvoie les métadonnées.

    `files` est une liste de couples (nom_du_fichier, contenu_en_octets).
    """
    if not files:
        raise TraceError("Aucun fichier chargé : importez au moins un entretien.")
    # Validation avant toute écriture pour ne pas laisser de run incomplet
    for name, _ in files:
        validate_extension(name)

    run_id = generate_run_id()
    input_dir, output_dir = create_run_dirs(run_id, inputs_root, outputs_root)

    file_infos = [save_input_file(name, content, input_dir) for name, content in files]
    save_problematique(problematique, output_dir / config.RUN_PROBLEMATIQUE_FILENAME)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "problematique": problematique or "",
        "file_count": len(file_infos),
        "files": file_infos,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pipeline": {step: "inactive" for step in config.PIPELINE_STEPS},
        "status": "initialisé",
    }
    save_metadata(metadata, output_dir)
    logger.info("Run %s initialisé avec %d fichier(s)", run_id, len(file_infos))
    return metadata
