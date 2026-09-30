"""Cache déterministe des analyses IA.

Une réponse du modèle n'est réutilisée que si TOUS les éléments suivants sont
identiques (ils forment la clé de cache) :
- SHA-256 du fichier source de l'entretien ;
- SHA-256 de la représentation exacte envoyée au modèle (transcript compact) ;
- nom et version de l'agent ;
- SHA-256 du prompt (consignes + gabarit du message) ;
- version et SHA-256 du schéma de sortie ;
- modèle demandé ;
- paramètres de génération influençant le résultat (effort, température).

Le cache est global (data/cache/analysis/<agent>/<clé>.json) : recharger la page
ou relancer un run sur les mêmes fichiers ne repaie pas l'appel. Il contient
des extraits d'entretiens : il n'est jamais versionné (.gitignore).
On y stocke la réponse VALIDÉE du modèle ; la validation des preuves, elle,
est recalculée à chaque utilisation (déterministe et gratuite).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from core import config

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = "1"

KEY_FIELDS = (
    "source_sha256", "transcript_sha256", "agent", "agent_version", "prompt_sha256",
    "schema_version", "schema_sha256", "model", "request_params",
)


def compute_cache_key(fields: dict) -> str:
    missing = [k for k in KEY_FIELDS if k not in fields]
    if missing:
        raise ValueError(f"Champs de clé de cache manquants : {missing}")
    canonical = json.dumps({k: fields[k] for k in KEY_FIELDS}, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, data: dict) -> None:
    """Écrit un JSON via un fichier temporaire puis un renommage (jamais de fichier à moitié écrit)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class AnalysisCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.CACHE_DIR)

    def path(self, agent: str, key: str) -> Path:
        return self.root / agent / f"{key}.json"

    def contains(self, agent: str, key: str) -> bool:
        return self.path(agent, key).is_file()

    def load(self, key_fields: dict, output_model: type[BaseModel]) -> dict | None:
        """Entrée de cache valide pour ces paramètres, sinon None (absente, corrompue ou différente)."""
        key = compute_cache_key(key_fields)
        path = self.path(key_fields["agent"], key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            stored = entry["key_fields"]
            if any(stored.get(k) != key_fields[k] for k in KEY_FIELDS):
                logger.warning("Entrée de cache %s incohérente : ignorée", path.name)
                return None
            output_model.model_validate(entry["output"])
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            logger.warning("Entrée de cache %s illisible ou invalide : ignorée", path.name)
            return None
        return entry

    def store(self, key_fields: dict, output: dict, call: dict) -> Path:
        key = compute_cache_key(key_fields)
        path = self.path(key_fields["agent"], key)
        write_json_atomic(path, {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_key": key,
            "key_fields": {k: key_fields[k] for k in KEY_FIELDS},
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "call": call,
            "output": output,
        })
        return path
