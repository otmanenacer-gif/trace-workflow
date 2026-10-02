"""Configuration centrale de TRACE.

Tous les chemins sont construits avec pathlib à partir de la racine du projet,
ce qui les rend indépendants du système d'exploitation.
"""

import os
from pathlib import Path

# Racine du projet (dossier contenant app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"      # copies des fichiers importés, par run
OUTPUTS_DIR = DATA_DIR / "outputs"    # métadonnées et futures sorties, par run
LOGS_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Cache des analyses IA (réutilisé d'un run à l'autre, jamais versionné)
CACHE_DIR = DATA_DIR / "cache" / "analysis"

# Fichier local de réglages facultatifs (tailles des blocs) — jamais versionné
ENV_FILE = PROJECT_ROOT / ".env"

# Problématique courante, sauvegardée depuis l'interface
PROBLEMATIQUE_FILE = DATA_DIR / "problematique.txt"

# Formats de fichiers acceptés pour le corpus (en minuscules, sans le point)
ALLOWED_EXTENSIONS = ("pdf", "docx", "txt")

# Étapes futures du pipeline (inactives dans cette version)
PIPELINE_STEPS = (
    "Structuration des entretiens",
    "Extraction des pratiques",
    "Analyse interactionnelle",
    "Construction des épisodes d'accountability",
    "Configuration et trajectoire intra-entretien",
    "Comparaison transversale",
    "Construction des régimes",
    "Challenger",
    "Vérification des preuves",
    "Rédaction du rapport",
)

METADATA_FILENAME = "metadata.json"
RUN_PROBLEMATIQUE_FILENAME = "problematique.txt"

# Ingestion : sorties par entretien dans data/outputs/<run_id>/interviews/<interview_id>/
INGESTION_STEP = PIPELINE_STEPS[0]
INTERVIEWS_SUBDIR = "interviews"
RAW_TEXT_FILENAME = "raw_text.txt"
STRUCTURED_TRANSCRIPT_FILENAME = "structured_transcript.json"
INGESTION_REPORT_FILENAME = "ingestion_report.json"

# Étape 3 : analyse IA, sorties dans data/outputs/<run_id>/interviews/<interview_id>/analysis/
PRACTICE_STEP = PIPELINE_STEPS[1]
INTERACTION_STEP = PIPELINE_STEPS[2]
ANALYSIS_SUBDIR = "analysis"
EVIDENCE_VALIDATION_FILENAME = "evidence_validation.json"

# Étape 4 : épisodes d'accountability, mêmes dossiers analysis/ (consomme les sorties de l'étape 3)
ACCOUNTABILITY_STEP = PIPELINE_STEPS[3]
ACCOUNTABILITY_EPISODES_FILENAME = "accountability_episodes.json"
ACCOUNTABILITY_MANIFEST_FILENAME = "accountability_episode_manifest.json"
ACCOUNTABILITY_VALIDATION_FILENAME = "accountability_episode_validation.json"

# Étape 5 : configuration et trajectoire intra-entretien (consomme les sorties de l'étape 4), mêmes dossiers
TRAJECTORY_STEP = PIPELINE_STEPS[4]
STUDENT_TRAJECTORY_FILENAME = "student_trajectory.json"
STUDENT_TRAJECTORY_VALIDATION_FILENAME = "student_trajectory_validation.json"
STUDENT_TRAJECTORY_MANIFEST_FILENAME = "student_trajectory_manifest.json"

# Étape 6 : comparaison inter-entretiens. Elle consomme UNIQUEMENT les triplets de l'étape 5 (importés depuis des
# fichiers ou repris du run courant) ; sorties par corpus dans data/outputs/cross_interview/<corpus_id>/analysis/
CROSS_INTERVIEW_STEP = PIPELINE_STEPS[5]
CROSS_INTERVIEW_DIR = OUTPUTS_DIR / "cross_interview"
CROSS_INTERVIEW_CORPUS_FILENAME = "cross_interview_corpus.json"
CROSS_INTERVIEW_COMPARISON_FILENAME = "cross_interview_comparison.json"
CROSS_INTERVIEW_VALIDATION_FILENAME = "cross_interview_validation.json"
CROSS_INTERVIEW_MANIFEST_FILENAME = "cross_interview_manifest.json"


def load_env_file(path: Path | None = None) -> list[str]:
    """Charge les variables d'un fichier .env (KEY=VALUE) dans l'environnement.

    Une variable déjà définie dans l'environnement n'est jamais écrasée ;
    une valeur vide est ignorée. Renvoie les noms des variables chargées (jamais leurs valeurs).
    """
    path = Path(path or ENV_FILE)
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
