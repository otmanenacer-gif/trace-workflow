"""Configuration centrale de TRACE.

Tous les chemins sont construits avec pathlib à partir de la racine du projet,
ce qui les rend indépendants du système d'exploitation.
"""

from pathlib import Path

# Racine du projet (dossier contenant app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"      # copies des fichiers importés, par run
OUTPUTS_DIR = DATA_DIR / "outputs"    # métadonnées et futures sorties, par run
LOGS_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

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
    "Trajectoires individuelles",
    "Comparaison transversale",
    "Construction des régimes",
    "Challenger",
    "Vérification des preuves",
    "Rédaction du rapport",
)

METADATA_FILENAME = "metadata.json"
RUN_PROBLEMATIQUE_FILENAME = "problematique.txt"
