"""TRACE — interface Streamlit (squelette technique, sans IA).

Lancement : streamlit run app.py
"""

import json
import logging
import time

import streamlit as st

from core import config
from core.run_manager import TraceError, get_extension, init_run, load_problematique, save_problematique

config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=config.LOGS_DIR / "trace.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    encoding="utf-8",
)

# Étapes simulées lors de l'initialisation (aucun appel IA)
SIMULATED_STEPS = (
    "Vérification des fichiers",
    "Création du dossier du run",
    "Copie du corpus et de la problématique",
    "Écriture des métadonnées",
)


def format_size(size_bytes: int) -> str:
    """Affiche une taille de fichier de façon lisible."""
    if size_bytes < 1024:
        return f"{size_bytes} o"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} Ko"
    return f"{size_bytes / 1024 ** 2:.1f} Mo"


st.set_page_config(page_title="TRACE", layout="wide")

# 1. Titre et 2. introduction
st.title("TRACE")
st.subheader("Analyse ethnométhodologique des usages étudiants des IAG")
st.markdown(
    "Outil **expérimental** d'analyse qualitative assistée par IA. "
    "Cette version ne contient que le squelette technique : "
    "import du corpus, gestion des runs et traçabilité des fichiers. "
    "Aucune analyse n'est encore effectuée."
)

# 3. Problématique de recherche
st.header("Problématique de recherche")
if "problematique" not in st.session_state:
    st.session_state.problematique = load_problematique()
problematique = st.text_area(
    "Problématique (facultative pour l'instant)",
    key="problematique",
    height=180,
)
if st.button("Sauvegarder la problématique"):
    try:
        path = save_problematique(problematique)
        st.success(f"Problématique sauvegardée dans {path}")
    except OSError as exc:
        st.error(f"Impossible de sauvegarder la problématique : {exc}")

# 4. Corpus
st.header("Corpus")
uploaded_files = st.file_uploader(
    "Importer des entretiens (PDF, DOCX, TXT)",
    type=list(config.ALLOWED_EXTENSIONS),
    accept_multiple_files=True,
)
if uploaded_files:
    st.table(
        [
            {
                "Nom": f.name,
                "Format": get_extension(f.name).upper(),
                "Taille": format_size(f.size),
                "Statut": "chargé",
            }
            for f in uploaded_files
        ]
    )
else:
    st.info("Aucun fichier chargé.")

# 5. Pipeline (étapes futures, inactives)
st.header("Pipeline")
st.caption("Étapes prévues — inactives dans cette version.")
for index, step in enumerate(config.PIPELINE_STEPS, start=1):
    st.markdown(f"⚪ **{index}. {step}** — _inactif_")

# 6. Lancement
if st.button("Lancer l'analyse", type="primary"):
    if not uploaded_files:
        st.error("Aucun fichier chargé : importez au moins un entretien avant de lancer l'analyse.")
    else:
        progress = st.progress(0, text="Initialisation…")
        try:
            for i, label in enumerate(SIMULATED_STEPS[:-1], start=1):
                progress.progress(i / len(SIMULATED_STEPS), text=label)
                time.sleep(0.3)
            files = [(f.name, f.getvalue()) for f in uploaded_files]
            st.session_state.last_run = init_run(files, problematique)
            progress.progress(1.0, text=SIMULATED_STEPS[-1])
            st.success("Pipeline initialisé avec succès")
        except TraceError as exc:
            progress.empty()
            st.error(str(exc))
        except OSError as exc:
            progress.empty()
            logging.exception("Erreur d'écriture pendant l'initialisation du run")
            st.error(f"Erreur d'écriture sur le disque : {exc}")

# 7. Résultats
st.header("Résultats")
run = st.session_state.get("last_run")
if run:
    st.markdown(f"**Identifiant du run :** `{run['run_id']}`")
    st.markdown(f"**Date / heure :** {run['created_at']}")
    st.markdown(f"**Nombre de fichiers :** {run['file_count']}")
    st.markdown("**Fichiers :** " + ", ".join(f["stored_name"] for f in run["files"]))
    st.markdown(f"**Emplacement des résultats :** `{run['output_dir']}`")
    st.download_button(
        "Télécharger les métadonnées (JSON)",
        data=json.dumps(run, ensure_ascii=False, indent=2),
        file_name=f"{run['run_id']}_{config.METADATA_FILENAME}",
        mime="application/json",
    )
else:
    st.info("Aucun run lancé pendant cette session.")
