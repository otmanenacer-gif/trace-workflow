"""TRACE — interface Streamlit (ingestion déterministe des entretiens, sans IA).

Lancement : streamlit run app.py
"""

import json
import logging
from pathlib import Path

import streamlit as st

from core import config
from core.ingestion import ingest_run
from core.run_manager import TraceError, get_extension, init_run, load_problematique, save_problematique

config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=config.LOGS_DIR / "trace.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    encoding="utf-8",
)

STATUS_ICONS = {"PASS": "✅", "PASS_WITH_WARNINGS": "⚠️", "FAIL": "❌"}
PREVIEW_TURNS = 10


def format_size(size_bytes: int) -> str:
    """Affiche une taille de fichier de façon lisible."""
    if size_bytes < 1024:
        return f"{size_bytes} o"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} Ko"
    return f"{size_bytes / 1024 ** 2:.1f} Mo"


def read_output(ingestion: dict, filename: str) -> str:
    """Relit un fichier de sortie d'un entretien (source de vérité : le disque)."""
    return (Path(ingestion["output_dir"]) / filename).read_text(encoding="utf-8")


def render_pipeline(run: dict | None) -> None:
    """Affiche les étapes : seule la structuration est active dans cette version."""
    for index, step in enumerate(config.PIPELINE_STEPS, start=1):
        if step == config.INGESTION_STEP:
            state = run["pipeline"][step] if run else "prête"
            icon = "🟢" if run and "échec" not in state else ("🟠" if run else "🔵")
            st.markdown(f"{icon} **{index}. {step}** — {state}")
        else:
            st.markdown(f"⚪ **{index}. {step}** — _inactif_")


st.set_page_config(page_title="TRACE", layout="wide")

# 1. Titre et 2. introduction
st.title("TRACE")
st.subheader("Analyse ethnométhodologique des usages étudiants des IAG")
st.markdown(
    "Outil **expérimental** d'analyse qualitative assistée par IA. "
    "Cette version réalise uniquement l'ingestion **déterministe** des entretiens : "
    "extraction du texte, découpage en tours de parole et contrôle qualité. "
    "Aucune IA n'est utilisée et aucune analyse n'est encore effectuée."
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

# 5. Pipeline (seule la structuration est active ; affichée après le lancement)
st.header("Pipeline")
st.caption("Seule la structuration des entretiens est active dans cette version (sans IA).")
pipeline_area = st.container()

# 6. Lancement
if st.button("Lancer l'analyse", type="primary"):
    if not uploaded_files:
        st.error("Aucun fichier chargé : importez au moins un entretien avant de lancer l'analyse.")
    else:
        progress = st.progress(0, text="Création du run et copie des fichiers…")
        try:
            files = [(f.name, f.getvalue()) for f in uploaded_files]
            run = init_run(files, problematique)

            def show_progress(index, total, filename):
                progress.progress(
                    (index + 1) / (total + 1),
                    text=f"Extraction, structuration et contrôle : {filename} ({index + 1}/{total})",
                )

            st.session_state.last_run = ingest_run(run, show_progress)
            progress.progress(1.0, text="Structuration terminée")
            st.success("Structuration des entretiens terminée")
        except TraceError as exc:
            progress.empty()
            st.error(str(exc))
        except OSError as exc:
            progress.empty()
            logging.exception("Erreur d'écriture pendant l'initialisation du run")
            st.error(f"Erreur d'écriture sur le disque : {exc}")

with pipeline_area:
    render_pipeline(st.session_state.get("last_run"))

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

    st.subheader("Ingestion des entretiens")
    ingested = [f for f in run["files"] if "ingestion" in f]
    st.table(
        [
            {
                "Nom": f["stored_name"],
                "Identifiant": f["ingestion"]["interview_id"],
                "Statut": f"{STATUS_ICONS.get(f['ingestion']['status'], '')} {f['ingestion']['status']}",
                "Tours": f["ingestion"]["turn_count"],
                "Avertissements": f["ingestion"]["warning_count"],
            }
            for f in ingested
        ]
    )
    for f in ingested:
        ingestion = f["ingestion"]
        with st.expander(f"{STATUS_ICONS.get(ingestion['status'], '')} {f['stored_name']} — aperçu"):
            try:
                transcript_json = read_output(ingestion, config.STRUCTURED_TRANSCRIPT_FILENAME)
                report_json = read_output(ingestion, config.INGESTION_REPORT_FILENAME)
            except OSError as exc:
                st.error(f"Sorties introuvables : {exc}")
                continue
            transcript, report = json.loads(transcript_json), json.loads(report_json)
            coverage = report.get("coverage") or {}
            st.markdown(
                f"Enquêteur : **{report['interviewer_turn_count']}** tours · "
                f"Enquêté : **{report['interviewee_turn_count']}** · "
                f"Inconnu : **{report['unknown_segment_count']}** · "
                f"Texte attribué : **{report['attributed_text_ratio']:.0%}** · "
                f"Couverture : **{'exacte' if coverage.get('exact_match') else coverage.get('token_coverage_ratio', 'n/a')}**"
            )
            if transcript["turns"]:
                st.markdown(f"**{min(PREVIEW_TURNS, len(transcript['turns']))} premiers tours de parole**")
                st.dataframe(
                    [
                        {
                            "turn_id": t["turn_id"],
                            "speaker": t["speaker"],
                            "speaker_raw": t["speaker_raw"] or "",
                            "page": t["source"]["page"] if t["source"]["page"] is not None else "",
                            "text": t["text"],
                        }
                        for t in transcript["turns"][:PREVIEW_TURNS]
                    ],
                    hide_index=True,
                )
            for w in report["warnings"]:
                where = f" ({w['turn_id']})" if "turn_id" in w else ""
                line = f"`{w['code']}`{where} — {w['message']}"
                if w["severity"] == "error":
                    st.error(line)
                elif w["severity"] == "warning":
                    st.warning(line)
                else:
                    st.info(line)
            col1, col2 = st.columns(2)
            col1.download_button(
                "Télécharger structured_transcript.json",
                data=transcript_json,
                file_name=f"{ingestion['interview_id']}_{config.STRUCTURED_TRANSCRIPT_FILENAME}",
                mime="application/json",
                key=f"transcript_{ingestion['interview_id']}",
            )
            col2.download_button(
                "Télécharger ingestion_report.json",
                data=report_json,
                file_name=f"{ingestion['interview_id']}_{config.INGESTION_REPORT_FILENAME}",
                mime="application/json",
                key=f"report_{ingestion['interview_id']}",
            )
else:
    st.info("Aucun run lancé pendant cette session.")
