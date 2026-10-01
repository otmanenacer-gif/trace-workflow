"""TRACE — interface Streamlit.

Ingestion déterministe des entretiens (sans IA), puis, uniquement sur action
explicite de l'utilisateur, l'étape 3 : deux agents IA indépendants
(Practice Extractor et Interaction Signal Reader).

Lancement : streamlit run app.py
"""

import json
import logging
from pathlib import Path

import streamlit as st

from core import analysis, config
from core.ingestion import ingest_run
from core.llm_client import LLMError, LLMSettings
from core.run_manager import TraceError, get_extension, init_run, load_problematique, save_problematique

config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=config.LOGS_DIR / "trace.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    encoding="utf-8",
)

config.load_env_file()  # .env local facultatif (jamais versionné) ; l'environnement a priorité

STATUS_ICONS = {"PASS": "✅", "PASS_WITH_WARNINGS": "⚠️", "FAIL": "❌"}
AI_STATUS_ICONS = {
    analysis.STATUS_PENDING: "⏳", analysis.STATUS_RUNNING: "🔄", analysis.STATUS_SUCCESS: "✅",
    analysis.STATUS_SUCCESS_WITH_WARNINGS: "⚠️", analysis.STATUS_CACHED: "♻️", analysis.STATUS_FAILED: "❌",
}
AI_AGENTS = {spec.name: spec for spec in analysis.AGENTS}
AUDITOR = analysis.AUDITOR
MODE_TEST = "Test — un entretien"
MODE_CORPUS = "Corpus complet"
PREVIEW_TURNS = 10
PREVIEW_PRACTICES = 3
PREVIEW_SIGNALS = 5
SPEAKER_AUDIT_NOTICE = "Ces suggestions ne modifient pas la transcription originale."


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


def format_count(value: int | None) -> str:
    """Entier lisible à la française : 18 432."""
    return f"{value or 0:,}".replace(",", "\u202f")


def render_pipeline(run: dict | None, llm_enabled: bool) -> None:
    """Affiche les étapes : structuration, puis les deux agents de l'étape 3 ; le reste est inactif."""
    ai_steps = (config.PRACTICE_STEP, config.INTERACTION_STEP)
    for index, step in enumerate(config.PIPELINE_STEPS, start=1):
        if step == config.INGESTION_STEP:
            state = run["pipeline"][step] if run else "prête"
            icon = "🟢" if run and "échec" not in state else ("🟠" if run else "🔵")
            st.markdown(f"{icon} **{index}. {step}** — {state}")
        elif step in ai_steps:
            state = run["pipeline"].get(step, "inactive") if run else "inactive"
            if state != "inactive":
                st.markdown(f"{'🟠' if 'échec' in state else '🟢'} **{index}. {step}** (IA) — {state}")
            elif llm_enabled:
                st.markdown(f"🔵 **{index}. {step}** (IA) — prête, sur action explicite")
            else:
                st.markdown(f"⚪ **{index}. {step}** (IA) — _désactivée (clé API ou modèle absent)_")
        else:
            st.markdown(f"⚪ **{index}. {step}** — _inactif_")


def read_analysis_file(summary: dict, filename: str) -> str | None:
    path = Path(summary["analysis_dir"]) / filename
    return path.read_text(encoding="utf-8") if path.is_file() else None


def render_analysis_launcher(run: dict, eligible: list[dict], settings: LLMSettings) -> None:
    """Choix du mode, estimation des appels, confirmation, puis lancement explicite."""
    if st.session_state.pop("ai_reset", False):
        # Après chaque lancement : « forcer » et la confirmation du corpus doivent être redonnés explicitement.
        st.session_state.ai_force = False
        st.session_state.ai_confirm_corpus = False
    flash = st.session_state.pop("ai_flash", None)  # message de fin d'analyse, conservé après st.rerun()
    if flash:
        (st.warning if flash[0] == "warning" else st.success)(flash[1])
    st.markdown(
        f"Clé API : **détectée** (jamais affichée) · Modèle : `{settings.model}` · "
        f"Appels simultanés au plus : **{settings.max_concurrency}**"
    )
    mode = st.radio("Mode", [MODE_TEST, MODE_CORPUS], key="ai_mode", horizontal=True)
    ids = [f["ingestion"]["interview_id"] for f in eligible]
    selected = [st.selectbox("Entretien à analyser", ids, key="ai_interview")] if mode == MODE_TEST else ids
    force = st.checkbox(
        "Forcer une nouvelle analyse (ignore le cache et refait les appels)", key="ai_force",
        help="Inutile en usage courant : une analyse identique (même entretien, prompt, version, schéma, modèle) est reprise du cache.",
    )
    try:
        plan = analysis.plan_analysis(run, selected, settings, force=force)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Transcriptions structurées illisibles : {exc}")
        return
    at_most = "" if plan["exact"] else "au plus "
    st.info(
        "Cette action effectue **2 appels LLM par entretien non présent dans le cache** (les deux agents), "
        "précédés d'**au plus 1 appel d'audit des locuteurs** par entretien, uniquement si des tours à "
        "l'attribution douteuse sont détectés (sinon aucun). "
        f"Pour cette sélection : **{at_most}{plan['calls']} appel(s) API** prévu(s), "
        f"{plan['cached']} résultat(s) déjà en cache (sans coût). "
        f"Présélection déterministe des locuteurs : {plan['candidate_turns']} tour(s) suspect(s), "
        f"{plan['audit_calls']} appel(s) d'audit."
    )
    confirmed = True
    if mode == MODE_CORPUS:
        confirmed = st.checkbox("Je confirme lancer l'analyse IA sur tout le corpus.", key="ai_confirm_corpus")
    clicked = st.button(
        f"Lancer les deux analyses IA ({at_most}{plan['calls']} appel(s) API payant(s))",
        type="primary", key="ai_launch", disabled=not confirmed,
    )
    if not clicked:
        return

    status_area = st.empty()
    states: dict[tuple[str, str], str] = {}

    def on_status(interview_id: str, agent: str, status: str) -> None:
        states[(interview_id, agent)] = status
        status_area.table([
            {"Entretien": iid,
             **{spec.label: f"{AI_STATUS_ICONS.get(states.get((iid, name)), '')} {states.get((iid, name), '')}"
                for name, spec in ((AUDITOR.name, AUDITOR), *AI_AGENTS.items())}}
            for iid in dict.fromkeys(i for i, _ in states)
        ])

    with st.spinner("Analyse IA en cours (audit des locuteurs, puis deux agents en parallèle par entretien)…"):
        try:
            st.session_state.last_run = analysis.analyze_run(
                run, selected, settings=settings, force=force, on_status=on_status)
        except LLMError as exc:
            st.error(exc.user_message)
            return
        except (OSError, ValueError, KeyError) as exc:
            logging.error("Analyse IA interrompue : %s", type(exc).__name__)
            st.error(f"Analyse IA interrompue ({type(exc).__name__}) : vérifiez les fichiers du run.")
            return
    usage = st.session_state.last_run["last_analysis"]["usage"]
    if usage["failed"]:
        st.session_state.ai_flash = ("warning", f"Analyse IA terminée avec {usage['failed']} échec(s) : voir le détail ci-dessous.")
    else:
        st.session_state.ai_flash = ("success", "Analyse IA terminée.")
    st.session_state.ai_reset = True
    st.rerun()  # l'estimation des appels (cache) et le pipeline reflètent immédiatement le nouvel état


def render_analysis_results(run: dict) -> None:
    """Statuts, comptes, consommation de tokens, aperçus et téléchargements."""
    analyzed = [f for f in run["files"] if "analysis" in f]
    if not analyzed:
        return
    usage = (run.get("last_analysis") or {}).get("usage")
    if usage:
        line = (f"**Dernière exécution :** {usage['api_calls']} appel(s) API — "
                f"{format_count(usage['input_tokens'])} tokens entrée — "
                f"{format_count(usage['output_tokens'])} tokens sortie")
        if usage["cache_read_input_tokens"] or usage["cache_creation_input_tokens"]:
            line += (f" — cache de prompt : {format_count(usage['cache_read_input_tokens'])} lus, "
                     f"{format_count(usage['cache_creation_input_tokens'])} écrits")
        line += f" — {usage['cached_results']} résultat(s) repris du cache TRACE"
        st.markdown(line)
        st.caption("Tokens rapportés par l'API. Aucun montant n'est calculé : le prix du modèle n'est pas connu localement de façon fiable.")

    rows = []
    for f in analyzed:
        agents = f["analysis"]["agents"]
        practice, signals = agents["practice_extractor"], agents["interaction_signal_reader"]
        billed = [a["usage"] for a in agents.values() if a.get("billed_this_run") and a.get("usage")]
        audit = f["analysis"].get("speaker_audit") or {}
        if audit.get("billed_this_run") and audit.get("usage"):
            billed.append(audit["usage"])
        rows.append({
            "Entretien": f["analysis"]["interview_id"],
            "Audit locuteurs": f"{AI_STATUS_ICONS.get(audit.get('status'), '')} {audit.get('status') or 'n/a'}",
            "Tours suspects": audit.get("candidate_count") or 0,
            "Locuteurs à vérifier": audit.get("review_count") or 0,
            "Practice Extractor": f"{AI_STATUS_ICONS.get(practice['status'], '')} {practice['status']}",
            "Interaction Reader": f"{AI_STATUS_ICONS.get(signals['status'], '')} {signals['status']}",
            "Pratiques": practice.get("item_count") or 0,
            "Signaux": signals.get("item_count") or 0,
            "Citations invalides": f["analysis"]["invalid_evidence_count"],
            "Tokens facturés (entrée / sortie)": " / ".join(
                format_count(sum(u.get(k) or 0 for u in billed)) for k in ("input_tokens", "output_tokens")),
            "Avertissements": "oui" if any(a.get("has_warnings") for a in agents.values()) else "non",
        })
    st.table(rows)

    for f in analyzed:
        summary = f["analysis"]
        iid = summary["interview_id"]
        with st.expander(f"Analyse IA — {iid} — aperçu"):
            for name, agent in summary["agents"].items():
                if agent.get("error"):
                    st.error(f"{AI_AGENTS[name].label} : {agent['error']['message']}")
            practice_json = read_analysis_file(summary, AI_AGENTS["practice_extractor"].output_filename)
            signals_json = read_analysis_file(summary, AI_AGENTS["interaction_signal_reader"].output_filename)
            validation_json = read_analysis_file(summary, config.EVIDENCE_VALIDATION_FILENAME)
            if practice_json:
                practices = json.loads(practice_json)["practices"]
                st.markdown(f"**Pratiques** ({len(practices)}) — {min(PREVIEW_PRACTICES, len(practices))} premières")
                st.dataframe([
                    {"id": p["practice_id"], "statut": p["use_status"], "outils": ", ".join(p["ai_tool"]),
                     "description": p["summary"], "tours": f"{p['turn_start']} → {p['turn_end']}",
                     "citations": " | ".join(("✓ " if e["validation"]["valid"] else "✗ ") + e["quote"] for e in p["evidence"]),
                     "à revoir": ", ".join(p["review_reasons"])}
                    for p in practices[:PREVIEW_PRACTICES]
                ], hide_index=True)
            if signals_json:
                signals = json.loads(signals_json)["signals"]
                st.markdown(f"**Signaux interactionnels** ({len(signals)}) — {min(PREVIEW_SIGNALS, len(signals))} premiers")
                st.dataframe([
                    {"id": s["signal_id"], "type": s["signal_type"], "forme": s["surface_form"],
                     "description": s["description"], "affect explicite": s["explicit_affect"] or "",
                     "citations": " | ".join(("✓ " if e["validation"]["valid"] else "✗ ") + e["quote"] for e in s["evidence"]),
                     "à revoir": ", ".join(s["review_reasons"])}
                    for s in signals[:PREVIEW_SIGNALS]
                ], hide_index=True)
            if validation_json:
                validation = json.loads(validation_json)
                issues = [i for a in validation["agents"].values() for i in a.get("issues", []) if i["severity"] != "info"]
                if issues:
                    st.markdown(f"**Anomalies de validation** ({len(issues)})")
                    st.dataframe([{"objet": i["object_id"], "code": i["code"], "gravité": i["severity"],
                                   "tour": i.get("turn_id", ""), "message": i["message"]} for i in issues],
                                 hide_index=True)
            cols = st.columns(3)
            for col, content, filename in (
                (cols[0], practice_json, AI_AGENTS["practice_extractor"].output_filename),
                (cols[1], signals_json, AI_AGENTS["interaction_signal_reader"].output_filename),
                (cols[2], validation_json, config.EVIDENCE_VALIDATION_FILENAME),
            ):
                if content:
                    col.download_button(f"Télécharger {filename}", data=content, file_name=f"{iid}_{filename}",
                                        mime="application/json", key=f"dl_{iid}_{filename}")
        render_speaker_audit(summary)


def render_speaker_audit(summary: dict) -> None:
    """Audit d'attribution des locuteurs : comptes, avertissements, téléchargement. Rien n'est modifiable ici."""
    audit_json = read_analysis_file(summary, AUDITOR.output_filename)
    if not audit_json:
        return
    audit = json.loads(audit_json)
    iid = summary["interview_id"]
    title = (f"Audit d'attribution des locuteurs — {iid} — {audit['candidate_count']} tour(s) suspect(s), "
             f"{audit['review_count']} à vérifier")
    with st.expander(title):
        st.info(SPEAKER_AUDIT_NOTICE + " Le locuteur officiel reste celui de structured_transcript.json ; "
                "les agents reçoivent seulement un avertissement sur les tours douteux.")
        mode = ("aucun tour suspect : aucun appel API" if audit["audit_mode"] == "heuristics_only"
                else "audit par le modèle en échec : candidats déterministes seuls" if audit["status"] == "FAILED"
                else "repris du cache TRACE" if audit.get("cache_hit") else "1 appel d'audit")
        st.markdown(f"Statut : **{audit['status']}** · Tours suspects (règles déterministes) : "
                    f"**{audit['candidate_count']}** · À vérifier : **{audit['review_count']}** · {mode}")
        if audit["items"]:
            st.dataframe([
                {"tour": i["turn_id"], "locuteur officiel": i["current_speaker"],
                 "locuteur suggéré": i["suggested_speaker"] or "—", "confiance": i["confidence"],
                 "raison": i["reason"], "règles": ", ".join(i["heuristics"]), "source": i["source"],
                 "à vérifier": "oui" if i["needs_review"] else "non",
                 "citations": " | ".join(("✓ " if e["validation"]["valid"] else "✗ ") + e["quote"]
                                         for e in i["evidence"])}
                for i in audit["items"]
            ], hide_index=True)
        st.download_button(f"Télécharger {AUDITOR.output_filename}", data=audit_json,
                           file_name=f"{iid}_{AUDITOR.output_filename}", mime="application/json",
                           key=f"dl_{iid}_{AUDITOR.output_filename}")


def render_analysis_section(run: dict) -> None:
    st.header("Analyse IA — Étape 3")
    st.markdown(
        "Deux agents **indépendants**, lancés **en parallèle** sur chaque entretien : "
        "**Practice Extractor** (ce que l'étudiant·e fait, avec ou sans IAG) et "
        "**Interaction Signal Reader** (comment il ou elle le raconte). Aucun des deux ne voit la sortie "
        "de l'autre ; aucune synthèse ni analyse théorique n'est produite à cette étape. "
        "Chaque citation est vérifiée mot pour mot dans l'entretien. "
        "Au préalable, un **audit de l'attribution des locuteurs** signale les tours dont le locuteur "
        "semble douteux (règles déterministes, puis un appel LLM seulement s'il y a des tours suspects) ; "
        "il ne modifie jamais la transcription."
    )
    eligible = analysis.eligible_files(run)
    settings = LLMSettings.from_env()
    for problem in settings.problems:
        st.warning(problem)
    if not eligible:
        st.info("Aucun entretien analysable (ingestion en échec ou sans tour de parole).")
    elif not settings.enabled:
        st.warning(settings.disabled_reason())
    else:
        render_analysis_launcher(run, eligible, settings)
    render_analysis_results(st.session_state.get("last_run") or run)


st.set_page_config(page_title="TRACE", layout="wide")

# 1. Titre et 2. introduction
st.title("TRACE")
st.subheader("Analyse ethnométhodologique des usages étudiants des IAG")
st.markdown(
    "Outil **expérimental** d'analyse qualitative assistée par IA. "
    "L'ingestion des entretiens est **déterministe** et sans IA : "
    "extraction du texte, découpage en tours de parole et contrôle qualité. "
    "L'étape 3 (deux agents IA descriptifs) ne s'exécute que sur **action explicite**, "
    "après l'ingestion ; aucun appel API n'est effectué au chargement des fichiers."
)
llm_settings = LLMSettings.from_env()

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

# 5. Pipeline (structuration + étape 3 ; affichée après le lancement)
st.header("Pipeline")
st.caption(
    "Actives : la structuration des entretiens (sans IA) et, sur action explicite, "
    "l'extraction des pratiques et l'analyse interactionnelle (étape 3). Les autres étapes restent inactives."
)
pipeline_area = st.container()

# 6. Lancement (ingestion uniquement : aucun appel IA ici)
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

    # 8. Étape 3 — analyse IA (jamais automatique)
    render_analysis_section(run)
else:
    st.info("Aucun run lancé pendant cette session.")

# Le pipeline est rempli en dernier : il reflète aussi une analyse IA lancée pendant cette exécution.
with pipeline_area:
    render_pipeline(st.session_state.get("last_run"), llm_settings.enabled)
