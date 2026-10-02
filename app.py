"""TRACE — interface Streamlit.

Ingestion déterministe des entretiens (sans IA), puis, uniquement sur action
explicite de l'utilisateur, l'étape 3 : deux agents IA indépendants
(Practice Extractor et Interaction Signal Reader), puis l'étape 4 : épisodes
d'accountability (candidats déterministes, au plus un appel LLM par entretien), puis
l'étape 5 : configuration et trajectoire intra-entretien (au plus un appel LLM par entretien), puis
l'étape 6 : comparaison inter-entretiens, sur les seules sorties de l'étape 5 importées (un appel LLM par corpus).

Lancement : streamlit run app.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import streamlit as st


PROJECT_PACKAGES = ("core", "agents")


def _project_modules() -> list:
    return [(name, module) for name, module in list(sys.modules.items())
            if name.split(".")[0] in PROJECT_PACKAGES and getattr(module, "__file__", None)]


def _process_start_time() -> float | None:
    try:
        return os.stat(f"/proc/{os.getpid()}").st_ctime  # Linux (Streamlit Cloud)
    except OSError:
        return None


def _purge_stale_project_modules() -> None:
    """Oublie les modules du projet (core, agents) modifiés sur disque depuis leur chargement.

    Streamlit ré-exécute app.py dans le même processus : après un redéploiement à chaud
    (git pull sur Streamlit Cloud), un ancien core.analysis resté dans sys.modules masquait
    le code à jour (AttributeError: analysis.AUDITOR). Ces modules sont alors réimportés.
    """
    process_start = _process_start_time()
    stale = False
    for _, module in _project_modules():
        loaded_at = getattr(module, "_trace_loaded_at", process_start)
        try:
            stale |= loaded_at is not None and os.path.getmtime(module.__file__) > loaded_at
        except OSError:
            stale = True
    if stale:
        for name, _ in _project_modules():
            del sys.modules[name]


def _stamp_project_modules() -> None:
    now = time.time()
    for _, module in _project_modules():
        if not hasattr(module, "_trace_loaded_at"):
            module._trace_loaded_at = now


_purge_stale_project_modules()

from core import accountability, analysis, config, stage3_restore, stage4_restore, trajectory  # noqa: E402
from core import cross_interview, cross_interview_corpus  # noqa: E402
from core import claude_code_workflow as workflow  # noqa: E402
from core.ingestion import ingest_run  # noqa: E402
from core.llm_client import LLMSettings  # noqa: E402
from core.run_manager import TraceError, get_extension, init_run, list_runs, load_metadata  # noqa: E402
from core.run_manager import load_problematique, save_problematique  # noqa: E402

_stamp_project_modules()

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
    analysis.STATUS_PARTIAL: "🟠",
}
AI_AGENTS = {spec.name: spec for spec in analysis.AGENTS}
AUDITOR = analysis.AUDITOR
MODE_TEST = "Test — un entretien"
MODE_CORPUS = "Corpus complet"
PREVIEW_TURNS = 10
PREVIEW_PRACTICES = 3
PREVIEW_SIGNALS = 5
SPEAKER_AUDIT_NOTICE = "Ces suggestions ne modifient pas la transcription originale."
ACC_STATUS_ICONS = {**AI_STATUS_ICONS, accountability.STATUS_BLOCKED: "⛔"}
ACC_ALL = "Tous les entretiens analysés"
PREVIEW_EPISODES = 3
EPISODE_STATUS_LABELS = {"accountability_episode": "épisode d'accountability", "ordinary_practice": "pratique ordinaire",
                         "uncertain": "incertain"}
CONFIGURATION_LABELS = {"temporal_trajectory": "trajectoire temporelle explicite",
                        "contextual_configuration": "configuration contextuelle",
                        "mixed": "mixte (changement temporel explicite et variations contextuelles)",
                        "no_clear_pattern": "pas de configuration nette"}
CLAIM_TYPE_LABELS = {"stable_boundary": "frontière stable", "recurring_accounting_move": "opération récurrente",
                     "contextual_variation": "variation contextuelle", "explicit_temporal_change": "changement temporel explicite",
                     "exception": "exception", "unresolved_tension": "tension non résolue", "ordinary_zone": "zone ordinaire"}
STAGE4_STATE_LABELS = {trajectory.STAGE4_COMPLETE: "disponible", trajectory.STAGE4_PARTIAL: "incomplète (PARTIAL)",
                       trajectory.STAGE4_FAILED: "en échec ou bloquée", trajectory.STAGE4_STALE: "périmée",
                       trajectory.STAGE4_NOT_RUN: "absente"}


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


def render_pipeline(run: dict | None) -> None:
    """Affiche les étapes : ingestion, puis les étapes IA (workflow Claude Code, aucun appel API)."""
    ai_steps = (config.PRACTICE_STEP, config.INTERACTION_STEP, config.ACCOUNTABILITY_STEP, config.TRAJECTORY_STEP)
    for index, step in enumerate(config.PIPELINE_STEPS, start=1):
        if step == config.INGESTION_STEP:
            state = run["pipeline"][step] if run else "prête"
            icon = "🟢" if run and "échec" not in state else ("🟠" if run else "🔵")
            st.markdown(f"{icon} **{index}. {step}** — {state}")
        elif step == config.CROSS_INTERVIEW_STEP:
            last = st.session_state.get("stage6_last")
            if last:
                warn = last["status"] not in cross_interview.DONE_STATUSES
                label = ("en attente du workflow Claude Code"
                         if (last.get("error") or {}).get("code") == workflow.AWAITING_AGENT else last["status"])
                st.markdown(f"{'🟠' if warn else '🟢'} **{index}. {step}** (IA) — {label} "
                            f"({last['corpus_n_usable']}/{last['corpus_n_total']} entretien(s) exploitable(s))")
            else:
                st.markdown(f"🔵 **{index}. {step}** (IA) — prête, workflow Claude Code (0 appel API), sur import "
                            "des sorties de l'étape 5")
        elif step in ai_steps:
            state = run["pipeline"].get(step, "inactive") if run else "inactive"
            if state != "inactive":
                warn = "échec" in state or "incomplet" in state or "bloqué" in state
                st.markdown(f"{'🟠' if warn else '🟢'} **{index}. {step}** (IA) — {state}")
            else:
                st.markdown(f"🔵 **{index}. {step}** (IA) — prête, workflow Claude Code (0 appel API)")
        else:
            st.markdown(f"⚪ **{index}. {step}** — _inactif_")


def read_analysis_file(summary: dict, filename: str) -> str | None:
    path = Path(summary["analysis_dir"]) / filename
    return path.read_text(encoding="utf-8") if path.is_file() else None


def chunk_label(agent: dict) -> str:
    """« 4/4 » (blocs réussis / total) pour un entretien long, « 1 » sinon."""
    chunking = agent.get("chunking") or {}
    if not chunking.get("chunking_used"):
        return "1"
    return f"{chunking.get('chunks_succeeded') or 0}/{chunking.get('chunk_count')}"


def render_chunking(agent: dict, label: str = "Interaction Reader") -> None:
    """Bilan de la lecture par blocs d'un agent (entretien long)."""
    chunking = agent.get("chunking") or {}
    if not chunking.get("chunking_used"):
        return
    practice = label == "Practice Extractor"
    noun, key = ("pratiques finales", "practices_before_dedup") if practice else ("signaux finaux", "signals_before_dedup")
    line = (f"**{label}** — entretien long : analyse en {chunking['chunk_count']} blocs "
            f"(chevauchement inclus) · blocs réussis : **{chunking.get('chunks_succeeded') or 0}/"
            f"{chunking['chunk_count']}** · {noun} : **{agent.get('item_count') or 0}**")
    if chunking.get(key) is not None:
        line += f" (avant dédoublonnage : {chunking[key]})"
    if practice:
        line += f" · statut : {agent.get('status')}"
    else:
        line += f" · lecture à longue distance : {chunking.get('long_distance_status') or 'n/a'}"
        line += f" · anomalies de validation : {agent.get('validation_issue_count') or 0}"
    st.markdown(line)
    if chunking.get("truncated_chunks"):  # anciennes sorties (par API) uniquement : le workflow n'en produit pas
        st.error(f"{label} — Bloc(s) tronqué(s) (limite de sortie atteinte) : "
                 + ", ".join(str(c) for c in chunking["truncated_chunks"])
                 + ". Le résultat est INCOMPLET ; relancez l'analyse pour ne refaire que les blocs manquants.")


def render_selectivity(agents: dict) -> None:
    """Étape 3.7 : signaux écartés (visibles dans interaction_signals.json) et indices de non-usage à vérifier."""
    selectivity = agents["interaction_signal_reader"].get("selectivity") or {}
    if selectivity.get("signals_set_aside"):
        reasons = ", ".join(f"{code} : {count}" for code, count in selectivity["set_aside_by_reason"].items())
        st.caption(f"Interaction Reader : {selectivity['signals_set_aside']} signal(aux) écarté(s) avant validation "
                   f"({reasons}) — conservés dans interaction_signals.json (set_aside_signals).")
    cues = agents["practice_extractor"].get("non_use_cues") or {}
    if cues.get("uncovered_count"):
        st.caption(f"Practice Extractor : {cues['uncovered_count']} tour(s) contenant une formulation de non-usage "
                   "sans pratique non_use / refusal associée (indice lexical, à vérifier ; voir non_use_cues).")


def ai_status_label(summary: dict, icons: dict = AI_STATUS_ICONS) -> str:
    if (summary.get("error") or {}).get("code") == workflow.AWAITING_AGENT:
        return "⏳ EN ATTENTE (workflow Claude Code)"
    return f"{icons.get(summary.get('status'), '')} {summary.get('status') or 'n/a'}"


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
        st.caption("Appels API et tokens : champs hérités des anciennes exécutions par API, toujours à 0 avec le "
                   "workflow Claude Code (aucun appel API, aucun coût).")

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
            "Audit locuteurs": ai_status_label(audit),
            "Tours suspects": audit.get("candidate_count") or 0,
            "Locuteurs à vérifier": audit.get("review_count") or 0,
            "Practice Extractor": ai_status_label(practice),
            "Interaction Reader": ai_status_label(signals),
            "Pratiques": practice.get("item_count") or 0,
            "Signaux": signals.get("item_count") or 0,
            "Citations invalides": f["analysis"]["invalid_evidence_count"],
            "Tokens API (hérité, toujours 0)": " / ".join(
                format_count(sum(u.get(k) or 0 for u in billed)) for k in ("input_tokens", "output_tokens")),
            "Avertissements": "oui" if any(a.get("has_warnings") for a in agents.values()) else "non",
            "Anomalies validation": sum(a.get("validation_issue_count") or 0 for a in agents.values()),
            "Blocs (Practice)": chunk_label(practice),
            "Blocs (Interaction)": chunk_label(signals),
        })
    st.table(rows)

    for f in analyzed:
        summary = f["analysis"]
        iid = summary["interview_id"]
        with st.expander(f"Analyse IA — {iid} — aperçu"):
            for name, agent in summary["agents"].items():
                if agent.get("error"):
                    show = st.info if agent["error"].get("code") == workflow.AWAITING_AGENT else st.error
                    show(f"{AI_AGENTS[name].label} : {agent['error']['message']}")
            render_chunking(summary["agents"]["practice_extractor"], "Practice Extractor")
            render_chunking(summary["agents"]["interaction_signal_reader"])
            render_selectivity(summary["agents"])
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


def render_stage3_workflow(run: dict, eligible: list[dict]) -> None:
    """Étape 3 sans API : TRACE prépare les tâches, Claude Code joue les agents, TRACE valide et enregistre."""
    flash = st.session_state.pop("wf_flash", None)
    if flash:
        (st.warning if flash[0] == "warning" else st.success)(flash[1])
    st.info(
        "**Workflow Claude Code** — aucune clé API, aucun appel API, aucun coût. TRACE prépare un paquet de tâche "
        "par agent (prompt du dépôt, matériau exact, schéma JSON attendu) ; Claude Code, ouvert dans ce dépôt, joue "
        "chaque agent et écrit sa réponse ; TRACE la valide (schéma, citations, garde-fou, sélectivité) et "
        "enregistre les sorties habituelles de l'étape 3. Procédure : `TRACE_WORKFLOW.md`."
    )
    mode = st.radio("Mode", [MODE_TEST, MODE_CORPUS], key="wf_mode", horizontal=True)
    ids = [f["ingestion"]["interview_id"] for f in eligible]
    selected = [st.selectbox("Entretien à analyser", ids, key="wf_interview")] if mode == MODE_TEST else ids
    if st.button("Préparer / reprendre l'étape 3 (workflow Claude Code, 0 appel API)", type="primary",
                 key="wf_stage3"):
        try:
            result = workflow.run_stage3(run, selected)
        except (OSError, ValueError, KeyError) as exc:
            logging.error("Workflow de l'étape 3 interrompu : %s", type(exc).__name__)
            st.error(f"Workflow de l'étape 3 interrompu ({type(exc).__name__}) : vérifiez les fichiers du run.")
            return
        st.session_state.last_run = result["metadata"]
        status = result["status"]
        st.session_state.wf_flash = (
            ("success", "Étape 3 terminée (workflow Claude Code, 0 appel API).")
            if status["status"] == workflow.STAGE3_COMPLETE else
            ("warning", f"Étape 3 : {workflow.STAGE3_STATUS_LABELS[status['status']]} — voir ci-dessous."))
        st.rerun()

    render_workflow_state(run, "3", f"Exécute les tâches TRACE en attente du run {run['run_id']}")


def render_workflow_state(run: dict, stage: str, instruction: str) -> None:
    """État du workflow Claude Code d'une étape : tâches en attente ou à corriger, instruction pour Claude Code."""
    state = (st.session_state.get("last_run") or run).get(f"stage{stage}_workflow")
    if not state:
        return
    st.markdown(f"**Workflow Claude Code (étape {stage}) :** {workflow.STATUS_LABELS[state['status']]} — "
                f"{state['answered_count']}/{state['task_count']} tâche(s) avec une réponse conforme — "
                f"{state['api_calls']} appel API")
    for interview in state["interviews"]:
        if interview.get("reason"):
            st.warning(f"{interview['interview_id']} : {interview['reason']}")
        for warning in interview.get("warnings") or []:
            st.warning(f"{interview['interview_id']} : {warning}")
    if state.get("skipped"):
        st.caption("Déjà terminée(s) et à jour, non rejouée(s) : " + ", ".join(state["skipped"]) + ".")
    todo = [(t, "réponse à corriger") for t in state["invalid"]] + [(t, "en attente") for t in state["pending"]]
    if todo:
        st.table([{"Tâche": t["task_id"], "Entretien": t["interview_id"], "Agent": t["agent_label"], "État": label,
                   "Erreur": t.get("error") or ""} for t, label in todo])
        st.markdown("Dans **Claude Code**, ouvert dans ce dépôt, demandez :")
        st.code(instruction, language=None)
        st.caption("Claude Code joue chaque agent (un sous-agent par tâche), valide sa réponse "
                   f"(`{workflow.CLI} check …`), puis rejoue l'étape. Revenez ensuite ici et cliquez de nouveau "
                   "sur « Préparer / reprendre » pour afficher les résultats.")


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
    else:
        render_stage3_workflow(run, eligible)
    render_analysis_results(st.session_state.get("last_run") or run)


# --- Étape 4 — épisodes d'accountability ----------------------------------------------------------

def render_stage4_workflow(run: dict, analyzed: list[dict]) -> None:
    """Étape 4 sans API : TRACE prépare les tâches, Claude Code joue l'Accountability Episode Builder."""
    flash = st.session_state.pop("wf4_flash", None)
    if flash:
        (st.warning if flash[0] == "warning" else st.success)(flash[1])
    st.info(
        "**Workflow Claude Code** — aucune clé API, aucun appel API, aucun coût. TRACE prépare les candidats "
        "(déterministes) et un paquet de tâche par bloc de composantes (un seul dans le cas normal) ; Claude Code joue "
        "l'Accountability Episode Builder (prompt du dépôt) ; TRACE valide chaque réponse (schéma, validateur des "
        "épisodes), fusionne les blocs et enregistre les sorties habituelles. Procédure : `TRACE_WORKFLOW.md`."
    )
    ids = [f["ingestion"]["interview_id"] for f in analyzed]
    choice = st.selectbox("Entretien(s) pour l'étape 4", ids + ([ACC_ALL] if len(ids) > 1 else []), key="wf4_interview")
    selected = ids if choice == ACC_ALL else [choice]
    if st.button("Préparer / reprendre l'étape 4 (workflow Claude Code, 0 appel API)", type="primary",
                 key="wf_stage4"):
        try:
            result = workflow.run_stage4(run, selected)
        except (OSError, ValueError, KeyError) as exc:
            logging.error("Workflow de l'étape 4 interrompu : %s", type(exc).__name__)
            st.error(f"Workflow de l'étape 4 interrompu ({type(exc).__name__}) : vérifiez les fichiers du run.")
            return
        st.session_state.last_run = result["metadata"]
        status = result["status"]["status"]
        st.session_state.wf4_flash = (
            ("success", "Étape 4 terminée (workflow Claude Code, 0 appel API).") if status == workflow.STAGE_COMPLETE
            else ("warning", f"Étape 4 : {workflow.STATUS_LABELS[status]} — voir ci-dessous."))
        st.rerun()
    render_workflow_state(run, "4", f"Exécute TRACE sur le run {run['run_id']} jusqu'à l'étape 4")


def render_stage4_results(run: dict) -> None:
    done = [f for f in run["files"] if "accountability" in f]
    if not done:
        return
    usage = (run.get("last_accountability") or {}).get("usage")
    if usage:
        st.markdown(f"**Dernière exécution (étape 4) :** {usage['api_calls']} appel(s) API — "
                    f"{format_count(usage['input_tokens'])} tokens entrée — {format_count(usage['output_tokens'])} "
                    f"tokens sortie — {usage['cached_results']} résultat(s) repris du cache TRACE")
    st.table([{
        "Entretien": s["interview_id"],
        "Statut étape 3": s.get("stage3_status") or "n/a",
        "Étape 4": ai_status_label(s, ACC_STATUS_ICONS),
        "Candidats": s.get("candidate_count") or 0,
        "Épisodes accountability": s.get("accountability_episode_count") or 0,
        "Pratiques ordinaires (examinées)": s.get("ordinary_practice_count") or 0,
        "Pratiques sans marqueur": s.get("unmarked_practice_count") or 0,
        "Incertains": s.get("uncertain_count") or 0,
        "Rejetés": s.get("rejected_episode_count") or 0,
        "Avertissements": (s.get("validation_warning_count") or 0) + (s.get("validation_error_count") or 0),
        "Appels API": s.get("api_calls") or 0,
    } for s in (f["accountability"] for f in done)])
    for f in done:
        summary = f["accountability"]
        iid = summary["interview_id"]
        with st.expander(f"Épisodes d'accountability — {iid} — aperçu"):
            if summary["status"] == accountability.STATUS_BLOCKED:
                st.error((summary.get("error") or {}).get("message") or "Étape 4 bloquée.")
            elif summary.get("error"):
                show = st.info if summary["error"].get("code") == workflow.AWAITING_AGENT else st.error
                show(f"Accountability Episode Builder : {summary['error']['message']}")
            if summary["status"] == analysis.STATUS_PARTIAL:
                st.warning("Analyse INCOMPLÈTE : les sorties de l'étape 3 sont partielles (analysis_complete = false). "
                           "Relancez l'étape 3 pour compléter les blocs manquants, puis l'étape 4.")
            episodes_json = read_analysis_file(summary, config.ACCOUNTABILITY_EPISODES_FILENAME)
            validation_json = read_analysis_file(summary, config.ACCOUNTABILITY_VALIDATION_FILENAME)
            if episodes_json:
                document = json.loads(episodes_json)
                episodes = document["episodes"]
                st.markdown(
                    f"Statut étape 3 : **{document['stage3_status']}** · candidats : **{document['candidate_count']}** · "
                    f"épisodes d'accountability : **{document['accountability_episode_count']}** · pratiques ordinaires : "
                    f"**{document['ordinary_practice_count']}** examinée(s) + **{document['unmarked_practice_count']}** "
                    f"sans marqueur · incertains : **{document['uncertain_count']}** · rejetés : "
                    f"**{document['rejected_episode_count']}** · "
                    + ("repris du cache TRACE" if document["cache_hit"] else
                       "réponse(s) d'agent du workflow Claude Code (0 appel API)"
                       if document.get("model") == workflow.WORKFLOW_MODEL else
                       "1 appel" if document["llm_called"] else "aucun appel (aucun candidat)"))
                if episodes:
                    st.markdown(f"**Épisodes** ({len(episodes)}) — {min(PREVIEW_EPISODES, len(episodes))} premiers")
                    st.dataframe([
                        {"id": e["episode_id"], "statut": EPISODE_STATUS_LABELS.get(e["episode_status"], e["episode_status"]),
                         "opérations": ", ".join(m["type"] for m in e["accounting_moves"]),
                         "frontières": ", ".join(e["boundary_objects"]),
                         "tours": f"{e['turn_start']} → {e['turn_end']}", "résumé": e["episode_summary"],
                         "citations": " | ".join(("✓ " if x["validation"]["valid"] else "✗ ") + x["quote"]
                                                 for x in e["evidence"]),
                         "validation": e["validation_status"], "à revoir": ", ".join(e["review_reasons"])}
                        for e in episodes[:PREVIEW_EPISODES]], hide_index=True)
            if validation_json:
                validation = json.loads(validation_json)
                issues = [i for i in validation.get("issues", []) if i["severity"] != "info"]
                if issues:
                    st.markdown(f"**Avertissements de validation** ({len(issues)})")
                    st.dataframe([{"objet": i["object_id"], "code": i["code"], "gravité": i["severity"],
                                   "message": i["message"]} for i in issues], hide_index=True)
            cols = st.columns(2)
            for col, content, filename in ((cols[0], episodes_json, config.ACCOUNTABILITY_EPISODES_FILENAME),
                                           (cols[1], validation_json, config.ACCOUNTABILITY_VALIDATION_FILENAME)):
                if content:
                    col.download_button(f"Télécharger {filename}", data=content, file_name=f"{iid}_{filename}",
                                        mime="application/json", key=f"dl_acc_{iid}_{filename}")


def render_stage3_restore(run: dict, analyzed: list[dict]) -> None:
    """Importer les 4 JSON d'une étape 3 déjà calculée (stockage local effacé par un redéploiement)."""
    flash = st.session_state.pop("restore_flash", None)
    if flash:
        st.success(flash)
    for f in analyzed:
        if f["analysis"].get("restored"):
            st.success(f"{stage3_restore.RESTORED_NOTICE} ({f['analysis']['interview_id']}, {f['analysis']['restored_at']}).")
            for warning in f["analysis"].get("restore_warnings") or []:
                st.caption(f"Restauration : {warning}")
    missing = [f for f in analysis.eligible_files(run) if accountability.stage3_state(
        Path(f["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR)["status"]
        in (accountability.STAGE3_NOT_RUN, accountability.STAGE3_FAILED)]
    if not missing:
        return
    with st.expander("Restaurer des résultats Stage 3 existants", expanded=not analyzed):
        st.markdown(
            "Le stockage local de l'application (runs, cache TRACE) **peut disparaître lors d'un redéploiement**. "
            "Si l'étape 3 de cet entretien a déjà été calculée et ses fichiers téléchargés, importez ici les "
            "**4 JSON** : `practice_extractor.json`, `interaction_signals.json`, `evidence_validation.json`, "
            "`speaker_attribution_audit.json`. Ils sont vérifiés (même entretien, même transcription, analyses "
            "complètes, citations revérifiées) puis installés tels quels : **aucun appel API**, ni étape 3, "
            "ni audit des locuteurs.")
        interview_id = st.selectbox("Entretien à restaurer", [f["ingestion"]["interview_id"] for f in missing],
                                    key="restore_interview")
        uploads = st.file_uploader("Les 4 fichiers JSON de l'étape 3", type=["json"], accept_multiple_files=True,
                                   key="restore_files")
        if st.button("Restaurer l'étape 3 depuis ces fichiers (0 appel API)", key="restore_launch",
                     disabled=not uploads):
            try:
                st.session_state.last_run = stage3_restore.restore_stage3(
                    run, interview_id, [(u.name, u.getvalue()) for u in uploads])
            except stage3_restore.RestoreError as exc:
                st.error("Restauration refusée :\n" + "\n".join(f"- {problem}" for problem in exc.problems))
                return
            except OSError as exc:
                st.error(f"Restauration impossible (écriture) : {exc}")
                return
            st.session_state.restore_flash = f"{stage3_restore.RESTORED_NOTICE} ({interview_id})."
            st.rerun()


def render_stage4_section(run: dict) -> None:
    st.header("Étape 4 — Épisodes d'accountability")
    st.markdown(
        "À partir des sorties de l'étape 3 (jamais de l'entretien entier), un programme **déterministe** propose "
        "des candidats : pratiques accompagnées de signaux proches, contradictions entre tours, usage et non-usage "
        "d'une même tâche, frontières explicites (« je fais X mais pas Y »). L'**Accountability Episode Builder** "
        "dit ensuite, pour chaque candidat, s'il s'agit d'un épisode d'accountability (une conduite rendue "
        "descriptible et intelligible), d'une **pratique ordinaire** (racontée comme allant de soi) ou d'un cas "
        "**incertain** ; un validateur déterministe vérifie chaque citation et identifiant. Une pratique sans aucun "
        "marqueur n'est pas envoyée au modèle."
    )
    analyzed = [f for f in analysis.eligible_files(run) if "analysis" in f]
    render_stage3_restore(run, analyzed)
    if not analyzed:
        st.info("Lancez d'abord l'étape 3, ou restaurez ses résultats depuis des fichiers déjà téléchargés : "
                "l'étape 4 en consomme les sorties.")
        return
    render_stage4_workflow(run, analyzed)
    render_stage4_results(st.session_state.get("last_run") or run)


# --- Étape 5 — configuration et trajectoire intra-entretien -----------------------------------------------

def _interview_dir(info: dict) -> Path:
    return Path(info["ingestion"]["output_dir"])


def render_stage4_restore(run: dict) -> None:
    """Importer les 2 JSON d'une étape 4 déjà calculée (stockage local effacé par un redéploiement)."""
    flash = st.session_state.pop("restore4_flash", None)
    if flash:
        st.success(flash)
    files = analysis.eligible_files(run)
    for f in files:
        summary = f.get("accountability") or {}
        if summary.get("restored"):
            st.success(f"{stage4_restore.RESTORED_NOTICE} ({summary['interview_id']}, {summary['restored_at']}).")
            for warning in summary.get("restore_warnings") or []:
                st.caption(f"Restauration (étape 4) : {warning}")
    missing = [f for f in files if trajectory.stage4_state(_interview_dir(f))["status"]
               in (trajectory.STAGE4_NOT_RUN, trajectory.STAGE4_FAILED, trajectory.STAGE4_STALE)]
    if not missing:
        return
    with st.expander("Restaurer des résultats Stage 4 existants", expanded=True):
        st.markdown(
            "Le stockage local de l'application **peut disparaître lors d'un redéploiement**. Si l'étape 4 de cet "
            "entretien a déjà été calculée et ses fichiers téléchargés, importez ici les **2 JSON** : "
            "`accountability_episodes.json` et `accountability_episode_validation.json`. Ils sont vérifiés (même "
            "entretien, versions compatibles, analyse complète, aucune erreur de validation, empreintes de l'étape 3 "
            "installée, candidats et épisodes recalculés par TRACE, citations revérifiées) puis installés tels quels : "
            "**aucun appel API**, l'étape 4 n'est pas relancée.")
        ready = [f for f in missing if accountability.stage3_state(
            _interview_dir(f) / config.ANALYSIS_SUBDIR)["status"] == accountability.STAGE3_COMPLETE]
        if not ready:
            st.info("L'étape 3 de cet entretien n'est pas disponible : restaurez-la (ou lancez-la) d'abord, dans la "
                    "section de l'étape 4.")
            return
        interview_id = st.selectbox("Entretien dont l'étape 4 est à restaurer",
                                    [f["ingestion"]["interview_id"] for f in ready], key="restore4_interview")
        uploads = st.file_uploader("Les 2 fichiers JSON de l'étape 4", type=["json"], accept_multiple_files=True,
                                   key="restore4_files")
        if st.button("Restaurer l'étape 4 depuis ces fichiers (0 appel API)", key="restore4_launch",
                     disabled=not uploads):
            try:
                st.session_state.last_run = stage4_restore.restore_stage4(
                    run, interview_id, [(u.name, u.getvalue()) for u in uploads])
            except stage3_restore.RestoreError as exc:
                st.error("Restauration refusée :\n" + "\n".join(f"- {problem}" for problem in exc.problems))
                return
            except OSError as exc:
                st.error(f"Restauration impossible (écriture) : {exc}")
                return
            st.session_state.restore4_flash = f"{stage4_restore.RESTORED_NOTICE} ({interview_id})."
            st.rerun()


def _claim_row(claim: dict) -> dict:
    anchors = " · ".join(f"« {a['text']} » ({a['turn_id'].rsplit('_', 1)[-1]})" for a in claim["validated_temporal_anchors"])
    label = CLAIM_TYPE_LABELS.get(claim["claim_type"], claim["claim_type"])
    if claim.get("model_claim_type"):
        label += f" (requalifié : {CLAIM_TYPE_LABELS.get(claim['model_claim_type'], claim['model_claim_type'])})"
    return {"id": claim["claim_id"].rsplit("_", 1)[-1], "type": label, "description": claim["description"],
            "appuis": ", ".join(i.rsplit("_", 1)[-1] for i in claim["support_ids"]),
            "contextes": ", ".join(claim["contexts"]), "ancrages temporels": anchors,
            "tours": ", ".join(t.rsplit("_", 1)[-1] for t in claim["evidence_turn_ids"]),
            "confiance": claim["confidence"], "à revoir": ", ".join(claim["review_reasons"]) or ("oui" if claim[
                "needs_review"] else "")}


def render_stage5_workflow(run: dict, available: list[dict]) -> None:
    """Étape 5 sans API : TRACE prépare un paquet par entretien, Claude Code joue le Trajectory Mapper."""
    flash = st.session_state.pop("wf5_flash", None)
    if flash:
        (st.warning if flash[0] == "warning" else st.success)(flash[1])
    st.info(
        "**Workflow Claude Code** — aucune clé API, aucun appel API, aucun coût. TRACE prépare sans IA la "
        "représentation de l'entretien (épisodes de l'étape 4, ancrages temporels, régularités) et UN paquet de tâche "
        "par entretien ; Claude Code joue le Trajectory Mapper (prompt du dépôt) ; TRACE valide la réponse (schéma, "
        "validateur de l'étape 5 avec ses requalifications) et enregistre les sorties habituelles. Une étape déjà "
        "terminée et à jour n'est pas rejouée. Procédure : `TRACE_WORKFLOW.md`."
    )
    ids = [f["ingestion"]["interview_id"] for f in available]
    choice = st.selectbox("Entretien(s) pour l'étape 5", ids + ([ACC_ALL] if len(ids) > 1 else []), key="wf5_interview")
    selected = ids if choice == ACC_ALL else [choice]
    if st.button("Préparer / reprendre l'étape 5 (workflow Claude Code, 0 appel API)", type="primary",
                 key="wf_stage5"):
        try:
            result = workflow.run_stage5(run, selected)
        except (OSError, ValueError, KeyError) as exc:
            logging.error("Workflow de l'étape 5 interrompu : %s", type(exc).__name__)
            st.error(f"Workflow de l'étape 5 interrompu ({type(exc).__name__}) : vérifiez les fichiers du run.")
            return
        st.session_state.last_run = result["metadata"]
        status = result["status"]["status"]
        st.session_state.wf5_flash = (
            ("success", "Étape 5 terminée (workflow Claude Code, 0 appel API).") if status == workflow.STAGE_COMPLETE
            else ("warning", f"Étape 5 : {workflow.STATUS_LABELS[status]} — voir ci-dessous."))
        st.rerun()
    render_workflow_state(run, "5", f"Exécute TRACE sur le run {run['run_id']} jusqu'à l'étape 5")


def render_stage5_results(run: dict) -> None:
    done = [f for f in run["files"] if "trajectory" in f]
    if not done:
        return
    usage = (run.get("last_trajectory") or {}).get("usage")
    if usage:
        st.markdown(f"**Dernière exécution (étape 5) :** {usage['api_calls']} appel(s) API — "
                    f"{format_count(usage['input_tokens'])} tokens entrée — {format_count(usage['output_tokens'])} "
                    f"tokens sortie — {usage['cached_results']} résultat(s) repris du cache TRACE")
    st.table([{
        "Entretien": s["interview_id"],
        "Statut étape 4": STAGE4_STATE_LABELS.get(s.get("stage4_status"), s.get("stage4_status") or "n/a"),
        "Étape 5": ai_status_label(s, ACC_STATUS_ICONS),
        "Configuration": CONFIGURATION_LABELS.get(s.get("configuration_type"), "—"),
        "Affirmations retenues": s.get("kept_claim_count") or 0,
        "Frontières stables": s.get("stable_boundaries_count") or 0,
        "Variations contextuelles": s.get("contextual_variations_count") or 0,
        "Changements temporels explicites": s.get("explicit_temporal_changes_count") or 0,
        "Exceptions": s.get("exceptions_count") or 0,
        "Tensions": s.get("unresolved_tensions_count") or 0,
        "Zones ordinaires": s.get("ordinary_zones_count") or 0,
        "Critères (métier d'étudiant)": s.get("student_role_criteria_count") or 0,
        "Avertissements": (s.get("validation_warning_count") or 0) + (s.get("validation_error_count") or 0),
        "Appels API": s.get("api_calls") or 0,
    } for s in (f["trajectory"] for f in done)])
    for f in done:
        summary = f["trajectory"]
        iid = summary["interview_id"]
        with st.expander(f"Configuration intra-entretien — {iid} — aperçu"):
            if summary["status"] == trajectory.STATUS_BLOCKED:
                st.error((summary.get("error") or {}).get("message") or "Étape 5 bloquée.")
            elif summary.get("error"):
                show = st.info if summary["error"].get("code") == workflow.AWAITING_AGENT else st.error
                show(f"Trajectory Mapper : {summary['error']['message']}")
            if summary["status"] == analysis.STATUS_PARTIAL:
                st.warning("Analyse INCOMPLÈTE : l'étape 4 est partielle (analysis_complete = false).")
            if not trajectory.trajectory_current(summary):
                st.warning("Résultat PÉRIMÉ : l'étape 4 (ou l'étape 3) de cet entretien a changé depuis ; relancez l'étape 5.")
            doc_json = read_analysis_file(summary, config.STUDENT_TRAJECTORY_FILENAME)
            validation_json = read_analysis_file(summary, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME)
            manifest_json = read_analysis_file(summary, config.STUDENT_TRAJECTORY_MANIFEST_FILENAME)
            if doc_json:
                document = json.loads(doc_json)
                line = (f"Configuration : **{CONFIGURATION_LABELS.get(document['configuration_type'])}** · "
                        f"étape 4 : **{STAGE4_STATE_LABELS.get(document['stage4_status'])}**"
                        + (" (restaurée depuis fichiers)" if document.get("stage4_restored") else "")
                        + f" · épisodes utilisables : **{document['material']['usable_episode_count']}** · pratiques "
                        f"sans marqueur : **{document['material']['unmarked_practice_count']}** · ancrages temporels : "
                        f"**{document['material']['temporal_anchor_count']}** · "
                        + ("repris du cache TRACE" if document["cache_hit"] else
                           "réponse d'agent du workflow Claude Code (0 appel API)"
                           if document.get("model") == workflow.WORKFLOW_MODEL else
                           "1 appel" if document["llm_called"] else "aucun appel (matériau insuffisant)"))
                st.markdown(line)
                if document.get("model_configuration_type"):
                    st.warning(f"Configuration proposée par le modèle : "
                               f"{CONFIGURATION_LABELS.get(document['model_configuration_type'])} — requalifiée par TRACE "
                               "(aucun changement temporel explicite validé).")
                st.markdown(f"**Synthèse** — {document['trajectory_summary']}")
                kept = [c for c in document["trajectory_claims"] if c["usable_for_next_stages"]]
                st.markdown(" · ".join(f"{CLAIM_TYPE_LABELS[t]} : **{len(document[key])}**"
                                       for t, key in trajectory.validator.LIST_KEYS.items()))
                if kept:
                    order = list(CLAIM_TYPE_LABELS)
                    st.markdown(f"**Affirmations retenues** ({len(kept)})")
                    st.dataframe([_claim_row(c) for c in sorted(kept, key=lambda c: order.index(c["claim_type"]))],
                                 hide_index=True)
                criteria = [c for c in document["student_role_criteria"] if c["usable_for_next_stages"]]
                if criteria:
                    st.markdown(f"**Critères du métier d'étudiant mobilisés dans cet entretien** ({len(criteria)})")
                    st.dataframe([{"id": c["criterion_id"].rsplit("_", 1)[-1], "critère": c["criterion"],
                                   "description": c["description"],
                                   "épisodes": ", ".join(i.rsplit("_", 1)[-1] for i in c["support_ids"]),
                                   "tours": ", ".join(t.rsplit("_", 1)[-1] for t in c["evidence_turn_ids"]),
                                   "confiance": c["confidence"], "à revoir": ", ".join(c["review_reasons"])}
                                  for c in criteria], hide_index=True)
            if validation_json:
                validation = json.loads(validation_json)
                issues = [i for i in validation.get("issues", []) if i["severity"] != "info"]
                if issues:
                    st.markdown(f"**Avertissements de validation** ({len(issues)})")
                    st.dataframe([{"objet": i["object_id"].rsplit("_", 1)[-1], "code": i["code"], "gravité": i["severity"],
                                   "message": i["message"]} for i in issues], hide_index=True)
            cols = st.columns(3)
            for col, content, filename in ((cols[0], doc_json, config.STUDENT_TRAJECTORY_FILENAME),
                                           (cols[1], validation_json, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME),
                                           (cols[2], manifest_json, config.STUDENT_TRAJECTORY_MANIFEST_FILENAME)):
                if content:
                    col.download_button(f"Télécharger {filename}", data=content, file_name=f"{iid}_{filename}",
                                        mime="application/json", key=f"dl_traj_{iid}_{filename}")


def render_stage5_section(run: dict) -> None:
    st.header("Étape 5 — Configuration et trajectoire intra-entretien")
    st.markdown(
        "À l'intérieur d'**un seul entretien**, à partir des épisodes de l'étape 4 (jamais de l'entretien entier) : "
        "quelles frontières, règles et manières de rendre compte de ses usages se **répètent**, restent **stables**, "
        "**varient** selon les tâches, comportent des **exceptions**, entrent en **tension**, ou **changent "
        "explicitement** dans le temps ; quelles zones sont racontées **sans justification** ; quels critères du "
        "« métier d'étudiant » sont mobilisés. L'ordre de l'entretien n'est jamais un ordre biographique : un "
        "changement temporel exige des ancrages explicites (« au lycée », « maintenant »…), sinon TRACE le requalifie "
        "en variation contextuelle. Aucune comparaison entre entretiens à cette étape."
    )
    render_stage4_restore(run)
    run = st.session_state.get("last_run") or run
    available = [f for f in analysis.eligible_files(run) if trajectory.stage4_state(_interview_dir(f))["status"]
                 in (trajectory.STAGE4_COMPLETE, trajectory.STAGE4_PARTIAL)]
    if not available:
        st.info("Lancez d'abord l'étape 4, ou restaurez ses résultats depuis des fichiers déjà téléchargés : l'étape 5 "
                "en consomme les épisodes.")
    else:
        render_stage5_workflow(run, available)
    render_stage5_results(st.session_state.get("last_run") or run)


# --- Étape 6 — comparaison inter-entretiens -----------------------------------------------------------------

MODE_LABELS = {cross_interview_corpus.MODE_BLOCKED: "bloqué (moins de deux entretiens exploitables)",
               cross_interview_corpus.MODE_EXPLORATORY: "EXPLORATOIRE (deux entretiens)",
               cross_interview_corpus.MODE_COMPARATIVE: "comparatif"}
CROSS_TYPE_LABELS = {
    "recurring_boundary": "frontière récurrente", "divergent_boundary": "frontières divergentes",
    "recurring_accounting_move": "manière de rendre compte récurrente",
    "divergent_accounting_move": "manières de rendre compte divergentes",
    "recurring_student_role_criterion": "critère du métier d'étudiant récurrent",
    "divergent_student_role_criterion": "critères du métier d'étudiant divergents",
    "ordinary_zone_pattern": "zone ordinaire", "exception_pattern": "exception", "contextual_association":
    "association contextuelle (observée, non causale)", "temporal_pattern": "changement temporel validé",
    "unresolved_cross_case_contrast": "contraste non résolu", "minority_configuration": "configuration minoritaire",
    "negative_case": "cas négatif"}
POSITION_LABELS = {"explicit_presence": "présence explicite", "explicit_refusal": "refus explicite",
                   "contrary_case": "cas contraire", "divergent_variant": "variante", "not_observed": "non observé",
                   "negative_case": "cas négatif"}
DIFFERENCE_KEYS = ("divergent_boundaries", "divergent_accounting_moves", "divergent_student_role_criteria",
                   "contextual_associations", "unresolved_cross_case_contrasts", "minority_configurations")


def _stage6_uploads(run: dict | None) -> list[tuple[str, bytes]]:
    uploads = st.file_uploader("Sorties de l'étape 5 (3 JSON par entretien : student_trajectory, validation, manifest)",
                               type=["json"], accept_multiple_files=True, key="stage6_files")
    files = [(u.name, u.getvalue()) for u in uploads or []]
    run_files = cross_interview_corpus.run_stage5_uploads(run) if run else []
    if run_files and st.checkbox(f"Inclure les sorties de l'étape 5 du run en cours ({len(run_files)} fichier(s))",
                                 value=True, key="stage6_include_run"):
        files += run_files
    return files


def render_stage6_corpus(prepared) -> None:
    checked = prepared.checked
    st.markdown(f"**N importés : {checked['n_total']}** · **N exploitables : {checked['n_usable']}** · mode : "
                f"**{MODE_LABELS[checked['mode']]}**")
    st.table([{"interview_id": r["interview_id"],
               "statut": ("✅ " if r["status"] == cross_interview_corpus.STATUS_USABLE else "❌ ") + r["status"],
               "claims": r["claims"] if r["claims"] is not None else "—",
               "needs_review": r["needs_review"] if r["needs_review"] is not None else "—",
               "configuration": CONFIGURATION_LABELS.get(r["configuration"], r["configuration"] or "—"),
               "importé": ", ".join(f["imported_name"] for f in r["files"].values())} for r in checked["rows"]])
    for r in checked["rows"]:
        if r["reasons"]:
            st.error(f"{r['interview_id']} exclu : " + " ".join(r["reasons"]))
        for warning in r["warnings"]:
            st.caption(f"{r['interview_id']} : {warning}")
    for item in checked["unrecognized"]:
        st.warning(f"Fichier ignoré — {item['file']} : {item['reason']}")
    for note in checked["notes"]:
        st.caption(note)
    if checked["mode"] == cross_interview_corpus.MODE_BLOCKED:
        st.error("Étape 6 bloquée : moins de deux entretiens exploitables. Importez au moins deux triplets valides.")
    elif checked["mode"] == cross_interview_corpus.MODE_EXPLORATORY:
        st.warning("Deux entretiens exploitables seulement : la comparaison sera marquée EXPLORATOIRE (« présent dans "
                   "les deux entretiens disponibles », jamais une généralité).")


def render_stage6_workflow(files: list[tuple[str, bytes]], prepared) -> None:
    """Étape 6 sans API : TRACE prépare UN paquet pour le corpus, Claude Code joue le Cross-Interview Comparator."""
    flash = st.session_state.pop("wf6_flash", None)
    if flash:
        (st.warning if flash[0] == "warning" else st.success)(flash[1])
    st.info(
        "**Workflow Claude Code** — aucune clé API, aucun appel API, aucun coût. TRACE prépare sans IA la "
        f"représentation du corpus (≈ {format_count(prepared.estimated_input_tokens)} tokens estimés) et UN paquet "
        "de tâche ; Claude Code joue le Cross-Interview Comparator (prompt du dépôt) ; TRACE valide la réponse (schéma, "
        "validateur de l'étape 6 : appuis, comptes en entretiens, non-observation, cas négatifs, requalifications) et "
        "enregistre les sorties habituelles. Un corpus déjà analysé à l'identique n'est pas rejoué. Procédure : "
        "`TRACE_WORKFLOW.md`.")
    if prepared.estimated_input_tokens > cross_interview.SINGLE_CALL_MAX_INPUT_TOKENS:
        st.warning(f"Représentation au-delà du seuil d'un appel unique ({cross_interview.SINGLE_CALL_MAX_INPUT_TOKENS} "
                   "tokens) : paquet unique conservé, signalé (PAYLOAD_OVER_THRESHOLD).")
    if st.button("Préparer / reprendre l'étape 6 (workflow Claude Code, 0 appel API)", type="primary",
                 key="wf_stage6"):
        try:
            result = workflow.run_stage6(files)
        except (OSError, ValueError, KeyError) as exc:
            logging.error("Workflow de l'étape 6 interrompu : %s", type(exc).__name__)
            st.error(f"Workflow de l'étape 6 interrompu ({type(exc).__name__}).")
            return
        st.session_state.stage6_last = result["manifest"]
        status = result["status"]["status"]
        st.session_state.wf6_flash = (
            ("success", "Étape 6 terminée (workflow Claude Code, 0 appel API).") if status == workflow.STAGE_COMPLETE
            else ("warning", f"Étape 6 : {workflow.STATUS_LABELS[status]} — voir ci-dessous."))
        st.rerun()
    state = workflow.read_corpus_status(cross_interview.corpus_dir_for(prepared))
    if not state:
        return
    st.markdown(f"**Workflow Claude Code (étape 6) :** {workflow.STATUS_LABELS[state['status']]} — "
                f"{state['answered_count']}/{state['task_count']} tâche(s) avec une réponse conforme — "
                f"{state['api_calls']} appel API")
    for warning in state["warnings"]:
        st.warning(warning)
    todo = [(t, "réponse à corriger") for t in state["invalid"]] + [(t, "en attente") for t in state["pending"]]
    if todo:
        st.table([{"Tâche": t["task_id"], "Corpus": state["corpus_id"], "Agent": t["agent_label"], "État": label,
                   "Erreur": t.get("error") or ""} for t, label in todo])
        st.markdown("Dans **Claude Code**, ouvert dans ce dépôt, demandez :")
        st.code(f"Exécute TRACE Stage 6 sur le corpus {state['corpus_id']}", language=None)
        st.caption("Claude Code joue le Comparator (un sous-agent), valide sa réponse "
                   f"(`{workflow.CLI} check …`), puis rejoue l'étape 6 (`{workflow.CLI} stage6 --corpus "
                   f"{state['corpus_id']}`). Revenez ensuite ici et cliquez de nouveau sur « Préparer / reprendre ».")


def _cross_row(claim: dict) -> dict:
    label = CROSS_TYPE_LABELS.get(claim["claim_type"], claim["claim_type"])
    if claim.get("model_claim_type"):
        label += f" (requalifié : {CROSS_TYPE_LABELS.get(claim['model_claim_type'], claim['model_claim_type'])})"
    return {"id": claim["cross_claim_id"], "type": label, "description": claim["description_display"],
            "entretiens": f"{claim['n_supporting_interviews']}/{claim['corpus_n_usable']} : "
                          + ", ".join(claim["interview_ids"]),
            "contre-exemples": ", ".join(f"{c['interview_id']} ({POSITION_LABELS.get(c['relation'], c['relation'])})"
                                         for c in claim["counterexamples"]),
            "non observés": claim["n_not_observed_interviews"],
            "appuis à revoir": ", ".join(claim["review_only_interview_ids"]),
            "confiance": claim["confidence"],
            "à revoir": ", ".join(claim["review_reasons"]) or ("oui" if claim["needs_review"] else "")}


def render_stage6_results(last: dict | None, current_corpus_id: str | None = None) -> None:
    if not last:
        return
    if current_corpus_id and last.get("corpus_id") != current_corpus_id:
        st.warning(f"Résultat affiché : corpus précédent ({last.get('corpus_id')}), différent des fichiers actuellement "
                   "importés — relancez la comparaison pour ce corpus.")
    outputs = cross_interview.read_outputs(Path(last["corpus_dir"]))
    manifest = json.loads(outputs["manifest"]) if outputs["manifest"] else last
    origin = ("repris du cache TRACE" if manifest.get("cache_hit") else
              "réponse d'agent du workflow Claude Code" if manifest.get("model") == workflow.WORKFLOW_MODEL else
              "nouvel appel" if manifest.get("llm_called") else "aucun appel")
    st.markdown(f"**Dernière exécution (étape 6) :** {manifest.get('api_calls') or 0} appel(s) API — {origin}"
                f" — statut **{ai_status_label(manifest, ACC_STATUS_ICONS)}** — "
                f"N importés : **{manifest['corpus_n_total']}** · N exploitables : **{manifest['corpus_n_usable']}**"
                f" · aucun appel aux étapes 3, 4 ou 5")
    if manifest.get("error"):
        show = st.info if manifest["error"].get("code") == workflow.AWAITING_AGENT else st.error
        show(manifest["error"].get("message") or manifest["error"].get("code"))
    if not outputs["comparison"]:
        return
    document = json.loads(outputs["comparison"])
    if document["exploratory"]:
        st.warning("Comparaison EXPLORATOIRE : deux entretiens seulement — rien n'est généralisable.")
    kept = [c for c in document["cross_case_claims"] if c["usable_for_next_stages"]]
    st.markdown(" · ".join([f"Affirmations retenues : **{len(kept)}**",
                            f"frontières récurrentes : **{len(document['recurring_boundaries'])}**",
                            f"différences : **{sum(len(document[k]) for k in DIFFERENCE_KEYS)}**",
                            f"critères du métier d'étudiant : **{len(document['student_role_criterion_patterns'])}**",
                            f"zones ordinaires : **{len(document['ordinary_zone_patterns'])}**",
                            f"exceptions : **{len(document['exception_patterns'])}**",
                            f"cas négatifs : **{len(document['negative_cases'])}**",
                            f"à revoir : **{document['cross_claims_needing_review_count']}**"]))
    st.markdown("**Configurations (étape 5), comptées en entretiens**")
    st.dataframe([{"configuration": CONFIGURATION_LABELS.get(k, k), "entretiens": v["share"],
                   "interview_ids": ", ".join(v["interview_ids"])}
                  for k, v in document["configuration_distribution"].items()], hide_index=True)
    st.markdown(f"**Synthèse** — {document['cross_case_summary_display']}")
    if kept:
        st.markdown(f"**Affirmations inter-entretiens** ({len(kept)})")
        st.dataframe([_cross_row(c) for c in kept], hide_index=True)
    if document["negative_cases"]:
        st.markdown(f"**Cas négatifs** ({len(document['negative_cases'])})")
        st.dataframe([{"id": n["negative_case_id"], "entretien": n["interview_id"],
                       "relation": ", ".join(POSITION_LABELS.get(r, r) for r in n["relations"]),
                       "complique": ", ".join(n["related_cross_claim_ids"]), "description": " / ".join(n["descriptions"]),
                       "appuis (étape 5)": ", ".join(n["support_ids"]), "statut": n["validation_status"],
                       "à revoir": "oui" if n["needs_review"] else ""} for n in document["negative_cases"]],
                     hide_index=True)
    with st.expander("Détail par catégorie (frontières, différences, critères, zones ordinaires, exceptions)"):
        by_id = {c["cross_claim_id"]: c for c in document["cross_case_claims"]}
        for title, keys in (("Frontières récurrentes", ("recurring_boundaries",)), ("Différences", DIFFERENCE_KEYS),
                            ("Zones ordinaires", ("ordinary_zone_patterns",)),
                            ("Exceptions", ("exception_patterns",)), ("Changements temporels validés", (
                                "temporal_patterns",))):
            ids = [i for k in keys for i in document[k]]
            st.markdown(f"**{title}** ({len(ids)})")
            for i in ids:
                c = by_id[i]
                st.caption(f"{i} — {c['description_display']} — {c['n_supporting_interviews']}/{c['corpus_n_usable']} "
                           f"entretiens : {', '.join(c['interview_ids'])}")
        patterns = document["student_role_criterion_patterns"]
        st.markdown(f"**Critères du métier d'étudiant** ({len(patterns)})")
        if patterns:
            st.dataframe([{"id": p["cross_claim_id"], "critère": p["criterion_label"] or "",
                           **{POSITION_LABELS[k]: ", ".join(v) for k, v in p["positions"].items()},
                           "à revoir": ", ".join(p["review_reasons"])} for p in patterns], hide_index=True)
    validation = json.loads(outputs["validation"]) if outputs["validation"] else {}
    issues = [i for i in validation.get("issues", []) if i["severity"] != "info"]
    if issues:
        st.markdown(f"**Avertissements de validation** ({len(issues)})")
        st.dataframe([{"objet": i["object_id"], "code": i["code"], "gravité": i["severity"], "message": i["message"]}
                      for i in issues], hide_index=True)
    cols = st.columns(3)
    for col, key, filename in ((cols[0], "comparison", config.CROSS_INTERVIEW_COMPARISON_FILENAME),
                               (cols[1], "validation", config.CROSS_INTERVIEW_VALIDATION_FILENAME),
                               (cols[2], "manifest", config.CROSS_INTERVIEW_MANIFEST_FILENAME)):
        if outputs[key]:
            col.download_button(f"Télécharger {filename}", data=outputs[key], file_name=filename,
                                mime="application/json", key=f"dl_stage6_{key}")


def render_stage6_section(run: dict | None) -> None:
    st.header("Étape 6 — Comparaison inter-entretiens")
    st.markdown(
        "Compare **plusieurs entretiens** à partir des seules sorties **validées de l'étape 5** (jamais les "
        "transcriptions) : quelles **frontières**, quels **critères du métier d'étudiant**, quelles **manières de "
        "rendre compte**, **zones ordinaires**, **exceptions** et **tensions** reviennent, varient ou s'opposent d'un "
        "entretien à l'autre, et quels **cas négatifs** compliquent ces régularités. Les comptes portent sur des "
        "**entretiens** (jamais des occurrences), un critère non mentionné est **non observé** (jamais « absent »), "
        "et aucune typologie de personnes n'est construite.")
    st.subheader("Constituer le corpus Stage 6")
    st.markdown(
        "Importez les **3 JSON de l'étape 5** de chaque entretien (`student_trajectory.json`, "
        "`student_trajectory_validation.json`, `student_trajectory_manifest.json`, préfixés ou non). Ils sont reconnus "
        "par leur contenu, regroupés par `interview_id` et vérifiés (versions, analyse complète, aucune erreur de "
        "validation, empreintes et comptes cohérents, aucune altération, pas de doublon ambigu). Un entretien invalide "
        "est exclu avec sa raison sans bloquer les autres. **Aucun appel API** pour cet import.")
    files = _stage6_uploads(run)
    prepared = None
    if not files:
        st.info("Aucune sortie de l'étape 5 importée.")
    else:
        prepared = cross_interview.prepare_stage6(files)
        render_stage6_corpus(prepared)
        if not prepared.blocked:
            render_stage6_workflow(files, prepared)
    render_stage6_results(st.session_state.get("stage6_last"), prepared.corpus_id if prepared else None)


st.set_page_config(page_title="TRACE", layout="wide")

# 1. Titre et 2. introduction
st.title("TRACE")
st.subheader("Analyse ethnométhodologique des usages étudiants des IAG")
st.markdown(
    "Outil **expérimental** d'analyse qualitative assistée par IA. "
    "L'ingestion des entretiens est **déterministe** et sans IA : "
    "extraction du texte, découpage en tours de parole et contrôle qualité. "
    "Les étapes 3 à 6 (agents IA descriptifs) forment un **workflow multi-agents exécuté dans Claude Code** "
    "(voir TRACE_WORKFLOW.md) : TRACE prépare les tâches et valide les réponses des agents ; aucune clé ni aucun "
    "appel API, rien n'est lancé au chargement des fichiers."
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

with st.expander("Ouvrir un run existant (par exemple préparé dans Claude Code)"):
    runs = list_runs(config.OUTPUTS_DIR)
    if not runs:
        st.caption("Aucun run enregistré sur ce disque.")
    else:
        chosen_run = st.selectbox("Run", runs, key="open_run_id")
        if st.button("Ouvrir ce run", key="open_run"):
            try:
                st.session_state.last_run = load_metadata(config.OUTPUTS_DIR / chosen_run)
                st.rerun()
            except (TraceError, OSError, ValueError) as exc:
                st.error(f"Run illisible : {exc}")

# 5. Pipeline (structuration + étape 3 ; affichée après le lancement)
st.header("Pipeline")
st.caption(
    "Actives : la structuration des entretiens (sans IA) et, sur action explicite, "
    "l'extraction des pratiques et l'analyse interactionnelle (étape 3), puis la construction des épisodes "
    "d'accountability (étape 4), la configuration intra-entretien (étape 5) et la comparaison inter-entretiens "
    "(étape 6). Les autres étapes restent inactives."
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
    # 9. Étape 4 — épisodes d'accountability (jamais automatique, après l'étape 3)
    render_stage4_section(st.session_state.get("last_run") or run)
    # 10. Étape 5 — configuration et trajectoire intra-entretien (jamais automatique, après l'étape 4)
    render_stage5_section(st.session_state.get("last_run") or run)
else:
    st.info("Aucun run lancé pendant cette session.")

# 11. Étape 6 — comparaison inter-entretiens (sorties de l'étape 5 importées ; aucun run nécessaire)
render_stage6_section(st.session_state.get("last_run"))

# Le pipeline est rempli en dernier : il reflète aussi une analyse IA lancée pendant cette exécution.
with pipeline_area:
    render_pipeline(st.session_state.get("last_run"))
