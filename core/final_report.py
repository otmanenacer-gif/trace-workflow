"""Rapport individuel expérimental (anciennement présenté comme « étape 7 — rapport final »), DÉTERMINISTE :
aucun appel au modèle, aucun recalcul. L'étape 7 du pipeline corpus est la théorisation transversale
(core/stage7_theory.py).

Assemble, pour chaque entretien d'un run, les sorties déjà VALIDÉES des étapes 3 à 5 :

    étape 5 (student_trajectory.json)      → résumé exécutif, configuration, trajectoires, critères, tensions
    étape 4 (accountability_episodes.json) → principaux épisodes d'accountability, citations représentatives
    étape 3 (practice_extractor.json, interaction_signals.json, evidence_validation.json) → pratiques, comptes
    + `needs_review`, limites méthodologiques et encadré sur l'étape 6

Rien n'est interprété à nouveau : les textes repris (résumés, descriptions, opérations) sont ceux des étapes
précédentes, signalés comme tels ; les citations sont exactes (validées mot pour mot) ; une information absente est
déclarée absente, jamais inventée. Étape 6 : avec moins de deux entretiens exploitables, la comparaison
inter-entretiens est NON APPLICABLE et l'encadré méthodologique le dit.

Rapport EXPÉRIMENTAL, assisté par modèle local, à relire qualitativement. Une relecture humaine peut signaler des
épisodes dont l'interprétation est à vérifier (`to_verify` : identifiants E…) : ils portent alors une mention visible,
sans que leur contenu soit modifié ; la liste est conservée avec le rapport (review_flags.json) et reprise à chaque
régénération.

Sorties : <run>/final_report/final_report.json et final_report.md.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from core import analysis, config
from core.analysis_cache import write_json_atomic
from core.schemas import SPEAKER_INTERVIEWER

REPORT_VERSION = "1.1"  # 1.1 : rapport expérimental, interprétations à vérifier signalées
TITLE = "Rapport final expérimental — résultats assistés par modèle local, à relire qualitativement"
TO_VERIFY_LABEL = "interprétation à vérifier"
FLAGS_FILENAME = "review_flags.json"
REPORT_DIRNAME = "final_report"
JSON_FILENAME = "final_report.json"
MD_FILENAME = "final_report.md"
STATUS_COMPLETE, STATUS_PARTIAL = "COMPLETE", "PARTIAL"
STAGE6_NOT_APPLICABLE = "NOT_APPLICABLE_SINGLE_INTERVIEW"
STAGE6_NOT_APPLICABLE_REASON = "La comparaison inter-entretiens nécessite au moins deux entretiens exploitables."
MAX_QUOTES = 6
QUOTE_MAX_CHARS = 350
MISSING = "information absente des sorties validées"

CONFIGURATION_LABELS = {"temporal_trajectory": "trajectoire temporelle explicite",
                        "contextual_configuration": "configuration contextuelle",
                        "mixed": "mixte (changement temporel explicite et variations contextuelles)",
                        "no_clear_pattern": "pas de configuration nette"}
CLAIM_TYPE_LABELS = {"stable_boundary": "frontière stable", "recurring_accounting_move": "opération récurrente",
                     "contextual_variation": "variation contextuelle",
                     "explicit_temporal_change": "changement temporel explicite", "exception": "exception",
                     "unresolved_tension": "tension non résolue", "ordinary_zone": "zone ordinaire"}
STATUS_LABELS = {"accountability_episode": "épisode d'accountability", "ordinary_practice": "pratique ordinaire",
                 "uncertain": "incertain"}
TENSION_KEYS = ("unresolved_tensions", "exceptions", "contextual_variations", "explicit_temporal_changes")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _short(identifier: str | None) -> str:
    return (identifier or "?").rsplit("_", 1)[-1]


def _clip(quote: str) -> str:
    return quote if len(quote) <= QUOTE_MAX_CHARS else quote[:QUOTE_MAX_CHARS].rstrip() + " […]"


def _real(value):
    return None if value in (None, "", "null") else value


# --- Un entretien --------------------------------------------------------------------------------------------

def interview_report(info: dict, to_verify: list[str] | None = None) -> dict:
    """Section d'un entretien, à partir de ses seules sorties validées sur le disque."""
    from core import local_pipeline as lp
    interview_dir = Path(info["ingestion"]["output_dir"])
    analysis_dir = interview_dir / config.ANALYSIS_SUBDIR
    interview_id = info["ingestion"]["interview_id"]
    states = {stage: lp.stage_state(stage, info)["status"] for stage in lp.STAGES}
    transcript = _read(interview_dir / config.STRUCTURED_TRANSCRIPT_FILENAME) or {"turns": []}
    speaker = {t["turn_id"]: t["speaker"] for t in transcript["turns"]}
    files = {name: analysis_dir / name for name in (
        analysis.PRACTICE.output_filename, analysis.INTERACTION.output_filename, config.EVIDENCE_VALIDATION_FILENAME,
        config.ACCOUNTABILITY_EPISODES_FILENAME, config.STUDENT_TRAJECTORY_FILENAME)}
    practices_doc = _read(files[analysis.PRACTICE.output_filename]) or {}
    signals_doc = _read(files[analysis.INTERACTION.output_filename]) or {}
    evidence_doc = _read(files[config.EVIDENCE_VALIDATION_FILENAME]) or {}
    episodes_doc = _read(files[config.ACCOUNTABILITY_EPISODES_FILENAME]) if states["4"] != "NOT_RUN" else None
    trajectory_doc = _read(files[config.STUDENT_TRAJECTORY_FILENAME]) if states["5"] != "NOT_RUN" else None
    missing = [f"Étape {stage} : {status}" for stage, status in states.items() if status != "COMPLETE"]

    # étape 3 : pratiques
    practices = practices_doc.get("practices", [])
    by_status: dict[str, int] = {}
    for p in practices:
        by_status[p.get("use_status") or "?"] = by_status.get(p.get("use_status") or "?", 0) + 1
    practice_by_id = {p["practice_id"]: p for p in practices}

    # étape 4 : épisodes
    episodes = (episodes_doc or {}).get("episodes", [])
    usable = [e for e in episodes if e.get("usable_for_next_stages")]
    main = [e for e in usable if e.get("episode_status") == "accountability_episode"]

    def quotes_of(episode: dict, limit: int | None = None) -> list[dict]:
        """Citations validées de l'enquêté·e ; jamais un tour de l'enquêteur ni un tour dont l'attribution du locuteur
        est douteuse (avertissement de l'audit des locuteurs)."""
        doubtful = {w["turn_id"] for w in episode.get("speaker_warnings", [])}
        out = [{"turn_id": e["turn_id"], "speaker": speaker.get(e["turn_id"], "unknown"), "quote": _clip(e["quote"]),
                "truncated": len(e["quote"]) > QUOTE_MAX_CHARS}
               for e in episode.get("evidence", []) if (e.get("validation") or {}).get("valid")
               and speaker.get(e["turn_id"]) != SPEAKER_INTERVIEWER and e["turn_id"] not in doubtful]
        return out[:limit] if limit else out

    episode_rows = [{
        "episode_id": e["episode_id"], "status": e.get("episode_status"),
        # question pratique reprise seulement si elle en a la forme (sinon : copie du résumé, défaut formel)
        "accountability_problem": (_real(e.get("accountability_problem"))
                                   if (_real(e.get("accountability_problem")) or "").rstrip().endswith("?") else None),
        "summary": e.get("episode_summary"),
        "accounting_moves": [{"type": m.get("type"), "description": m.get("description")}
                             for m in e.get("accounting_moves", [])],
        "boundary_objects": e.get("boundary_objects", []),
        "practices": [{"practice_id": pid, "summary": (practice_by_id.get(pid) or {}).get("summary")}
                      for pid in e.get("practice_ids", [])],
        "quotes": quotes_of(e, 2), "needs_review": bool(e.get("needs_review")),
        "review_reasons": e.get("review_reasons", []), "validation_status": e.get("validation_status"),
        "repaired_by_trace": "trace_repair" in e} for e in main]
    others = [{"episode_id": e["episode_id"], "status": e.get("episode_status"), "summary": e.get("episode_summary"),
               "needs_review": bool(e.get("needs_review"))} for e in usable if e not in main]
    for row in episode_rows + others:  # relecture humaine : mention visible, contenu inchangé
        row["interpretation_to_verify"] = _short(row["episode_id"]) in (to_verify or ())
    representative = []
    for e in main:
        for q in quotes_of(e, 1):
            representative.append({**q, "episode_id": e["episode_id"],
                                   "interpretation_to_verify": _short(e["episode_id"]) in (to_verify or ())})
    representative = representative[:MAX_QUOTES]

    # étape 5 : configuration et trajectoires
    t = trajectory_doc or {}
    claims = [c for c in t.get("trajectory_claims", []) if c.get("usable_for_next_stages")]
    claim_by_id = {c["claim_id"]: c for c in t.get("trajectory_claims", [])}
    trajectories = [{"claim_id": c["claim_id"], "type": c.get("claim_type"), "description": c.get("description"),
                     "episode_ids": c.get("episode_ids", []), "confidence": c.get("confidence"),
                     "needs_review": bool(c.get("needs_review")), "review_reasons": c.get("review_reasons", [])}
                    for c in claims]
    tensions = [{"kind": key, "claim_id": cid, "description": (claim_by_id.get(cid) or {}).get("description")}
                for key in TENSION_KEYS for cid in t.get(key, [])]
    criteria = [{"criterion": c.get("criterion"), "description": c.get("description"),
                 "needs_review": bool(c.get("needs_review"))} for c in t.get("student_role_criteria", [])
                if c.get("usable_for_next_stages", True)]
    boundaries = sorted({b for e in main for b in e.get("boundary_objects", [])})

    # à revoir
    review = []
    if t.get("needs_review"):
        review.append(f"Étape 5 : configuration à revoir (statut {t.get('status')}, {t.get('validation_warning_count', 0)} "
                      "avertissement(s) de validation).")
    review += [f"Étape 5 — {_short(c['claim_id'])} ({CLAIM_TYPE_LABELS.get(c['type'], c['type'])}) : "
               f"{', '.join(c['review_reasons']) or 'à revoir'}" for c in trajectories if c["needs_review"]]
    review += [f"Étape 4 — {_short(e['episode_id'])} ({STATUS_LABELS.get(e.get('episode_status'), '?')}) : "
               f"{', '.join(e.get('review_reasons', [])) or 'à revoir'}" for e in episodes if e.get("needs_review")]
    rejected = [e["episode_id"] for e in episodes if e.get("validation_status") == "rejected"]
    if rejected:
        review.append("Étape 4 : épisode(s) rejeté(s) par la validation, exclus du rapport : "
                      + ", ".join(_short(r) for r in rejected))
    if evidence_doc.get("total_objects_needing_review"):
        review.append(f"Étape 3 : {evidence_doc['total_objects_needing_review']} pratique(s) ou signal(aux) à revoir, "
                      f"{evidence_doc.get('total_invalid_evidence', 0)} citation(s) invalide(s).")
    warned = sorted({w["turn_id"] for e in usable for w in e.get("speaker_warnings", [])})
    if warned:
        review.append("Attribution du locuteur douteuse sur : " + ", ".join(_short(w) for w in warned))

    counts = {"practices": len(practices), "practices_set_aside": len(practices_doc.get("set_aside_practices", [])),
              "practices_by_use_status": by_status, "signals": len(signals_doc.get("signals", [])),
              "episodes": len(episodes), "usable_episodes": len(usable), "accountability_episodes": len(main),
              "ordinary_practices": sum(e.get("episode_status") == "ordinary_practice" for e in usable),
              "uncertain": sum(e.get("episode_status") == "uncertain" for e in usable),
              "rejected_episodes": len(rejected), "trajectory_claims": len(trajectories),
              "trajectory_claims_needing_review": sum(c["needs_review"] for c in trajectories)}
    configuration = t.get("configuration_type")
    summary = _real(t.get("trajectory_summary"))
    executive = (
        f"Entretien {interview_id} — configuration : "
        f"{CONFIGURATION_LABELS.get(configuration, configuration) if configuration else MISSING} (étape 5"
        + (f", confiance {t.get('confidence')}" if t.get("confidence") else "")
        + (", à revoir" if t.get("needs_review") else "") + "). "
        f"Matériau : {counts['practices']} pratique(s) et {counts['signals']} signal(aux) (étape 3) ; "
        f"{counts['usable_episodes']} épisode(s) exploitable(s), dont {counts['accountability_episodes']} épisode(s) "
        f"d'accountability, {counts['ordinary_practices']} pratique(s) ordinaire(s) et {counts['uncertain']} "
        f"incertain(s) (étape 4) ; {counts['trajectory_claims']} affirmation(s) de trajectoire, dont "
        f"{counts['trajectory_claims_needing_review']} à revoir (étape 5).")
    return {
        "interview_id": interview_id, "stage_states": states, "missing": missing,
        "executive_summary": executive,
        "stage5_summary": summary or MISSING,
        "configuration": {"type": configuration, "label": CONFIGURATION_LABELS.get(configuration),
                          "confidence": t.get("confidence"), "needs_review": bool(t.get("needs_review")),
                          "status": t.get("status")} if t else None,
        "trajectories": trajectories, "student_role_criteria": criteria, "tensions": tensions,
        "boundary_objects": boundaries, "accountability_episodes": episode_rows, "other_episodes": others,
        "representative_quotes": representative, "needs_review": review, "counts": counts,
        "models": sorted({m for m in (practices_doc.get("model"), (episodes_doc or {}).get("model"), t.get("model"))
                          if m}),
        "sources": {name: {"sha256": _sha(path), "present": path.is_file()} for name, path in files.items()},
    }


# --- Le run ------------------------------------------------------------------------------------------------

def stage6_box(n_usable: int) -> dict:
    if n_usable < 2:
        return {"status": STAGE6_NOT_APPLICABLE, "reason": STAGE6_NOT_APPLICABLE_REASON,
                "note": f"Le corpus de ce run ne contient que {n_usable} entretien(s) exploitable(s) à l'étape 5 : "
                        "l'étape 6 n'a pas été exécutée, aucune comparaison n'a été produite."}
    return {"status": "NOT_INCLUDED", "reason": None,
            "note": f"{n_usable} entretiens exploitables : la comparaison inter-entretiens relève de l'étape 6, "
                    "exécutée séparément ; elle n'est pas intégrée à ce rapport minimal."}


def normalize_ids(values) -> list[str]:
    """« E005, e9 ; OTMANE_NACER_E011 » → ['E005', 'E009', 'E011']."""
    import re
    out = []
    for token in re.split(r"[\s,;]+", values if isinstance(values, str) else " ".join(values or [])):
        match = re.search(r"E0*(\d+)$", token.strip(), re.IGNORECASE)
        if match:
            out.append(f"E{int(match.group(1)):03d}")
    return list(dict.fromkeys(out))


def build(metadata: dict, to_verify: list[str] | None = None) -> dict:
    to_verify = normalize_ids(to_verify)
    files = analysis.eligible_files(metadata)
    interviews = [interview_report(info, to_verify) for info in files]
    known = {_short(e["episode_id"]) for i in interviews for e in i["accountability_episodes"] + i["other_episodes"]}
    n_usable = sum(i["stage_states"]["5"] == "COMPLETE" for i in interviews)
    limitations = [
        "Rapport assemblé sans aucun appel au modèle à partir des sorties validées des étapes 3 à 5 : aucune "
        "interprétation nouvelle n'y est ajoutée.",
        "Les citations sont exactes (vérifiées mot pour mot sur la transcription) ; les résumés, opérations, "
        "trajectoires et critères ont été produits par un modèle local et restent des propositions à relire.",
        "Le contrôle sémantique des épisodes (Grounding Checker) est expérimental, en rapport seulement : il n'a pas "
        "été appliqué à ce rapport.",
        "Les tours de l'enquêteur, et ceux dont l'attribution du locuteur est douteuse, sont du contexte : ils ne sont "
        "jamais cités comme position de l'étudiant·e.",
    ]
    if len(interviews) == 1:
        limitations.append("Un seul entretien : aucun résultat de ce rapport n'est généralisable.")
    if any(i["missing"] for i in interviews):
        limitations.append("Certaines étapes ne sont pas complètes : les sections correspondantes le signalent.")
    if to_verify:
        limitations.append("Relecture humaine : l'interprétation des épisodes " + ", ".join(to_verify)
                           + f" est à vérifier (mention « {TO_VERIFY_LABEL} ») ; leur contenu n'est pas modifié.")
    return {"report_version": REPORT_VERSION, "title": TITLE, "experimental": True,
            "interpretations_to_verify": to_verify, "unknown_episode_ids": sorted(set(to_verify) - known),
            "generated_at": _now(), "run_id": metadata.get("run_id"),
            "status": STATUS_COMPLETE if interviews and all(not i["missing"] for i in interviews) else STATUS_PARTIAL,
            "llm_called": False, "interview_count": len(interviews), "stage5_usable_interview_count": n_usable,
            "stage6": stage6_box(n_usable), "limitations": limitations, "interviews": interviews}


# --- Markdown ------------------------------------------------------------------------------------------------

def to_markdown(report: dict) -> str:
    flag = f" — ⚠ **{TO_VERIFY_LABEL}**"
    out = [f"# TRACE — {report.get('title', TITLE)}", "",
           f"Run `{report['run_id']}` · généré le {report['generated_at']} · {report['interview_count']} entretien(s) · "
           f"statut **{report['status']}** · aucun appel au modèle pour ce rapport", ""]
    for item in report["interviews"]:
        out += [f"## Entretien {item['interview_id']}", "", "### Résumé exécutif", "", item["executive_summary"], "",
                f"> Synthèse de l'étape 5 (modèle local, à relire) : {item['stage5_summary']}", ""]
        if item["missing"]:
            out += ["**Étapes incomplètes :** " + " ; ".join(item["missing"]), ""]
        out += ["### Trajectoires et configuration (étape 5)", ""]
        if item["configuration"]:
            c = item["configuration"]
            out += [f"Configuration : **{c['label'] or c['type']}** — confiance {c['confidence'] or '?'}"
                    + (" — à revoir" if c["needs_review"] else ""), ""]
        out += [f"- **{CLAIM_TYPE_LABELS.get(x['type'], x['type'])}** ({_short(x['claim_id'])}) : {x['description']} "
                f"— épisodes {', '.join(_short(e) for e in x['episode_ids']) or '—'}"
                + (" — *à revoir*" if x["needs_review"] else "") for x in item["trajectories"]] or [f"- {MISSING}"]
        out += ["", "**Critères du métier d'étudiant mobilisés**", ""]
        out += [f"- {c['criterion']} : {c['description']}" + (" — *à revoir*" if c["needs_review"] else "")
                for c in item["student_role_criteria"]] or [f"- {MISSING}"]
        out += ["", "### Principaux épisodes d'accountability (étape 4)", ""]
        if not item["accountability_episodes"]:
            out += [f"- {MISSING}", ""]
        for e in item["accountability_episodes"]:
            out += [f"**{_short(e['episode_id'])}** — {e['summary']}" + (" — *à revoir*" if e["needs_review"] else "")
                    + (flag if e.get("interpretation_to_verify") else "")]
            if e["accountability_problem"]:
                out.append(f"- Question pratique : {e['accountability_problem']}")
            for m in e["accounting_moves"]:
                out.append(f"- Opération `{m['type']}` : {m['description']}")
            if e["boundary_objects"]:
                out.append(f"- Frontières : {' ; '.join(e['boundary_objects'])}")
            for q in e["quotes"]:
                out.append(f"- « {q['quote']} » ({_short(q['turn_id'])})")
            out.append("")
        if item["other_episodes"]:
            out += ["Autres candidats examinés :", ""]
            out += [f"- {_short(e['episode_id'])} ({STATUS_LABELS.get(e['status'], e['status'])}) : {e['summary']}"
                    + (flag if e.get("interpretation_to_verify") else "") for e in item["other_episodes"]]
            out.append("")
        counts = item["counts"]
        out += ["### Pratiques et tensions", "",
                f"- Pratiques (étape 3) : {counts['practices']} retenues, {counts['practices_set_aside']} écartées par la "
                "sélectivité ; par statut : " + (", ".join(f"{k} {v}" for k, v in counts["practices_by_use_status"].items())
                                                  or MISSING)]
        out += [f"- {CLAIM_TYPE_LABELS.get(CLAIM_KIND.get(x['kind']), x['kind'])} ({_short(x['claim_id'])}) : "
                f"{x['description'] or MISSING}" for x in item["tensions"]]
        if item["boundary_objects"]:
            out.append("- Frontières construites dans les épisodes : " + " ; ".join(item["boundary_objects"]))
        out += ["", "### Citations représentatives (validées)", ""]
        out += [f"- « {q['quote']} » — {_short(q['turn_id'])}, épisode {_short(q['episode_id'])}"
                + (f" ({TO_VERIFY_LABEL})" if q.get("interpretation_to_verify") else "")
                for q in item["representative_quotes"]] or [f"- {MISSING}"]
        out += ["", "### À revoir (`needs_review`)", ""]
        out += [f"- {r}" for r in item["needs_review"]] or ["- Aucun élément signalé."]
        out.append("")
    box = report["stage6"]
    out += ["## Encadré méthodologique", "",
            f"> **Étape 6 — comparaison inter-entretiens : {box['status']}.** "
            + (f"{box['reason']} " if box["reason"] else "") + box["note"], ""]
    out += ["### Limites", ""] + [f"- {x}" for x in report["limitations"]] + [""]
    return "\n".join(out)


CLAIM_KIND = {"unresolved_tensions": "unresolved_tension", "exceptions": "exception",
              "contextual_variations": "contextual_variation", "explicit_temporal_changes": "explicit_temporal_change"}


# --- Écriture ------------------------------------------------------------------------------------------------

def report_dir(metadata: dict) -> Path:
    return Path(metadata["output_dir"]) / REPORT_DIRNAME


def saved_flags(metadata: dict) -> list[str]:
    """Épisodes signalés lors de la dernière génération (relecture humaine), repris par défaut."""
    return list((_read(report_dir(metadata) / FLAGS_FILENAME) or {}).get("interpretations_to_verify", []))


def generate(metadata: dict, to_verify: list[str] | str | None = None) -> dict:
    """Rapport du run : construit, écrit (JSON et Markdown), renvoyé avec ses chemins. Aucun appel au modèle.
    `to_verify` : épisodes dont l'interprétation est à vérifier ; None reprend la liste enregistrée."""
    to_verify = saved_flags(metadata) if to_verify is None else normalize_ids(to_verify)
    report = build(metadata, to_verify)
    directory = report_dir(metadata)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / FLAGS_FILENAME, {"interpretations_to_verify": report["interpretations_to_verify"],
                                                   "updated_at": report["generated_at"]})
    write_json_atomic(directory / JSON_FILENAME, report)
    markdown = to_markdown(report)
    (directory / MD_FILENAME).write_text(markdown, encoding="utf-8")
    return {"report": report, "markdown": markdown, "json_path": directory / JSON_FILENAME,
            "md_path": directory / MD_FILENAME}


def read_existing(metadata: dict) -> dict | None:
    directory = report_dir(metadata)
    if not (directory / JSON_FILENAME).is_file() or not (directory / MD_FILENAME).is_file():
        return None
    return {"report": _read(directory / JSON_FILENAME), "markdown": (directory / MD_FILENAME).read_text(encoding="utf-8"),
            "json_path": directory / JSON_FILENAME, "md_path": directory / MD_FILENAME}
