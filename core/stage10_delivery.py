"""Étape 10 — Livraison finale du rapport corpus : AUCUNE analyse, AUCUN appel au modèle, AUCUN recalcul.

    python scripts/trace_local.py stage10 <run_id>

    <run>/stage9/stage9_validated_report.json + stage9_validation.json (seules sources du contenu) ; métadonnées
    seulement : stage8_report_draft.json, stage7_theory.json, stage6_corpus.json, batch_manifest.json, metadata.json
      → entrée : étape 9 COMPLETE ou SUCCESS_WITH_WARNINGS ; étape 9 absente, BLOCKED ou rapport validé absent →
        étape 10 BLOCKED (les avertissements n'empêchent jamais la livraison)
      → paragraphes validés recopiés MOT POUR MOT, rangés dans les 13 sections finales selon leur section et
        sous-section de l'étape 8 (ordre du plan conservé) ; un paragraphe exclu par l'étape 9 n'apparaît jamais
      → seuls ajouts, tous déterministes : titres, numérotation, table des matières, encadré statistique (lu dans les
        manifestes, jamais estimé), note de transparence méthodologique, annexe de traçabilité
      → <run>/stage10/final_report.md, final_report.json (provenance complète) et delivery_manifest.json.

Statut : COMPLETE (aucun avertissement), SUCCESS_WITH_WARNINGS (livrable avec avertissements : pipeline réussi),
BLOCKED. Relancer avec les mêmes entrées produit le même rapport (seul generated_at du manifeste change).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core import local_pipeline as lp
from core.analysis import eligible_files
from core.analysis_cache import write_json_atomic
from core.stage7_theory import INDIVIDUAL, _read

STAGE10_VERSION = "1.0"
OUT_DIRNAME = "stage10"
REPORT_MD = "final_report.md"
REPORT_JSON = "final_report.json"
MANIFEST = "delivery_manifest.json"
COMPLETE, SUCCESS_WITH_WARNINGS, BLOCKED = "COMPLETE", "SUCCESS_WITH_WARNINGS", "BLOCKED"
DELIVERABLE = (COMPLETE, SUCCESS_WITH_WARNINGS)
TITLE = "Rapport final du corpus TRACE — usages de l'IA générative et travail étudiant (version expérimentale)"
UPSTREAM = {"stage9_validation": Path("stage9") / "stage9_validation.json",
            "stage9_validated_report": Path("stage9") / "stage9_validated_report.json",
            "stage8_report_draft": Path("stage8") / "stage8_report_draft.json",
            "stage7_theory": Path("stage7") / "stage7_theory.json",
            "stage6_corpus": Path("stage6") / "stage6_corpus.json",
            "batch_manifest": Path("batch") / "batch_manifest.json"}
# Sections finales : (numéro, identifiant, titre). Les sections 5 à 10 et 12 reçoivent les paragraphes validés.
SECTIONS = ((1, "title", "Titre"), (2, "executive_summary", "Résumé exécutif"), (3, "corpus", "Corpus analysé"),
            (4, "methodology", "Méthodologie TRACE"), (5, "results", "Résultats principaux"),
            (6, "logics", "Configurations et logiques d'usage"),
            (7, "boundaries", "Frontières et critères du travail étudiant"),
            (8, "variations", "Variations et tensions"), (9, "negative_cases", "Cas négatifs et exceptions"),
            (10, "discussion", "Discussion"), (11, "limitations", "Limites"), (12, "conclusion", "Conclusion"),
            (13, "annex", "Annexe de traçabilité"))
# (section de l'étape 8, sous-section) -> section finale ; à défaut, section par défaut de la section de l'étape 8.
ROUTES = {("results", "logics"): "logics", ("results", "boundaries"): "boundaries",
          ("results", "criteria"): "boundaries", ("variations", "trajectories"): "logics",
          ("variations", "negative_cases"): "negative_cases"}
DEFAULT_ROUTE = {"results": "results", "variations": "variations", "discussion": "discussion",
                 "conclusion": "conclusion"}
# Lignes devenues fausses une fois le brouillon validé (purement techniques) : retirées des limites.
OBSOLETE_LIMITATIONS = ("Brouillon expérimental, non validé",)
INDIVIDUAL_MARK = "*(Hypothèse — un seul entretien)* "
MIXED_MARK = "*(Hypothèses individuelles et régularités mêlées)* "
HYPOTHESIS_STARTS = ("Hypothèse issue d'un seul entretien",)
STAT_LABELS = (("planned", "Entretiens prévus"), ("ingested", "Entretiens ingérés"),
               ("analysed_stage5", "Entretiens analysés jusqu'à l'étape 5"),
               ("included_stage6", "Entretiens inclus à l'étape 6"), ("excluded", "Entretiens exclus"),
               ("theory_claims", "Propositions théoriques (étape 7)"), ("stage8_paragraphs", "Paragraphes rédigés (étape 8)"),
               ("stage9_pass", "Paragraphes PASS (étape 9)"), ("stage9_warn", "Paragraphes WARN (étape 9)"),
               ("stage9_fail", "Paragraphes FAIL (étape 9)"), ("stage9_excluded", "dont exclus du rapport"),
               ("quotes_used", "Citations exactes reproduites"), ("unresolved_warnings", "Avertissements non résolus"))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def paths(metadata: dict) -> dict[str, Path]:
    base = Path(metadata["output_dir"]) / OUT_DIRNAME
    return {"md": base / REPORT_MD, "json": base / REPORT_JSON, "manifest": base / MANIFEST}


def _rel(path: Path, metadata: dict) -> str:
    return str(path.relative_to(Path(metadata["output_dir"]))).replace("\\", "/")


# --- Lecture des manifestes (jamais de recalcul) -------------------------------------------------------------

def load_upstream(metadata: dict) -> dict[str, dict | None]:
    base = Path(metadata["output_dir"])
    return {key: _read(base / rel) for key, rel in UPSTREAM.items()}


def api_calls_of(metadata: dict, docs: dict) -> dict:
    """Appels à une API externe déclarés par chaque manifeste ; un manifeste muet est « non vérifiable »."""
    sources = {}
    batch = docs.get("batch_manifest")
    if batch is not None and "api_calls" in batch:
        sources["stages_3_5"] = batch["api_calls"]
    else:
        local = [metadata.get(f"stage{s}_local") for s in ("3", "4", "5")]
        sources["stages_3_5"] = (sum(x["api_calls"] for x in local) if all(x and "api_calls" in x for x in local)
                                 else None)
    for key, label in (("stage6_corpus", "stage6"), ("stage7_theory", "stage7"), ("stage8_report_draft", "stage8"),
                       ("stage9_validation", "stage9")):
        doc = docs.get(key)
        sources[label] = doc.get("api_calls") if doc is not None else None
    sources["stage10"] = 0  # cette étape n'appelle aucun modèle, local ou distant
    unverifiable = [k for k, v in sources.items() if not isinstance(v, int)]
    return {"total": sum(v for v in sources.values() if isinstance(v, int)), "verified": not unverifiable,
            "by_source": sources, "unverifiable": unverifiable}


def models_of(metadata: dict, docs: dict) -> list[str]:
    found = [d.get("model") for d in docs.values() if d is not None]
    found += [(metadata.get(f"stage{s}_local") or {}).get("model") for s in ("3", "4", "5")]
    return sorted({m for m in found if m})


def statistics(metadata: dict, docs: dict, final_paragraphs: list[dict]) -> dict:
    batch, stage6, theory = docs.get("batch_manifest") or {}, docs.get("stage6_corpus") or {}, docs.get("stage7_theory") or {}
    draft, validation = docs.get("stage8_report_draft") or {}, docs.get("stage9_validation") or {}
    coverage = theory.get("corpus_coverage") or {}
    verdicts = [c.get("final_verdict") for c in validation.get("paragraph_checks", [])]
    planned = batch.get("total") or len(batch.get("interviews") or {}) or coverage.get("expected_interviews") \
        or metadata.get("file_count")
    included = stage6.get("interview_ids") if "interview_ids" in stage6 else coverage.get("included_interview_ids")
    excluded = stage6.get("excluded_interviews") if "excluded_interviews" in stage6 else coverage.get("excluded_interviews")
    return {"planned": planned, "ingested": len(eligible_files(metadata)),
            "analysed_stage5": len(batch["stage5_valid"]) if "stage5_valid" in batch else None,
            "included_stage6": len(included) if included is not None else None,
            "excluded": len(excluded) if excluded is not None else None,
            "theory_claims": len(theory["theory_claims"]) if "theory_claims" in theory else None,
            "stage8_paragraphs": sum(p.get("status") == "accepted" for p in draft["paragraphs"])
            if "paragraphs" in draft else None,
            "stage9_pass": verdicts.count("PASS"), "stage9_warn": verdicts.count("WARN"),
            "stage9_fail": verdicts.count("FAIL"), "stage9_excluded": len(validation.get("excluded_paragraphs", [])),
            "quotes_used": sum(len(p.get("quotes_used") or []) for p in final_paragraphs),
            "unresolved_warnings": len(validation.get("unresolved_warnings", []))}


# --- Assemblage (éditorial uniquement) ------------------------------------------------------------------------

def route(section: dict, paragraph: dict) -> str:
    kind = section.get("kind") or section.get("section_id", "").split("-")[0]
    return ROUTES.get((kind, paragraph.get("subsection"))) or DEFAULT_ROUTE.get(kind, "discussion")


def arrange(report: dict, excluded: set[str]) -> tuple[dict[str, list[dict]], list[str]]:
    """Paragraphes validés -> sections finales, dans l'ordre du plan ; textes, citations et provenance intacts."""
    placed: dict[str, list[dict]] = {sid: [] for _, sid, _ in SECTIONS}
    dropped = []
    for section in report.get("analytic_sections", []):
        for p in section.get("paragraphs", []):
            if p.get("paragraph_id") in excluded or p.get("status", "accepted") != "accepted":
                dropped.append(p.get("paragraph_id"))  # filet de sécurité : jamais un paragraphe exclu
                continue
            placed[route(section, p)].append({**p, "source_section_id": section.get("section_id")})
    return placed, dropped


def transparency_note(docs: dict, stats: dict, api: dict, models: list[str]) -> list[str]:
    validation = docs.get("stage9_validation") or {}
    semantic = (validation.get("validation_manifest") or {}).get("semantic", "non renseignée")
    model = ", ".join(models) or "non renseigné"
    api_line = (f"Appels à une API externe déclarés par les manifestes : {api['total']}."
                if api["verified"] else f"Appels à une API externe déclarés : {api['total']} — NON VÉRIFIABLE pour : "
                f"{', '.join(api['unverifiable'])} (manifeste absent ou muet).")
    return [f"Modèle de langue LOCAL ({model}), servi par Ollama sur l'ordinateur de l'analyste ; aucune API payante.",
            api_line,
            "Étapes ayant utilisé le modèle local : 3 (pratiques et signaux), 4 (épisodes d'accountability), "
            "5 (configuration intra-entretien), 6 (comparaison par blocs), 7 (théorisation), 8 (rédaction des "
            f"paragraphes), 9 (vérification sémantique légère : {semantic}).",
            "Étapes déterministes (sans modèle) : 1-2 (ingestion, structuration), validation de chaque réponse du "
            "modèle (identifiants, citations, comptes), fusion des blocs de l'étape 6, contrôles et corrections de "
            "l'étape 9, et cette livraison (étape 10, qui ne fait aucune analyse).",
            f"Entretiens réellement inclus dans l'analyse transversale : {stats['included_stage6']} sur "
            f"{stats['planned']} prévus.",
            f"Avertissements non résolus : {stats['unresolved_warnings']} (voir l'annexe) ; ils n'invalident pas le "
            "rapport mais signalent les passages à relire.",
            "Les passages entre guillemets précédés de « > » sont des citations EXACTES des enquêté·es, revérifiées sur "
            "les transcriptions ; tout le reste du texte analytique est une interprétation générée par le modèle local "
            "puis validée par TRACE (étape 9), à lire comme telle.",
            "TRACE est un outil de recherche EXPÉRIMENTAL : ce rapport soutient l'analyse du chercheur, il ne la "
            "remplace pas."]


def build(metadata: dict, docs: dict) -> dict:
    report, validation = docs["stage9_validated_report"], docs["stage9_validation"]
    excluded = {e["paragraph_id"] for e in validation.get("excluded_paragraphs", [])}
    placed, dropped = arrange(report, excluded)
    final_paragraphs = [p for _, sid, _ in SECTIONS for p in placed[sid]]
    stats = statistics(metadata, docs, final_paragraphs)
    api, models = api_calls_of(metadata, docs), models_of(metadata, docs)
    unresolved = list(validation.get("unresolved_warnings", []))
    outcome = validation.get("validation_outcome")
    status = COMPLETE if outcome == COMPLETE and not unresolved else SUCCESS_WITH_WARNINGS
    limitations = [x for x in report.get("limitations", []) if not x.startswith(OBSOLETE_LIMITATIONS)]
    verdicts = {c["paragraph_id"]: c for c in validation.get("paragraph_checks", [])}
    used_in = {}
    for p in final_paragraphs:
        for t in p.get("theory_claim_refs", []):
            used_in.setdefault(t, []).append(p["paragraph_id"])
    return {
        "stage10_version": STAGE10_VERSION, "title": TITLE, "experimental": True, "validated": True,
        "run_id": metadata.get("run_id"), "status": status, "stage9_outcome": outcome,
        "validated_at": validation.get("updated_at") or validation.get("generated_at"),
        "statistics": stats, "api_calls": api, "models": models,
        "executive_summary": list(report.get("executive_summary", [])),
        "corpus_lines": list(report.get("coverage_lines", [])),
        "excluded_interviews": (docs.get("stage6_corpus") or {}).get("excluded_interviews", []),
        "methodology": transparency_note(docs, stats, api, models), "limitations": limitations,
        "sections": [{"number": n, "section_id": sid, "title": f"{n}. {title}",
                      "paragraph_ids": [p["paragraph_id"] for p in placed[sid]]} for n, sid, title in SECTIONS],
        "paragraphs": [{"paragraph_id": p["paragraph_id"], "final_section_id": sid, "text": p["text"],
                        "source_section_id": p["source_section_id"], "subsection": p.get("subsection"),
                        "scope": p.get("scope"), "theory_claim_refs": p.get("theory_claim_refs", []),
                        "supporting_interview_ids": p.get("interview_refs", []),
                        "supporting_cross_claim_ids": p.get("supporting_cross_claim_ids", []),
                        "episode_refs": p.get("episode_refs", []),
                        "representative_episode_ids": p.get("representative_episode_ids", []),
                        "counterexamples": p.get("counterexamples", []), "quotes": p.get("quotes_used", []),
                        "stage9_verdict": p.get("stage9_verdict"), "stage9_action": p.get("stage9_action"),
                        "warnings": p.get("warnings", []) + p.get("stage9_issues", []),
                        "corrections_applied": (verdicts.get(p["paragraph_id"]) or {}).get("corrections", []),
                        "needs_review": p.get("needs_review")}
                       for _, sid, _ in SECTIONS for p in placed[sid]],
        "theory_claims": [{**{k: c.get(k) for k in ("theory_claim_id", "proposition_type", "level", "scope",
                                                     "formulation", "supporting_interview_ids",
                                                     "supporting_cross_claim_ids", "representative_episode_ids",
                                                     "counterexamples", "needs_review")},
                           "used_in_paragraphs": used_in.get(c.get("theory_claim_id"), [])}
                          for c in report.get("traceability_annex", [])],
        "excluded_paragraphs": validation.get("excluded_paragraphs", []), "dropped_paragraph_ids": dropped,
        "removed_quotes": validation.get("removed_quotes", []), "unresolved_warnings": unresolved,
        "upstream_warnings": validation.get("upstream_warnings", []),
        "provenance": {"content": "stage9/stage9_validated_report.json (paragraphes recopiés sans réécriture)",
                       "validation": "stage9/stage9_validation.json",
                       "stage9_manifest": report.get("stage9_manifest"),
                       "metadata_only": [str(UPSTREAM[k]).replace("\\", "/") for k in
                                         ("stage8_report_draft", "stage7_theory", "stage6_corpus", "batch_manifest")]}}


# --- Markdown -------------------------------------------------------------------------------------------------

def _value(v) -> str:
    return "non disponible" if v is None else str(v)


def to_markdown(final: dict) -> str:
    stats = final["statistics"]
    out = [f"# {final['title']}", "",
           f"Run `{final['run_id']}` · livraison **{final['status']}** · rapport validé par l'étape 9 "
           f"({final['stage9_outcome']}) · document expérimental produit par TRACE", "", "## Sommaire", ""]
    out += [f"- {s['title']}" for s in final["sections"]] + [""]
    out += ["## 1. Titre", "", f"**{final['title']}**", "",
            "Rapport issu d'une analyse qualitative assistée par un modèle de langue local, à lire comme un support "
            "de travail expérimental et non comme une conclusion définitive.", ""]
    out += ["## 2. Résumé exécutif", ""] + final["executive_summary"] + [""]
    out += ["## 3. Corpus analysé", "", "| Indicateur | Valeur |", "|---|---|"]
    out += [f"| {label} | {_value(stats[key])} |" for key, label in STAT_LABELS] + [""]
    out += final["corpus_lines"] + [""]
    out += ["## 4. Méthodologie TRACE", "", "**Note de transparence méthodologique**", ""]
    out += [f"- {x}" for x in final["methodology"]] + [""]
    by_id = {p["paragraph_id"]: p for p in final["paragraphs"]}
    for section in final["sections"][4:]:
        sid = section["section_id"]
        out += [f"## {section['title']}", ""]
        if sid == "limitations":
            out += [f"- {x}" for x in final["limitations"]] + [""]
            continue
        if sid == "annex":
            out += annex_markdown(final)
            continue
        if not section["paragraph_ids"]:
            out += ["*Aucun paragraphe validé pour cette section.*", ""]
            continue
        for pid in section["paragraph_ids"]:
            p = by_id[pid]
            mark = "" if p["text"].startswith(HYPOTHESIS_STARTS) else \
                INDIVIDUAL_MARK if p["scope"] == INDIVIDUAL else MIXED_MARK if p["scope"] == "mixed" else ""
            out += [f"{mark}{p['text']} *[{pid}]*"]
            out += [f"> « {q['quote']} » — {q['interview_id']}, {q['turn_id'].rsplit('_', 1)[-1]}" for q in p["quotes"]]
            out += [""]
    return "\n".join(out).rstrip() + "\n"


def annex_markdown(final: dict) -> list[str]:
    out = ["Chaque paragraphe du rapport est suivi de son identifiant (P8-…), qui renvoie à la ligne correspondante "
           "ci-dessous puis aux propositions de l'étape 7. Provenance complète : `stage10/final_report.json`.", "",
           "### Paragraphes", "", "| Paragraphe | Section | Propositions | Entretiens | Épisodes | Verdict étape 9 | "
           "Corrections |", "|---|---|---|---|---|---|---|"]
    numbers = {s["section_id"]: s["number"] for s in final["sections"]}
    for p in final["paragraphs"]:
        out.append(f"| {p['paragraph_id']} | {numbers[p['final_section_id']]} | {', '.join(p['theory_claim_refs']) or '—'} "
                   f"| {', '.join(p['supporting_interview_ids']) or '—'} | {', '.join(p['episode_refs']) or '—'} | "
                   f"{p['stage9_verdict'] or '—'} | {len(p['corrections_applied'])} |")
    out += ["", "### Propositions théoriques (étape 7)", ""]
    for c in final["theory_claims"]:
        out.append(f"- **{c['theory_claim_id']}** ({c['proposition_type']}, {c['scope']}) : {c['formulation']} — "
                   f"entretiens : {', '.join(c['supporting_interview_ids'] or []) or '—'}"
                   + (f" — paragraphes : {', '.join(c['used_in_paragraphs'])}" if c["used_in_paragraphs"] else "")
                   + (" — **à revoir**" if c["needs_review"] else ""))
    if final["excluded_paragraphs"]:
        out += ["", "### Paragraphes exclus par l'étape 9 (absents du rapport)", ""]
        out += [f"- {e['paragraph_id']}" for e in final["excluded_paragraphs"]]
    if final["unresolved_warnings"]:
        out += ["", "### Avertissements non résolus", ""] + [f"- {w}" for w in final["unresolved_warnings"]]
    return out + [""]


# --- Exécution ------------------------------------------------------------------------------------------------

def run(metadata: dict, *, log=print) -> dict:
    out = paths(metadata)
    docs = load_upstream(metadata)
    validation, report = docs["stage9_validation"], docs["stage9_validated_report"]
    base = Path(metadata["output_dir"])
    manifest = {"stage10_version": STAGE10_VERSION, "run_id": metadata.get("run_id"), "status": None, "reason": None,
                "generated_at": _now(), "pipeline_version": lp.PIPELINE_VERSION, "execution": "locale",
                "model_calls": 0,
                "upstream_files": {k: {"path": str(rel).replace("\\", "/"), "present": docs[k] is not None}
                                   for k, rel in UPSTREAM.items()},
                "stage_statuses": stage_statuses(metadata, docs)}
    reason = None
    if validation is None:
        reason = "Étape 9 absente : lancez d'abord trace_local.py stage9 <run>."
    elif validation.get("status") == BLOCKED or validation.get("validation_outcome") not in DELIVERABLE:
        reason = f"Étape 9 non livrable ({validation.get('validation_outcome') or validation.get('status')})" + \
            (f" — {validation['reason']}" if validation.get("reason") else "") + "."
    elif report is None or report.get("validated") is not True:
        reason = "Rapport validé de l'étape 9 absent ou non validé : relancez trace_local.py stage9 <run>."
    if reason:
        manifest.update(status=BLOCKED, reason=reason, output_files={"delivery_manifest": _rel(out["manifest"], metadata)})
        write_json_atomic(out["manifest"], manifest)
        log(f"Étape 10 — BLOQUÉE : {reason}")
        return manifest
    final = build(metadata, docs)
    write_json_atomic(out["json"], final)
    out["md"].write_text(to_markdown(final), encoding="utf-8")
    batch = docs.get("batch_manifest") or {}
    manifest.update(
        status=final["status"], model=", ".join(final["models"]) or None, models=final["models"],
        api_calls=final["api_calls"], interview_counts=final["statistics"],
        stages={"llm": ["3", "4", "5", "6", "7", "8", "9 (vérification sémantique)"],
                "deterministic": ["1", "2", "6 (fusion)", "9 (contrôles et corrections)", "10"]},
        output_files={"final_report_md": _rel(out["md"], metadata), "final_report_json": _rel(out["json"], metadata),
                      "delivery_manifest": _rel(out["manifest"], metadata)},
        warnings=final["upstream_warnings"], unresolved_warnings=final["unresolved_warnings"],
        exclusions={"interviews": final["excluded_interviews"], "paragraphs": final["excluded_paragraphs"]},
        validation_summary={"outcome": final["stage9_outcome"], "pass": final["statistics"]["stage9_pass"],
                            "warn": final["statistics"]["stage9_warn"], "fail": final["statistics"]["stage9_fail"],
                            "excluded": final["statistics"]["stage9_excluded"],
                            "corrected": len(validation.get("corrected_paragraphs", [])),
                            "removed_quotes": len(validation.get("removed_quotes", [])),
                            "semantic": (validation.get("validation_manifest") or {}).get("semantic")},
        durations={"batch_interview_seconds": round(sum(e.get("duration_seconds") or 0
                                                        for e in (batch.get("interviews") or {}).values()), 1)
                   if batch.get("interviews") else None,
                   "batch_started_at": batch.get("started_at"), "batch_finished_at": batch.get("finished_at"),
                   "stage_generated_at": {k: (d or {}).get("generated_at") for k, d in docs.items()
                                          if k != "batch_manifest"}})
    write_json_atomic(out["manifest"], manifest)
    log(f"Étape 10 — livraison {final['status']} : {len(final['paragraphs'])} paragraphe(s) validé(s), "
        f"{final['statistics']['quotes_used']} citation(s) exacte(s)")
    return manifest


def stage_statuses(metadata: dict, docs: dict) -> dict:
    batch = docs.get("batch_manifest") or {}
    return {"1-2": metadata.get("status"), "3-5": batch.get("state") or batch.get("counts"),
            "6": (docs.get("stage6_corpus") or {}).get("status"), "7": (docs.get("stage7_theory") or {}).get("status"),
            "8": (docs.get("stage8_report_draft") or {}).get("status"),
            "9": (docs.get("stage9_validation") or {}).get("validation_outcome")}
