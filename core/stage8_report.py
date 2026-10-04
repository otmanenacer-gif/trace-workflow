"""Étape 8 — Rédaction analytique du rapport corpus (BROUILLON que l'étape 9 vérifiera ; jamais présenté comme validé).

    python scripts/trace_local.py stage8 <run_id>

    <run>/stage7/stage7_theory.json (COMPLETE) + <run>/stage6/stage6_corpus.json — jamais les transcriptions ;
    étape 4 seulement pour retrouver les citations EXACTES des épisodes représentatifs déjà identifiés
      → plan DÉTERMINISTE : chaque proposition TH… de l'étape 7 attribuée aux sections selon son type et son niveau
      → sections FACTUELLES écrites par TRACE, sans appel : titre et statut expérimental, résumé exécutif, corpus et
        couverture, méthodologie, limites, annexe de traçabilité
      → sections ANALYTIQUES (résultats ; variations, tensions et cas négatifs ; discussion ; conclusion) : UN appel
        du Report Section Writer chacune (au plus MAX_CLAIMS propositions par appel ; section sans proposition : aucun
        appel), réponse mise en cache et section enregistrée dès sa validation ; une section terminée n'est jamais
        rejouée, une section en échec seule
      → validation de chaque paragraphe (P8-001…) : propositions TH… citées connues (sinon paragraphe rejeté, jamais
        dans le texte) ; entretiens, affirmations de l'étape 6, épisodes et contre-exemples RECONSTITUÉS par TRACE à
        partir des propositions citées ; citations verbatim insérées par TRACE depuis l'étape 4 (vérifiées mot pour mot
        sur la transcription), une citation introuvable supprimée avec un avertissement ; tout passage entre guillemets
        écrit par le modèle et absent des citations exactes est retiré ; un paragraphe qui ne repose que sur des
        hypothèses individuelles est étiqueté comme tel, jamais comme une régularité
      → <run>/stage8/stage8_report_draft.json (provenance complète) et stage8_report_draft.md (lisible, chaque
        paragraphe suivi de sa provenance abrégée).

Pas d'appel final de cohérence : les sections reposent sur des ensembles de propositions déjà fusionnés par l'étape 7
(aucune répétition à corriger par un modèle) ; ordre et transitions sont fixés par le plan. Aucun recalcul des
étapes 1 à 7, aucun Grounding Checker, aucun benchmark ; modèle local via Ollama, 0 API.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from agents import report_writer as rw
from core import config
from core import local_pipeline as lp
from core.analysis import eligible_files
from core.analysis_cache import AnalysisCache, write_json_atomic
from core.final_report import STAGE6_NOT_APPLICABLE_REASON
from core.llm_client import LLMError
from core.schemas import SPEAKER_INTERVIEWER
from core.stage7_theory import INDIVIDUAL, _call, _read

STAGE8_VERSION = "1.0"
OUT_DIRNAME = "stage8"
JSON_FILENAME = "stage8_report_draft.json"
MD_FILENAME = "stage8_report_draft.md"
STAGE7_PATH = Path("stage7") / "stage7_theory.json"
STAGE6_PATH = Path("stage6") / "stage6_corpus.json"
MAX_CLAIMS = 25
QUOTE_MAX_CHARS = 400
MAX_QUOTES_PER_PARAGRAPH = 2
TITLE = "Rapport analytique du corpus — BROUILLON expérimental (étape 8), à vérifier (étape 9)"
REMOVED_QUOTE = "[citation retirée par TRACE]"
GENERALIZING = re.compile(r"\b(?:les étudiant|la plupart|majorit|généralement|en général|tendance|tous les|toutes les|"
                          r"souvent)", re.IGNORECASE)
QUOTED = re.compile(r"«\s*([^»]+?)\s*»|“([^”]+)”|\"([^\"]+)\"")
SECTIONS = (
    {"id": "results", "title": "5. Résultats principaux",
     "subsections": ("categories", "regularities", "logics", "boundaries", "criteria"),
     "types": ("analytic_category", "recurring_mechanism", "usage_logic", "normative_boundary", "good_work_criterion"),
     "include_structuring": True},
    {"id": "variations", "title": "6. Variations et tensions",
     "subsections": ("variations", "contradictions", "trajectories", "negative_cases"),
     "types": ("configuration_variation", "tension", "negative_case", "usage_logic")},
    {"id": "discussion", "title": "7. Discussion analytique",
     "subsections": ("relations", "mechanisms", "propositions", "scope"),
     "types": ("theoretical_proposition", "recurring_mechanism"), "include_related": True},
    {"id": "conclusion", "title": "9. Conclusion", "subsections": ("conclusion",), "types": (),
     "include_structuring": True, "include_hypotheses": True},
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def paths(metadata: dict) -> tuple[Path, Path]:
    directory = Path(metadata["output_dir"]) / OUT_DIRNAME
    return directory / JSON_FILENAME, directory / MD_FILENAME


# --- Plan (déterministe) ---------------------------------------------------------------------------------

def plan(theory: dict) -> list[dict]:
    """Propositions de chaque section analytique (une proposition peut servir plusieurs sections)."""
    claims = theory.get("theory_claims", [])
    related = {r["source"] for r in theory.get("relations", [])} | {r["target"] for r in theory.get("relations", [])}
    sections = []
    for spec in SECTIONS:
        chosen = [c for c in claims if c["proposition_type"] in spec["types"]
                  or (spec.get("include_structuring") and c.get("level") == "structuring")
                  or (spec.get("include_related") and c["theory_claim_id"] in related)
                  or (spec.get("include_hypotheses") and c.get("scope") == INDIVIDUAL)]
        ids = list(dict.fromkeys(c["theory_claim_id"] for c in chosen))
        parts = [ids[i:i + MAX_CLAIMS] for i in range(0, len(ids), MAX_CLAIMS)] or [[]]
        for n, part in enumerate(parts, start=1):
            suffix = f"-{n}" if len(parts) > 1 else ""
            sections.append({"section_id": f"{spec['id']}{suffix}", "kind": spec["id"], "title": spec["title"],
                             "subsections": spec["subsections"], "theory_claim_ids": part})
    return sections


# --- Citations exactes (étape 4) ------------------------------------------------------------------------

class Quotes:
    """Citation verbatim d'un épisode : première citation validée d'un tour de l'enquêté·e (ni enquêteur, ni locuteur
    douteux), revérifiée mot pour mot sur la transcription. Introuvable : None (jamais reconstruite)."""

    def __init__(self, metadata: dict):
        self.dirs = {f["ingestion"]["interview_id"]: Path(f["ingestion"]["output_dir"]) for f in eligible_files(metadata)}
        self._cache: dict[str, dict] = {}

    def _interview(self, iid: str) -> dict:
        if iid not in self._cache:
            base = self.dirs[iid]
            episodes = (_read(base / config.ANALYSIS_SUBDIR / config.ACCOUNTABILITY_EPISODES_FILENAME) or {}).get(
                "episodes", [])
            turns = {t["turn_id"]: t for t in (_read(base / config.STRUCTURED_TRANSCRIPT_FILENAME) or {"turns": []})
                     ["turns"]}
            self._cache[iid] = {"episodes": {e["episode_id"]: e for e in episodes}, "turns": turns}
        return self._cache[iid]

    def get(self, episode_id: str) -> dict | None:
        iid = next((i for i in self.dirs if episode_id.startswith(f"{i}_")), None)
        if iid is None:
            return None
        data = self._interview(iid)
        episode = data["episodes"].get(episode_id)
        if episode is None:
            return None
        doubtful = {w["turn_id"] for w in episode.get("speaker_warnings", [])}
        for e in episode.get("evidence", []):
            turn = data["turns"].get(e.get("turn_id"))
            if (e.get("validation") or {}).get("valid") and turn and turn["speaker"] != SPEAKER_INTERVIEWER \
                    and e["turn_id"] not in doubtful and e.get("quote") and e["quote"] in turn["text"]:
                quote = e["quote"]
                return {"episode_id": episode_id, "interview_id": iid, "turn_id": e["turn_id"],
                        "quote": quote if len(quote) <= QUOTE_MAX_CHARS else quote[:QUOTE_MAX_CHARS].rstrip() + " […]",
                        "truncated": len(quote) > QUOTE_MAX_CHARS}
        return None


# --- Une section analytique ------------------------------------------------------------------------------

def section_message(section: dict, claims: dict[str, dict], theory: dict, coverage: dict) -> tuple[str, dict]:
    episodes: dict[str, str] = {}
    lines = []
    for tid in section["theory_claim_ids"]:
        c = claims[tid]
        refs = []
        for eid in c.get("representative_episode_ids", []):
            local = next((k for k, v in episodes.items() if v == eid), None) or f"EP{len(episodes) + 1}"
            episodes[local] = eid
            refs.append(local)
        lines.append(json.dumps({"id": tid, "type": c["proposition_type"], "level": c.get("level"),
                                 "scope": c.get("scope"), "category": c.get("category"),
                                 "formulation": c.get("formulation"), "n_interviews": len(c["supporting_interview_ids"]),
                                 "counterexamples": [x.get("description") for x in c.get("counterexamples", [])][:4],
                                 "needs_review": c.get("needs_review"), "episodes": refs},
                                ensure_ascii=False, sort_keys=True))
    relations_block = ""
    if section["kind"] == "discussion":
        wanted = set(section["theory_claim_ids"])
        rel = [r for r in theory.get("relations", []) if r["source"] in wanted and r["target"] in wanted]
        if rel:
            relations_block = "\n<relations>\n" + "\n".join(json.dumps(
                {"source": r["source"], "target": r["target"], "type": r["relation_type"],
                 "description": r.get("description")}, ensure_ascii=False, sort_keys=True) for r in rel) + "\n</relations>\n"
    message = rw.SECTION_TEMPLATE.format(
        n_included=coverage.get("n_included"), n_expected=coverage.get("expected_interviews"), title=section["title"],
        subsections=", ".join(section["subsections"]), claims_json="\n".join(lines), relations_block=relations_block)
    return message, episodes


def _strip_invented_quotes(text: str, verbatim: list[str]) -> tuple[str, list[str]]:
    """Retire tout passage cité (≥ 4 mots) écrit par le modèle et absent des citations exactes."""
    warnings = []

    def replace(match):
        quoted = next(g for g in match.groups() if g is not None)
        if len(quoted.split()) < 4 or any(quoted.strip() in v for v in verbatim):
            return match.group(0)
        # le passage n'est jamais reproduit, même dans l'avertissement (il pourrait passer pour une citation)
        warnings.append(f"passage entre guillemets ({len(quoted.split())} mots) absent des citations exactes : retiré")
        return REMOVED_QUOTE
    return QUOTED.sub(replace, text), warnings


def validate_paragraphs(output: dict, section: dict, claims: dict[str, dict], episodes: dict[str, str],
                        quotes: Quotes) -> list[dict]:
    allowed = set(section["theory_claim_ids"])
    rows = []
    for draft in output.get("paragraphs", []):
        warnings = []
        refs = [t for t in draft.get("theory_claim_refs", []) if t in allowed]
        unknown = [t for t in draft.get("theory_claim_refs", []) if t not in allowed]
        if unknown:
            warnings.append(f"proposition(s) inconnue(s) de cette section ignorée(s) : {', '.join(unknown)}")
        row = {"section_id": section["section_id"], "subsection": draft.get("subsection")
               if draft.get("subsection") in section["subsections"] else "other", "warnings": warnings}
        if not refs:
            rows.append({**row, "status": "rejected_no_evidence", "text": draft.get("text"), "theory_claim_refs": [],
                         "needs_review": True})
            continue
        used = [claims[t] for t in refs]
        interviews = sorted({i for c in used for i in c["supporting_interview_ids"]})
        cross = list(dict.fromkeys(x for c in used for x in c.get("supporting_cross_claim_ids", [])))
        representative = list(dict.fromkeys(e for c in used for e in c.get("representative_episode_ids", [])))
        requested = []
        for ref in draft.get("evidence_refs", []):
            if ref in allowed:
                continue  # une proposition citée en preuve : déjà dans theory_claim_refs
            eid = episodes.get(ref)
            if eid is None:
                warnings.append(f"épisode demandé inconnu : {ref} — aucune citation ajoutée")
            elif eid not in representative:
                warnings.append(f"épisode {ref} hors des propositions citées — aucune citation ajoutée")
            else:
                requested.append(eid)
        quotes_used = []
        for eid in requested[:MAX_QUOTES_PER_PARAGRAPH]:
            found = quotes.get(eid)
            if found is None:
                warnings.append(f"citation introuvable pour l'épisode {eid} : supprimée, jamais reconstruite")
            else:
                quotes_used.append(found)
        text, quote_warnings = _strip_invented_quotes(draft.get("text") or "", [q["quote"] for q in quotes_used])
        warnings += quote_warnings
        scopes = {c.get("scope") for c in used}
        scope = INDIVIDUAL if scopes == {INDIVIDUAL} else "mixed" if INDIVIDUAL in scopes else "corpus"
        if scope == INDIVIDUAL and GENERALIZING.search(text):
            warnings.append("formulation générale sur une hypothèse individuelle : présentée comme hypothèse")
        counter = [x for c in used for x in c.get("counterexamples", [])]
        rows.append({**row, "status": "accepted", "text": text, "theory_claim_refs": refs,
                     "interview_refs": interviews, "supporting_cross_claim_ids": cross,
                     "episode_refs": [q["episode_id"] for q in quotes_used], "representative_episode_ids": representative,
                     "counterexamples": counter, "quotes_used": quotes_used, "scope": scope,
                     "needs_review": scope != "corpus" or any(c.get("needs_review") for c in used) or bool(warnings)})
    return rows


# --- Sections factuelles (déterministes) ------------------------------------------------------------------

def coverage_section(coverage: dict, stage6: dict, theory: dict) -> list[str]:
    excluded = coverage.get("excluded_interviews") or []
    lines = [f"- Entretiens prévus : **{coverage.get('expected_interviews')}** ; analysés (étapes 6 et 7) : "
             f"**{coverage.get('n_included')}**.",
             f"- Entretiens inclus : {', '.join(coverage.get('included_interview_ids') or []) or '—'}."]
    lines.append("- Entretiens exclus : " + ("; ".join(f"{e['interview_id']} ({e['reason']})" for e in excluded)
                                             if excluded else "aucun") + ".")
    blocks = stage6.get("blocks", [])
    lines.append(f"- Étape 6 : {stage6.get('status')}, {len(blocks)} bloc(s) de comparaison ; étape 7 : "
                 f"{theory.get('status')}, {len(theory.get('theory_claims', []))} proposition(s) théorique(s).")
    return lines


METHODOLOGY = [
    "Chaîne TRACE : ingestion et structuration des entretiens (étapes 1-2, sans IA) ; pratiques et signaux "
    "interactionnels (étape 3) ; épisodes d'accountability (étape 4) ; configuration intra-entretien (étape 5) ; "
    "comparaison inter-entretiens par blocs (étape 6) ; théorisation transversale (étape 7) ; ce brouillon (étape 8).",
    "Le modèle local (Ollama, aucune API) a produit les interprétations ; chaque étape a été validée de façon "
    "déterministe (identifiants, citations, comptes) avant d'alimenter la suivante.",
    "Les citations reproduites ici sont des citations EXACTES de l'enquêté·e, recopiées par TRACE depuis les épisodes "
    "et revérifiées sur la transcription ; les paragraphes sont des interprétations générées, à vérifier.",
    "Chaque paragraphe analytique porte un identifiant (P8-…) et sa provenance : propositions de l'étape 7, "
    "entretiens, affirmations de l'étape 6, épisodes et contre-exemples.",
]


def limitations_of(theory: dict, stage6: dict, warnings: list[str], failed: list[str]) -> list[str]:
    limits = ["Brouillon expérimental, non validé : l'étape 9 doit le vérifier avant toute diffusion.",
              "Une proposition appuyée par un seul entretien est une hypothèse, jamais une tendance du corpus.",
              "Les comptes portent sur des entretiens, jamais des occurrences ; un élément non mentionné est non observé."]
    limits += [x for x in theory.get("limitations", []) if x not in limits]
    limits += [x for x in stage6.get("limitations", []) if x not in limits]
    if failed:
        limits.append("Section(s) non générée(s) : " + ", ".join(failed) + " — relancez l'étape 8.")
    if warnings:
        limits.append(f"{len(warnings)} avertissement(s) de rédaction : voir warnings.")
    return limits


# --- Exécution ---------------------------------------------------------------------------------------------

def run(metadata: dict, *, settings=None, runner=None, cache: AnalysisCache | None = None, log=print) -> dict:
    settings = settings or lp.runtime_settings()
    cache = cache or AnalysisCache()
    json_path, md_path = paths(metadata)
    draft = {"stage8_version": STAGE8_VERSION, "title": TITLE, "experimental": True, "validated": False,
             "run_id": metadata.get("run_id"), "generated_at": _now(), "model": settings.model, "execution": "locale",
             "api_calls": 0, "status": None, "reason": None}

    def save() -> None:
        draft["updated_at"] = _now()
        write_json_atomic(json_path, draft)
        md_path.write_text(to_markdown(draft), encoding="utf-8")

    theory = _read(Path(metadata["output_dir"]) / STAGE7_PATH)
    stage6 = _read(Path(metadata["output_dir"]) / STAGE6_PATH) or {}
    if theory is None or theory.get("status") in (lp.STAGE_BLOCKED, None):
        draft.update(status=lp.STAGE_BLOCKED, reason="Étape 7 absente ou bloquée : lancez d'abord trace_local.py stage7 <run>.")
        save()
        return draft
    if theory["status"] == lp.STAGE_NOT_APPLICABLE:
        draft.update(status=lp.STAGE_NOT_APPLICABLE, reason=STAGE6_NOT_APPLICABLE_REASON)
        save()
        return draft
    if theory["status"] != lp.STAGE_COMPLETE:
        draft.update(status=lp.STAGE_BLOCKED, reason=f"Étape 7 non terminée ({theory['status']}"
                     + (f" — {theory['reason']}" if theory.get("reason") else "") + ") : relancez trace_local.py stage7.")
        save()
        return draft
    coverage = theory.get("corpus_coverage") or {}
    claims = {c["theory_claim_id"]: c for c in theory.get("theory_claims", [])}
    upstream = list(theory.get("needs_review", []))
    draft.update(corpus_coverage=coverage, upstream_warnings=upstream, stage7_status=theory["status"],
                 stage6_status=stage6.get("status"))
    sections = plan(theory)
    draft["analytic_sections"] = [{**s, "status": "PENDING", "paragraphs": []} for s in sections]
    manifest = {"plan": "déterministe (types et niveaux des propositions de l'étape 7)", "coherence_pass": "non exécuté "
                "(sections fondées sur des propositions déjà fusionnées par l'étape 7)", "calls": []}
    draft["generation_manifest"] = manifest
    save()
    quotes = Quotes(metadata)
    if runner is None:
        runner = lp.make_runner(settings, journal_dir=Path(metadata["output_dir"]) / lp.LOCAL_RUNS_SUBDIR / "stage8")
    try:
        for section in draft["analytic_sections"]:
            if not section["theory_claim_ids"]:
                section.update(status="EMPTY", note="Aucune proposition de l'étape 7 pour cette section : aucun appel.")
                save()
                continue
            message, episodes = section_message(section, claims, theory, coverage)
            log(f"Étape 8 — section {section['section_id']} ({len(section['theory_claim_ids'])} proposition(s))…")
            record = _call(rw.SPEC, message, f"{metadata['run_id']}/{rw.SPEC.name}/{section['section_id']}",
                           metadata["run_id"], runner, cache, settings)
            manifest["calls"].append({"section_id": section["section_id"], "status": record["status"],
                                      "cache_hit": record["cache_hit"], "cache_key": record["cache_key"],
                                      "error": record["error"]})
            if record["output"] is None:
                section.update(status="FAILED", error=record["error"])
                log(f"Étape 8 — section {section['section_id']} : ÉCHEC")
            else:
                paragraphs = validate_paragraphs(record["output"], section, claims, episodes, quotes)
                warned = any(p["warnings"] for p in paragraphs)
                section.update(status="SUCCESS_WITH_WARNINGS" if warned else "SUCCESS", paragraphs=paragraphs,
                               cache_hit=record["cache_hit"])
                log(f"Étape 8 — section {section['section_id']} : {section['status']} ({len(paragraphs)} paragraphe(s))")
            save()
    except LLMError as error:
        draft.update(status=lp.STAGE_FAILED, reason=f"runtime local indisponible ({error.code}) : relancez pour reprendre")
        save()
        raise
    finalize(draft, claims, theory, stage6)
    save()
    return draft


def finalize(draft: dict, claims: dict[str, dict], theory: dict, stage6: dict) -> None:
    """Identifiants stables des paragraphes, index de provenance, sections factuelles, statut."""
    paragraphs, number = [], 0
    for section in draft["analytic_sections"]:
        for p in section.get("paragraphs", []):
            if p["status"] == "accepted":
                number += 1
                p["paragraph_id"] = f"P8-{number:03d}"
            paragraphs.append(p)
    accepted = [p for p in paragraphs if p["status"] == "accepted"]
    warnings = [f"{p.get('paragraph_id', p['section_id'])} : {w}" for p in paragraphs for w in p["warnings"]]
    warnings += [f"paragraphe rejeté ({p['section_id']}) : aucune proposition de l'étape 7 identifiable"
                 for p in paragraphs if p["status"] != "accepted"]
    failed = [s["section_id"] for s in draft["analytic_sections"] if s["status"] == "FAILED"]
    structuring = [claims[t] for t in theory.get("structuring_regularities", []) if t in claims]
    hypotheses = [c for c in claims.values() if c.get("scope") == INDIVIDUAL]
    coverage = draft["corpus_coverage"]
    executive = [f"Corpus de {coverage.get('n_included')} entretien(s) analysé(s) sur {coverage.get('expected_interviews')} "
                 f"prévu(s). L'étape 7 a établi {len(claims)} proposition(s) théorique(s), dont {len(structuring)} "
                 f"régularité(s) structurante(s) et {len(hypotheses)} hypothèse(s) reposant sur un seul entretien."]
    executive += [f"- {c['formulation']} ({c['theory_claim_id']}, {len(c['supporting_interview_ids'])} entretien(s))"
                  + (" — à revoir" if c.get("needs_review") else "") for c in structuring]
    if theory.get("synthesis_summary"):
        executive.append(f"Synthèse de l'étape 7 (modèle local, à relire) : {theory['synthesis_summary']}")
    annex = [{"theory_claim_id": c["theory_claim_id"], "proposition_type": c["proposition_type"], "level": c.get("level"),
              "scope": c.get("scope"), "formulation": c.get("formulation"),
              "supporting_interview_ids": c["supporting_interview_ids"],
              "supporting_cross_claim_ids": c.get("supporting_cross_claim_ids", []),
              "representative_episode_ids": c.get("representative_episode_ids", []),
              "counterexamples": c.get("counterexamples", []), "needs_review": c.get("needs_review"),
              "used_in_paragraphs": [p["paragraph_id"] for p in accepted if c["theory_claim_id"] in p["theory_claim_refs"]]}
             for c in claims.values()]
    draft.update(
        executive_summary=executive,
        coverage_lines=coverage_section(coverage, stage6, theory),
        methodology=METHODOLOGY,
        paragraphs=[{k: p.get(k) for k in ("paragraph_id", "section_id", "subsection", "status", "scope",
                                           "theory_claim_refs", "interview_refs", "supporting_cross_claim_ids",
                                           "episode_refs", "needs_review")} for p in paragraphs],
        theory_claim_refs=sorted({t for p in accepted for t in p["theory_claim_refs"]}),
        interview_refs=sorted({i for p in accepted for i in p["interview_refs"]}),
        evidence_refs=sorted({e for p in accepted for e in p["episode_refs"]} |
                             {x for p in accepted for x in p["supporting_cross_claim_ids"]}),
        quotes_used=[{"paragraph_id": p["paragraph_id"], **q} for p in accepted for q in p["quotes_used"]],
        needs_review=[p["paragraph_id"] for p in accepted if p["needs_review"]] + draft.get("upstream_warnings", []),
        warnings=warnings, limitations=limitations_of(theory, stage6, warnings, failed), traceability_annex=annex,
        sections=[{"section_id": s, "title": t} for s, t in (
            ("title", "1. Titre et statut"), ("executive_summary", "2. Résumé exécutif"),
            ("coverage", "3. Corpus et couverture"), ("methodology", "4. Méthodologie TRACE"))]
        + [{"section_id": s["section_id"], "title": s["title"], "status": s["status"]} for s in draft["analytic_sections"]]
        + [{"section_id": "limitations", "title": "8. Limites"}, {"section_id": "annex", "title": "10. Annexe de traçabilité"}],
        status=lp.STAGE_FAILED if failed else lp.STAGE_COMPLETE,
        result_status="SUCCESS_WITH_WARNINGS" if warnings or draft.get("upstream_warnings") else "SUCCESS",
        reason=("section(s) en échec : " + ", ".join(failed)) if failed else None)


# --- Markdown ------------------------------------------------------------------------------------------------

def _provenance(p: dict) -> str:
    parts = [p.get("paragraph_id") or "P8-… (en cours)", ", ".join(p["theory_claim_refs"]),
             f"{len(p['interview_refs'])} entretien(s)"]
    if p.get("counterexamples"):
        parts.append(f"{len(p['counterexamples'])} contre-exemple(s)")
    if p["needs_review"]:
        parts.append("à revoir")
    return "*[" + " · ".join(parts) + "]*"


def to_markdown(draft: dict) -> str:
    out = [f"# {draft['title']}", "", f"Run `{draft['run_id']}` · {draft['generated_at']} · statut **{draft['status']}**"
           + (f" ({draft.get('result_status')})" if draft.get("result_status") else "")
           + " · **non validé** — brouillon à vérifier par l'étape 9", ""]
    if draft.get("reason"):
        out += [f"> {draft['reason']}", ""]
    if "executive_summary" in draft:
        out += ["## 2. Résumé exécutif", ""] + draft["executive_summary"] + [""]
        out += ["## 3. Corpus et couverture", ""] + draft["coverage_lines"] + [""]
        out += ["## 4. Méthodologie TRACE", ""] + [f"- {x}" for x in draft["methodology"]] + [""]
    titles_done = set()
    for section in draft.get("analytic_sections", []):
        if section["title"] not in titles_done:
            out += [f"## {section['title']}", ""]
            titles_done.add(section["title"])
        if section["status"] == "FAILED":
            out += ["*Section non générée (échec) : relancez l'étape 8.*", ""]
            continue
        if section["status"] == "EMPTY":
            out += [f"*{section.get('note')}*", ""]
            continue
        for p in section.get("paragraphs", []):
            if p["status"] != "accepted":
                continue
            prefix = "*(Hypothèse — un seul entretien)* " if p["scope"] == INDIVIDUAL else \
                "*(Hypothèses individuelles et régularités mêlées)* " if p["scope"] == "mixed" else ""
            out += [prefix + p["text"]]
            out += [f"> « {q['quote']} » — {q['interview_id']}, {q['turn_id'].rsplit('_', 1)[-1]}" for q in p["quotes_used"]]
            out += [_provenance(p), ""]
    if "limitations" in draft:
        out += ["## 8. Limites", ""] + [f"- {x}" for x in draft["limitations"]] + [""]
        out += ["## 10. Annexe de traçabilité", ""]
        for c in draft["traceability_annex"]:
            out.append(f"- **{c['theory_claim_id']}** ({c['proposition_type']}, {c['level']}, {c['scope']}) : "
                       f"{c['formulation']} — entretiens : {', '.join(c['supporting_interview_ids']) or '—'} — "
                       f"épisodes : {', '.join(e.rsplit('_', 1)[-1] for e in c['representative_episode_ids']) or '—'}"
                       + (" — **à revoir**" if c["needs_review"] else "")
                       + (f" — paragraphes : {', '.join(c['used_in_paragraphs'])}" if c["used_in_paragraphs"] else ""))
        if draft.get("warnings"):
            out += ["", "### Avertissements de rédaction", ""] + [f"- {w}" for w in draft["warnings"]]
    return "\n".join(out) + "\n"


def summary_lines(draft: dict) -> list[str]:
    lines = [f"Run {draft['run_id']} — étape 8 (brouillon du rapport corpus) : {draft['status']}"
             + (f" ({draft.get('result_status')})" if draft.get("result_status") else "")
             + (f" — {draft['reason']}" if draft.get("reason") else "")]
    for section in draft.get("analytic_sections", []):
        lines.append(f"  {section['section_id']} {section['status']} : "
                     f"{sum(p['status'] == 'accepted' for p in section.get('paragraphs', []))} paragraphe(s)")
    if "paragraphs" in draft:
        lines.append(f"  paragraphes : {sum(p['status'] == 'accepted' for p in draft['paragraphs'])} ; citations "
                     f"insérées : {len(draft['quotes_used'])} ; avertissements : {len(draft['warnings'])}")
    return lines
