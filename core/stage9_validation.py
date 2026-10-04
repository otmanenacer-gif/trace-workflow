"""Étape 9 — Validation du brouillon du rapport corpus (étape 8), paragraphe par paragraphe : PASS / WARN / FAIL.

    python scripts/trace_local.py stage9 <run_id>

    <run>/stage8/stage8_report_draft.json + stage7_theory.json + stage6_corpus.json ; étape 4 et transcriptions
    seulement pour revérifier les citations déjà insérées — aucun recalcul des étapes 1 à 8
      A. contrôles DÉTERMINISTES (aucun appel) de chaque paragraphe P8-… :
         identifiants (TH…, entretiens, affirmations de l'étape 6, épisodes) ; citations (épisode, entretien, tour,
         locuteur autorisé, texte exact de la transcription) ; généralisation (portée et nombre d'entretiens des
         propositions de l'étape 7, contre-exemples connus) ; provenance (propositions et entretiens) ; avertissements
         en amont (propagés, jamais FAIL à eux seuls)
      B. vérification sémantique LÉGÈRE : Report Validator, un appel par groupe d'au plus GROUP_SIZE paragraphes
         (texte, propositions citées, portée, entretiens, contre-exemples, avertissements, citations validées — jamais
         les transcriptions) ; cache, enregistrement après chaque groupe ; modèle indisponible → validation
         déterministe seule, SUCCESS_WITH_WARNINGS
      C. correction sans boucle : PASS gardé ; WARN gardé et signalé ; FAIL corrigé de façon déterministe quand c'est
         possible (citation invalide retirée, marqueur d'hypothèse individuelle, périmètre en entretiens, prudence,
         contre-exemple déjà connu), sinon paragraphe EXCLU de la version validée — jamais de régénération
      → stage9_validation.json, stage9_validated_report.json (validated: true) et stage9_validated_report.md.

Statut global : COMPLETE (aucun avertissement), SUCCESS_WITH_WARNINGS (livrable), BLOCKED (étape 8 absente,
incompatible, aucune section analytique valide ou provenance largement absente). Quelques FAIL ne bloquent pas.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path

from agents import report_validator as rv
from core import config
from core import local_pipeline as lp
from core import stage8_report as s8
from core.analysis import eligible_files
from core.analysis_cache import AnalysisCache, write_json_atomic
from core.llm_client import LLMError
from core.schemas import SPEAKER_INTERVIEWER
from core.stage7_theory import INDIVIDUAL, _call, _read

STAGE9_VERSION = "1.0"
OUT_DIRNAME = "stage9"
VALIDATION_FILENAME = "stage9_validation.json"
REPORT_JSON = "stage9_validated_report.json"
REPORT_MD = "stage9_validated_report.md"
GROUP_SIZE = 8
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
RANK = {PASS: 0, WARN: 1, FAIL: 2}
COMPLETE, SUCCESS_WITH_WARNINGS, BLOCKED = "COMPLETE", "SUCCESS_WITH_WARNINGS", "BLOCKED"
HYPOTHESIS_MARKERS = re.compile(r"hypoth|un seul entretien|dans un entretien|dans cet entretien|à confirmer|cas individuel",
                                re.IGNORECASE)
CORPUS_WIDE = re.compile(r"\b(?:tous les étudiant|toutes les étudiant|l'ensemble des étudiant|les étudiants en général|"
                         r"en général|généralement|la plupart des étudiant)", re.IGNORECASE)
COUNTER_MARKERS = re.compile(r"contre-exemple|cas contraire|à l'inverse|exception|sauf|cependant|toutefois|nuanc|"
                             r"en revanche|complique", re.IGNORECASE)
HYPOTHESIS_PREFIX = "Hypothèse issue d'un seul entretien, à confirmer : "
CAUTION_PREFIX = "Avec prudence (à vérifier) : "
MIN_WORDS_AFTER_CORRECTION = 8


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def paths(metadata: dict) -> dict[str, Path]:
    directory = Path(metadata["output_dir"]) / OUT_DIRNAME
    return {"validation": directory / VALIDATION_FILENAME, "json": directory / REPORT_JSON, "md": directory / REPORT_MD}


# --- Citations (revérifiées sur l'étape 4 et la transcription) ---------------------------------------------

class QuoteChecker:
    def __init__(self, metadata: dict):
        self.dirs = {f["ingestion"]["interview_id"]: Path(f["ingestion"]["output_dir"]) for f in eligible_files(metadata)}
        self._data: dict[str, dict] = {}

    def _interview(self, iid: str) -> dict:
        if iid not in self._data:
            base = self.dirs[iid]
            episodes = (_read(base / config.ANALYSIS_SUBDIR / config.ACCOUNTABILITY_EPISODES_FILENAME) or {}).get(
                "episodes", [])
            turns = {t["turn_id"]: t for t in (_read(base / config.STRUCTURED_TRANSCRIPT_FILENAME) or {"turns": []})
                     ["turns"]}
            self._data[iid] = {"episodes": {e["episode_id"]: e for e in episodes}, "turns": turns}
        return self._data[iid]

    def problem(self, quote: dict, episode_refs: list[str]) -> str | None:
        """Raison d'invalidité d'une citation insérée, ou None si elle est exacte et autorisée."""
        iid, eid, tid = quote.get("interview_id"), quote.get("episode_id"), quote.get("turn_id")
        text = (quote.get("quote") or "").removesuffix(" […]").strip()
        if iid not in self.dirs:
            return f"entretien inconnu ({iid})"
        if not eid or not eid.startswith(f"{iid}_") or not (tid or "").startswith(f"{iid}_"):
            return "épisode ou tour d'un autre entretien"
        if eid not in episode_refs:
            return f"épisode {eid} absent des références du paragraphe"
        data = self._interview(iid)
        episode, turn = data["episodes"].get(eid), data["turns"].get(tid)
        if episode is None:
            return f"épisode {eid} introuvable à l'étape 4"
        if turn is None:
            return f"tour {tid} introuvable dans la transcription"
        if turn["speaker"] == SPEAKER_INTERVIEWER or tid in {w["turn_id"] for w in episode.get("speaker_warnings", [])}:
            return "locuteur non autorisé (enquêteur ou attribution douteuse)"
        if not text or text not in turn["text"]:
            return "texte absent de la transcription (citation modifiée ou reconstruite)"
        if not any(e.get("turn_id") == tid and (e.get("validation") or {}).get("valid") and text in (e.get("quote") or "")
                   for e in episode.get("evidence", [])):
            return "citation absente des preuves validées de l'épisode"
        return None


# --- A. Contrôles déterministes d'un paragraphe ------------------------------------------------------------

def check_paragraph(p: dict, claims: dict[str, dict], known: dict, quotes: QuoteChecker, n_included: int) -> dict:
    """Verdict déterministe et paragraphe corrigé (ou exclu). Aucun appel."""
    issues, corrections, removed = [], [], []
    verdict = PASS
    text = p.get("text") or ""

    def flag(level: str, message: str) -> None:
        nonlocal verdict
        issues.append({"level": level, "message": message})
        verdict = max(verdict, level, key=RANK.get)

    # 1. identifiants
    refs = [t for t in p.get("theory_claim_refs", []) if t in claims]
    for t in p.get("theory_claim_refs", []):
        if t not in claims:
            flag(WARN, f"proposition inventée ou inconnue de l'étape 7 : {t} — retirée")
    for key, label, valid in (("interview_refs", "entretien", known["interviews"]),
                              ("supporting_cross_claim_ids", "affirmation de l'étape 6", known["cross_claims"]),
                              ("episode_refs", "épisode", known["episodes"])):
        for value in p.get(key) or []:
            if value not in valid:
                flag(WARN, f"{label} inconnu : {value} — retiré")
    # 4. provenance (recalculée sur les seules propositions existantes)
    used = [claims[t] for t in refs]
    interviews = sorted({i for c in used for i in c["supporting_interview_ids"]})
    if not refs or not interviews:
        flag(FAIL, "provenance absente : aucune proposition de l'étape 7 ou aucun entretien identifiable")
        return {"verdict": FAIL, "action": "excluded", "issues": issues, "corrections": [], "removed_quotes": [],
                "paragraph": None}
    representative = list(dict.fromkeys(e for c in used for e in c.get("representative_episode_ids", [])))
    episode_refs = [e for e in (p.get("episode_refs") or []) if e in representative]
    # 2. citations
    kept_quotes = []
    for q in p.get("quotes_used") or []:
        reason = quotes.problem(q, episode_refs)
        if reason is None:
            kept_quotes.append(q)
            continue
        removed.append({"paragraph_id": p["paragraph_id"], **q, "reason": reason})
        quoted = (q.get("quote") or "").removesuffix(" […]").strip()
        if quoted and quoted in text:  # l'argument cite le passage invalide dans le texte même
            text = text.replace(f"« {quoted} »", s8.REMOVED_QUOTE).replace(quoted, s8.REMOVED_QUOTE)
            flag(FAIL, f"citation invalide intégrée au texte ({reason}) — retirée du texte")
            corrections.append("citation invalide retirée du texte")
        else:
            flag(WARN, f"citation invalide retirée ({reason})")
    verbatim = [q["quote"].removesuffix(" […]") for q in kept_quotes]
    for match in s8.QUOTED.finditer(text):
        passage = next(g for g in match.groups() if g is not None)
        if len(passage.split()) >= 4 and not any(passage.strip() in v for v in verbatim):
            text = text.replace(match.group(0), s8.REMOVED_QUOTE)
            flag(WARN, "passage entre guillemets absent des citations exactes — retiré")
            corrections.append("passage entre guillemets non vérifié retiré")
    if (text != (p.get("text") or "")
            and len(re.sub(re.escape(s8.REMOVED_QUOTE), "", text).split()) < MIN_WORDS_AFTER_CORRECTION):
        flag(FAIL, "le paragraphe reposait entièrement sur des citations invalides")
        return {"verdict": FAIL, "action": "excluded", "issues": issues, "corrections": corrections,
                "removed_quotes": removed, "paragraph": None}
    # 3. généralisation
    scopes = {c.get("scope") for c in used}
    individual = scopes == {INDIVIDUAL}
    if individual and (not HYPOTHESIS_MARKERS.search(text) or s8.GENERALIZING.search(text)):
        flag(FAIL, "hypothèse individuelle formulée comme une généralisation — marquée hypothèse")
        text = HYPOTHESIS_PREFIX + text
        corrections.append("marqueur d'hypothèse individuelle ajouté")
    elif INDIVIDUAL in scopes:
        flag(WARN, "paragraphe mêlant hypothèses individuelles et régularités du corpus")
    max_support = max(len(c["supporting_interview_ids"]) for c in used)
    if not individual and CORPUS_WIDE.search(text) and max_support < n_included:
        flag(WARN, f"généralisation au-delà du périmètre ({max_support} entretien(s) sur {n_included})")
        text += f" (Périmètre : {max_support} entretien(s) sur {n_included}.)"
        corrections.append("périmètre en entretiens ajouté")
    counter = [x for c in used for x in c.get("counterexamples", []) if x.get("description")]
    if counter and not COUNTER_MARKERS.search(text):
        first = counter[0]
        flag(WARN, "contre-exemple connu absent du texte — ajouté")
        text += f" Contre-exemple connu : {first['description']}" + (
            f" ({first['interview_id']})." if first.get("interview_id") else ".")
        corrections.append("contre-exemple déjà connu ajouté")
    # 5. avertissements en amont : jamais FAIL à eux seuls
    upstream = list(p.get("warnings") or [])
    if p.get("needs_review") or upstream or any(c.get("needs_review") for c in used):
        flag(WARN, "avertissement en amont (needs_review)" + (f" : {'; '.join(upstream)}" if upstream else ""))
    corrected = {**p, "text": text, "theory_claim_refs": refs, "interview_refs": interviews,
                 "supporting_cross_claim_ids": [x for x in dict.fromkeys(
                     x for c in used for x in c.get("supporting_cross_claim_ids", [])) if x in known["cross_claims"]],
                 "episode_refs": [q["episode_id"] for q in kept_quotes], "quotes_used": kept_quotes,
                 "counterexamples": counter}
    action = "corrected" if corrections else "kept"
    return {"verdict": verdict, "action": action, "issues": issues, "corrections": corrections,
            "removed_quotes": removed, "paragraph": corrected}


# --- B. Vérification sémantique légère ---------------------------------------------------------------------

def group_message(group_id: str, paragraphs: list[dict], claims: dict[str, dict], n_included: int) -> str:
    lines = []
    for p in paragraphs:
        lines.append(json.dumps({
            "paragraph_id": p["paragraph_id"], "text": p["text"],
            "claims": [{"id": t, "formulation": claims[t].get("formulation"), "scope": claims[t].get("scope"),
                        "n_interviews": len(claims[t]["supporting_interview_ids"]),
                        "counterexamples": [x.get("description") for x in claims[t].get("counterexamples", [])][:3]}
                       for t in p["theory_claim_refs"]],
            "warnings": (p.get("warnings") or [])[:3], "validated_quotes": [q["quote"] for q in p["quotes_used"]]},
            ensure_ascii=False, sort_keys=True))
    return rv.VALIDATION_TEMPLATE.format(n_included=n_included, group_id=group_id, count=len(paragraphs),
                                         paragraphs_json="\n".join(lines))


def apply_semantic(check: dict, semantic: dict | None) -> dict:
    """Combine le verdict sémantique au verdict déterministe ; FAIL sémantique : correction déterministe si possible
    (prudence, hypothèse individuelle), sinon exclusion. Jamais de réécriture par le modèle."""
    if check["paragraph"] is None:
        return check
    if semantic is None:
        check["issues"].append({"level": WARN, "message": "vérification sémantique non effectuée"})
        check["verdict"] = max(check["verdict"], WARN, key=RANK.get)
        return check
    level = semantic["verdict"]
    if level == PASS:
        return check
    message = f"sémantique ({semantic['issue_type']}) : {semantic['explanation']}"
    if level == WARN:
        check["issues"].append({"level": WARN, "message": message})
        check["verdict"] = max(check["verdict"], WARN, key=RANK.get)
        return check
    paragraph = check["paragraph"]
    if semantic["issue_type"] == "overgeneralization" or semantic["suggested_action"] in ("add_caution",
                                                                                          "mark_individual_case"):
        individual = paragraph.get("scope") == INDIVIDUAL
        prefix = HYPOTHESIS_PREFIX if individual or semantic["suggested_action"] == "mark_individual_case" else CAUTION_PREFIX
        if not paragraph["text"].startswith(prefix):
            paragraph["text"] = prefix + paragraph["text"]
        check["issues"].append({"level": FAIL, "message": message + " — corrigé (marqueur de prudence)"})
        check["corrections"].append("marqueur de prudence ajouté (vérification sémantique)")
        check["verdict"], check["action"] = FAIL, "corrected"
        return check
    check["issues"].append({"level": FAIL, "message": message + " — paragraphe exclu (aucune correction sûre)"})
    check.update(verdict=FAIL, action="excluded", paragraph=None)
    return check


# --- Exécution ---------------------------------------------------------------------------------------------

def _upstream(stage6: dict, theory: dict, draft: dict) -> list[str]:
    warnings = [f"étape 6 : {w}" for w in stage6.get("claims_needing_review", [])]
    warnings += [f"étape 6 : entretien exclu — {e['interview_id']} ({e['reason']})" for e in stage6.get("excluded_interviews", [])]
    warnings += [f"étape 7 : {w}" for w in theory.get("needs_review", [])]
    warnings += [f"étape 8 : {w}" for w in draft.get("warnings", [])]
    return list(dict.fromkeys(warnings))


def run(metadata: dict, *, settings=None, runner=None, cache: AnalysisCache | None = None, log=print) -> dict:
    settings = settings or lp.runtime_settings()
    cache = cache or AnalysisCache()
    out = paths(metadata)
    validation = {"stage9_version": STAGE9_VERSION, "run_id": metadata.get("run_id"), "generated_at": _now(),
                  "model": settings.model, "execution": "locale", "api_calls": 0}

    def save_validation() -> None:
        validation["updated_at"] = _now()
        write_json_atomic(out["validation"], validation)

    def blocked(reason: str) -> dict:
        validation.update(status=BLOCKED, result_status=BLOCKED, validation_outcome=BLOCKED, reason=reason)
        save_validation()
        return validation

    base = Path(metadata["output_dir"])
    draft = _read(base / s8.OUT_DIRNAME / s8.JSON_FILENAME)
    theory = _read(base / "stage7" / "stage7_theory.json")
    stage6 = _read(base / "stage6" / "stage6_corpus.json") or {}
    if draft is None or "analytic_sections" not in draft:
        return blocked("Étape 8 absente ou incompatible : lancez d'abord trace_local.py stage8 <run>.")
    if theory is None or not theory.get("theory_claims"):
        return blocked("Étape 7 absente ou sans proposition : la provenance du brouillon ne peut pas être vérifiée.")
    claims = {c["theory_claim_id"]: c for c in theory["theory_claims"]}
    coverage = theory.get("corpus_coverage") or {}
    n_included = coverage.get("n_included") or len(coverage.get("included_interview_ids") or [])
    evidence = theory.get("evidence_index") or {}
    known = {"interviews": set(coverage.get("included_interview_ids") or []),
             "cross_claims": set(evidence.get("cross_claims") or {}) | {x for c in claims.values()
                                                                       for x in c.get("supporting_cross_claim_ids", [])},
             "episodes": set(evidence.get("episodes") or {}) | {e for c in claims.values()
                                                               for e in c.get("representative_episode_ids", [])}}
    paragraphs = [p for s in draft["analytic_sections"] for p in s.get("paragraphs", []) if p.get("status") == "accepted"]
    if not paragraphs:
        return blocked("Aucun paragraphe analytique dans le brouillon de l'étape 8.")
    quotes = QuoteChecker(metadata)
    checks = {p["paragraph_id"]: check_paragraph(p, claims, known, quotes, n_included) for p in paragraphs}
    no_provenance = sum(c["paragraph"] is None and any("provenance absente" in i["message"] for i in c["issues"])
                        for c in checks.values())
    validation.update(upstream_warnings=_upstream(stage6, theory, draft), deterministic_checks={
        pid: {"verdict": c["verdict"], "issues": c["issues"]} for pid, c in checks.items()})
    if no_provenance > len(paragraphs) / 2:
        return blocked(f"Provenance largement absente ({no_provenance} paragraphe(s) sur {len(paragraphs)}).")
    candidates = [c["paragraph"] for c in checks.values() if c["paragraph"] is not None]
    groups = [candidates[i:i + GROUP_SIZE] for i in range(0, len(candidates), GROUP_SIZE)]
    validation["semantic_groups"] = [{"group_id": f"G{n:02d}", "paragraph_ids": [p["paragraph_id"] for p in g],
                                      "status": "PENDING"} for n, g in enumerate(groups, start=1)]
    save_validation()
    semantic: dict[str, dict] = {}
    unavailable = False
    if runner is None:
        runner = lp.make_runner(settings, journal_dir=base / lp.LOCAL_RUNS_SUBDIR / "stage9")
    for group, members in zip(validation["semantic_groups"], groups):
        if unavailable:
            group["status"] = "SKIPPED_MODEL_UNAVAILABLE"
            continue
        log(f"Étape 9 — vérification sémantique {group['group_id']} ({len(members)} paragraphe(s))…")
        try:
            record = _call(rv.SPEC, group_message(group["group_id"], members, claims, n_included),
                           f"{metadata['run_id']}/{rv.SPEC.name}/{group['group_id']}", metadata["run_id"], runner,
                           cache, settings)
        except LLMError as error:  # modèle indisponible : validation déterministe seule
            unavailable = True
            group.update(status="SKIPPED_MODEL_UNAVAILABLE", error=error.code)
            log(f"Étape 9 — modèle local indisponible ({error.code}) : validation déterministe seule")
            save_validation()
            continue
        if record["output"] is None:
            group.update(status="FAILED", error=record["error"])
        else:
            wanted = set(group["paragraph_ids"])
            for v in record["output"].get("verdicts", []):
                if v.get("paragraph_id") in wanted:
                    semantic[v["paragraph_id"]] = v
            group.update(status="DONE", cache_hit=record["cache_hit"])
        validation["semantic_checks"] = semantic
        save_validation()
    for pid, check in checks.items():
        if check["paragraph"] is not None:
            apply_semantic(check, semantic.get(pid))
    finalize(validation, draft, checks, semantic, unavailable, out, metadata)
    return validation


def finalize(validation: dict, draft: dict, checks: dict, semantic: dict, unavailable: bool, out: dict,
             metadata: dict) -> None:
    kept = {pid for pid, c in checks.items() if c["paragraph"] is not None}
    passed = [pid for pid, c in checks.items() if c["paragraph"] is not None and c["verdict"] == PASS]
    warned = [pid for pid, c in checks.items() if c["paragraph"] is not None and c["verdict"] != PASS]
    excluded = [pid for pid in checks if pid not in kept]
    unresolved = [f"{pid} : {i['message']}" for pid, c in checks.items() if c["paragraph"] is not None
                  for i in c["issues"] if i["level"] != PASS]
    report = copy.deepcopy(draft)
    for section in report["analytic_sections"]:
        section["paragraphs"] = [{**checks[p["paragraph_id"]]["paragraph"],
                                  "stage9_verdict": checks[p["paragraph_id"]]["verdict"],
                                  "stage9_action": checks[p["paragraph_id"]]["action"],
                                  "stage9_issues": [i["message"] for i in checks[p["paragraph_id"]]["issues"]]}
                                 for p in section.get("paragraphs", [])
                                 if p.get("status") == "accepted" and p.get("paragraph_id") in kept]
    valid_sections = [s for s in report["analytic_sections"] if s["paragraphs"]]
    for entry in report.get("traceability_annex", []):
        entry["used_in_paragraphs"] = [pid for pid in entry.get("used_in_paragraphs", []) if pid in kept]
    report["paragraphs"] = [p for p in report.get("paragraphs", []) if p.get("paragraph_id") in kept]
    report["quotes_used"] = [q for q in report.get("quotes_used", []) if q.get("paragraph_id") in kept
                             and not any(r["paragraph_id"] == q["paragraph_id"] and r.get("episode_id") == q.get("episode_id")
                                         for c in checks.values() for r in c["removed_quotes"])]
    limitations = list(draft.get("limitations", []))
    if unavailable:
        limitations.append("Vérification sémantique non effectuée (modèle local indisponible) : contrôles déterministes seuls.")
    if excluded:
        limitations.append(f"{len(excluded)} paragraphe(s) exclu(s) par la validation (incompatibles avec les preuves, sans "
                           "correction sûre) : " + ", ".join(excluded) + ".")
    blocked = not valid_sections
    outcome = BLOCKED if blocked else (SUCCESS_WITH_WARNINGS if warned or excluded or validation.get("upstream_warnings")
                                       or unavailable else COMPLETE)
    validation.update(
        status=BLOCKED if blocked else lp.STAGE_COMPLETE, result_status=outcome, validation_outcome=outcome,
        reason="aucune section analytique valide après validation" if blocked else None,
        paragraph_checks=[{"paragraph_id": pid, "deterministic": {"verdict": validation["deterministic_checks"][pid]["verdict"]},
                           "semantic": semantic.get(pid), "final_verdict": c["verdict"], "action": c["action"],
                           "issues": c["issues"], "corrections": c["corrections"]} for pid, c in checks.items()],
        semantic_checks=semantic, passed_paragraph_ids=passed, warned_paragraph_ids=warned,
        failed_paragraph_ids=[pid for pid, c in checks.items() if c["verdict"] == FAIL],
        removed_quotes=[r for c in checks.values() for r in c["removed_quotes"]],
        corrected_paragraphs=[{"paragraph_id": pid, "corrections": c["corrections"]} for pid, c in checks.items()
                              if c["action"] == "corrected"],
        excluded_paragraphs=[{"paragraph_id": pid, "issues": [i["message"] for i in checks[pid]["issues"]]}
                             for pid in excluded],
        unresolved_warnings=unresolved, limitations=limitations,
        validation_manifest={"deterministic": "identifiants, citations, généralisation, provenance, avertissements",
                             "semantic": "indisponible" if unavailable else f"{len(validation.get('semantic_groups', []))} "
                             f"groupe(s) d'au plus {GROUP_SIZE} paragraphes", "corrections": "déterministes uniquement",
                             "regeneration": "aucune"})
    if validation.get("semantic_checks") is None:
        validation["semantic_checks"] = {}
    report.update(validated=not blocked, experimental=True, stage9_status=outcome,
                  title="Rapport analytique du corpus — version validée (étape 9), expérimentale",
                  stage9_manifest=str(out["validation"].relative_to(Path(metadata["output_dir"]))),
                  limitations=limitations, result_status=outcome,
                  stage9_summary={"passed": len(passed), "warned": len(warned), "excluded": len(excluded)})
    write_json_atomic(out["validation"], validation)
    write_json_atomic(out["json"], report)
    markdown = s8.to_markdown(report).replace(
        " · **non validé** — brouillon à vérifier par l'étape 9",
        f" · **validé par l'étape 9** ({outcome}) : {len(passed)} paragraphe(s) PASS, {len(warned)} WARN, "
        f"{len(excluded)} exclu(s)")
    out["md"].write_text(markdown, encoding="utf-8")


def summary_lines(validation: dict) -> list[str]:
    lines = [f"Run {validation['run_id']} — étape 9 (validation du rapport) : {validation.get('validation_outcome')}"
             + (f" — {validation['reason']}" if validation.get("reason") else "")]
    if "passed_paragraph_ids" in validation:
        lines.append(f"  PASS {len(validation['passed_paragraph_ids'])} · WARN {len(validation['warned_paragraph_ids'])}"
                     f" · FAIL {len(validation['failed_paragraph_ids'])} (corrigés "
                     f"{len(validation['corrected_paragraphs'])}, exclus {len(validation['excluded_paragraphs'])})"
                     f" · citations retirées {len(validation['removed_quotes'])}")
        lines.append(f"  vérification sémantique : {validation['validation_manifest']['semantic']}")
    return lines
