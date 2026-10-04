"""Étape 7 — Théorisation transversale du corpus (structure théorique exploitable par l'étape 8, pas encore un rapport).

    python scripts/trace_local.py stage7 <run_id>

    <run>/stage6/stage6_corpus.json (COMPLETE) — jamais les transcriptions ; étapes 5 et 4 seulement pour retrouver
    les épisodes représentatifs des affirmations de trajectoire citées
      A. blocs THÉMATIQUES suivant les catégories de l'étape 6 (régularités + configurations ; variations +
         trajectoires ; tensions ; cas négatifs + exceptions ; critères + frontières), au plus MAX_ITEMS éléments par
         appel ; un bloc vide ne fait aucun appel. Theory Block Analyst : propositions théoriques qui citent les
         identifiants des éléments reçus (X…, K…, R…)
      B. Theory Synthesizer : UN appel sur les seules propositions validées des blocs (P…) — fusion des doublons,
         relations entre catégories, hiérarchie (structuring / secondary / hypothesis)
    Traçabilité DÉTERMINISTE (le modèle ne recopie aucun identifiant de preuve) : chaque proposition hérite des
    entretiens, des affirmations de l'étape 6 (`<bloc étape 6>:<cross_claim_id>`), des affirmations de l'étape 5 et
    des épisodes représentatifs des éléments cités ; sans élément identifiable : rejetée. Appuyée par moins de deux
    entretiens : `individual_case_hypothesis` (niveau `hypothesis`, à revoir), jamais une régularité du corpus.
    Robustesse : chaque appel passe par le cache TRACE (clé : le message exact) et chaque bloc est enregistré dès sa
    validation dans <run>/stage7/stage7_theory.json ; une relance ne refait que les appels manquants ; une proposition
    que la synthèse n'a pas reprise est conservée (« non fusionnée », à revoir).

Aucun recalcul des étapes 1 à 6, aucun Grounding Checker, aucun benchmark ; modèle local via Ollama, 0 API.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

from agents import theory_builder as tb
from agents.base import sha256_text
from core import config
from core import local_pipeline as lp
from core.analysis import _run_coroutine, eligible_files
from core.analysis_cache import AnalysisCache, compute_cache_key, write_json_atomic
from core.final_report import STAGE6_NOT_APPLICABLE_REASON
from core.llm_client import RUNTIME_ERROR_CODES, LLMError
from core.schemas import SPEAKER_INTERVIEWER

STAGE7_VERSION = "1.0"
OUT_DIRNAME = "stage7"
OUT_FILENAME = "stage7_theory.json"
STAGE6_PATH = Path("stage6") / "stage6_corpus.json"
BATCH_MANIFEST = Path("batch") / "batch_manifest.json"
MAX_ITEMS = 20
MAX_EPISODES = 3
QUOTE_MAX_CHARS = 300
CORPUS, INDIVIDUAL, NEGATIVE = "corpus_regularity", "individual_case_hypothesis", "negative_case"
CRITERION_BOUNDARY_TYPES = ("recurring_student_role_criterion", "divergent_student_role_criterion",
                            "recurring_boundary", "divergent_boundary")
THEMES = (
    ("T1", "régularités et configurations récurrentes", ("regularities",), "recurring_configurations"),
    ("T2", "variations et différences de trajectoires", ("variations", "trajectory_differences"), None),
    ("T3", "tensions et contradictions", ("tensions",), None),
    ("T4", "cas négatifs et exceptions", ("negative_cases_and_exceptions",), None),
    ("T5", "critères du bon travail étudiant et frontières", None, "recurring_student_role_criteria"),
)
SECTION_TYPES = {
    "mechanisms": ("recurring_mechanism",), "variations": ("configuration_variation",),
    "usage_logics": ("usage_logic",), "tensions": ("tension",), "negative_cases": ("negative_case",),
    "criteria_and_boundaries": ("good_work_criterion", "normative_boundary"),
    "analytic_categories": ("analytic_category",), "theoretical_propositions": ("theoretical_proposition",),
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def out_path(metadata: dict) -> Path:
    return Path(metadata["output_dir"]) / OUT_DIRNAME / OUT_FILENAME


# --- Éléments (étape 6 fusionnée) et blocs thématiques -------------------------------------------------------

def evidence_items(stage6: dict) -> dict[str, dict]:
    """Toutes les affirmations validées de l'étape 6 et les répartitions déterministes, par clé de preuve stable."""
    items = {}
    for category in ("regularities", "variations", "tensions", "negative_cases_and_exceptions",
                     "trajectory_differences"):
        for claim in stage6.get(category, []):
            key = f"{claim['block_id']}:{claim['cross_claim_id']}"
            items[key] = {"key": key, "kind": "cross_claim", "category": category, **claim}
    for kind, field in (("configuration", "recurring_configurations"), ("criterion", "recurring_student_role_criteria")):
        for entry in stage6.get(field, []):
            key = f"{kind}:{entry['value']}"
            items[key] = {"key": key, "kind": kind, "claim_type": f"recurring_{kind}", "description": entry["value"],
                          "interview_ids": entry["interview_ids"], "n_supporting_interviews": entry["n_interviews"],
                          "trajectory_claim_ids": [], "counterexamples": [], "needs_review": False}
    return items


def thematic_blocks(stage6: dict, items: dict[str, dict]) -> list[dict]:
    """Blocs thématiques (catégories de l'étape 6), découpés au-delà de MAX_ITEMS ; les blocs vides sont omis."""
    blocks = []
    for theme_id, theme, categories, distribution in THEMES:
        keys = []
        if categories:
            keys += [k for k, i in items.items() if i["kind"] == "cross_claim" and i["category"] in categories]
        else:
            keys += [k for k, i in items.items() if i["kind"] == "cross_claim" and i["claim_type"] in CRITERION_BOUNDARY_TYPES]
        if distribution:
            kind = "configuration" if distribution == "recurring_configurations" else "criterion"
            keys += [k for k, i in items.items() if i["kind"] == kind]
        for part in range(0, len(keys), MAX_ITEMS):
            chunk = keys[part:part + MAX_ITEMS]
            suffix = f"-{part // MAX_ITEMS + 1}" if len(keys) > MAX_ITEMS else ""
            blocks.append({"block_id": f"{theme_id}{suffix}", "theme": theme, "item_keys": chunk})
    return blocks


def _local_ids(keys: list[str], items: dict[str, dict]) -> dict[str, str]:
    counters = {"cross_claim": 0, "configuration": 0, "criterion": 0}
    prefix = {"cross_claim": "X", "configuration": "K", "criterion": "R"}
    local = {}
    for key in keys:
        kind = items[key]["kind"]
        counters[kind] += 1
        local[f"{prefix[kind]}{counters[kind]:02d}" if kind == "cross_claim" else f"{prefix[kind]}{counters[kind]}"] = key
    return local


def block_message(block: dict, items: dict[str, dict], coverage: dict) -> tuple[str, dict[str, str]]:
    local = _local_ids(block["item_keys"], items)
    shown = []
    for lid, key in local.items():
        item = items[key]
        entry = {"id": lid, "type": item.get("claim_type"), "description": item.get("description"),
                 "n_interviews": item.get("n_supporting_interviews") or len(item.get("interview_ids", []))}
        if item.get("criterion_label"):
            entry["criterion"] = item["criterion_label"]
        if item.get("counterexamples"):
            entry["counterexamples"] = [c.get("description") for c in item["counterexamples"]]
        if item.get("needs_review"):
            entry["needs_review"] = True
        shown.append(entry)
    message = tb.BLOCK_TEMPLATE.format(
        n_included=coverage["n_included"], n_expected=coverage["expected_interviews"], theme=block["theme"],
        block_id=block["block_id"], item_count=len(local),
        items_json="\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in shown))
    return message, local


# --- Preuves : entretiens, étape 5, épisodes (déterministe) ------------------------------------------------

class EvidenceIndex:
    """Index des preuves du corpus : affirmations de l'étape 6, de l'étape 5 et épisodes de l'étape 4 cités."""

    def __init__(self, metadata: dict, items: dict[str, dict]):
        self.items = items
        self.dirs = {f["ingestion"]["interview_id"]: Path(f["ingestion"]["output_dir"]) for f in eligible_files(metadata)}
        self._trajectory: dict[str, dict] = {}
        self._episodes: dict[str, dict] = {}
        self.trajectory_claims: dict[str, dict] = {}
        self.episodes: dict[str, dict] = {}

    def _interview_of(self, object_id: str) -> str | None:
        return next((iid for iid in self.dirs if object_id.startswith(f"{iid}_")), None)

    def _stage5(self, iid: str) -> dict:
        if iid not in self._trajectory:
            path = self.dirs[iid] / config.ANALYSIS_SUBDIR / config.STUDENT_TRAJECTORY_FILENAME
            self._trajectory[iid] = _read(path) or {}
        return self._trajectory[iid]

    def _stage4(self, iid: str) -> dict:
        if iid not in self._episodes:
            analysis_dir = self.dirs[iid] / config.ANALYSIS_SUBDIR
            episodes = (_read(analysis_dir / config.ACCOUNTABILITY_EPISODES_FILENAME) or {}).get("episodes", [])
            transcript = _read(self.dirs[iid] / config.STRUCTURED_TRANSCRIPT_FILENAME) or {"turns": []}
            speaker = {t["turn_id"]: t["speaker"] for t in transcript["turns"]}
            self._episodes[iid] = {"episodes": {e["episode_id"]: e for e in episodes}, "speaker": speaker}
        return self._episodes[iid]

    def resolve(self, keys: list[str]) -> dict:
        """Preuves d'un ensemble d'éléments : entretiens, affirmations des étapes 6 et 5, épisodes, contre-exemples."""
        interviews, cross, trajectory, counter = set(), [], [], []
        for key in keys:
            item = self.items.get(key)
            if item is None:
                continue
            interviews.update(i for i in item.get("interview_ids", []) if i)
            if item["kind"] == "cross_claim":
                cross.append(key)
            trajectory += item.get("trajectory_claim_ids", [])
            counter += [{"interview_id": c.get("interview_id"), "description": c.get("description"),
                         "relation": c.get("relation"), "source": key} for c in item.get("counterexamples", [])]
        trajectory = list(dict.fromkeys(trajectory))
        episodes = []
        for tcid in trajectory:
            iid = self._interview_of(tcid)
            if iid is None:
                continue
            claim = next((c for c in self._stage5(iid).get("trajectory_claims", []) if c.get("claim_id") == tcid), None)
            if claim is None:
                continue
            self.trajectory_claims[tcid] = {"interview_id": iid, "claim_type": claim.get("claim_type"),
                                            "description": claim.get("description"),
                                            "episode_ids": claim.get("episode_ids", []),
                                            "needs_review": bool(claim.get("needs_review"))}
            for eid in claim.get("episode_ids", []):
                if eid not in episodes:
                    episodes.append(eid)
        for eid in episodes[:MAX_EPISODES]:
            self._index_episode(eid)
        return {"supporting_interview_ids": sorted(interviews), "supporting_cross_claim_ids": cross,
                "supporting_trajectory_claim_ids": trajectory, "representative_episode_ids": episodes[:MAX_EPISODES],
                "counterexamples": counter}

    def _index_episode(self, eid: str) -> None:
        if eid in self.episodes:
            return
        iid = self._interview_of(eid)
        if iid is None:
            return
        stage4 = self._stage4(iid)
        episode = stage4["episodes"].get(eid)
        if episode is None:
            return
        quote = next((e["quote"] for e in episode.get("evidence", []) if (e.get("validation") or {}).get("valid")
                      and stage4["speaker"].get(e["turn_id"]) != SPEAKER_INTERVIEWER), None)
        self.episodes[eid] = {"interview_id": iid, "episode_status": episode.get("episode_status"),
                              "summary": episode.get("episode_summary"),
                              "quote": (quote[:QUOTE_MAX_CHARS] + " […]") if quote and len(quote) > QUOTE_MAX_CHARS else quote,
                              "validation_status": episode.get("validation_status")}


def scope_of(proposition_type: str, interview_ids: list[str]) -> str:
    if proposition_type == "negative_case":
        return NEGATIVE
    return CORPUS if len(interview_ids) >= 2 else INDIVIDUAL


# --- Validation déterministe -------------------------------------------------------------------------------

def validate_block(output: dict, local: dict[str, str], index: EvidenceIndex, block_id: str) -> dict:
    accepted, rejected, issues = [], [], []
    for n, p in enumerate(output.get("propositions", []), start=1):
        unknown = [i for i in p.get("supporting_item_ids", []) + p.get("counterexample_item_ids", []) if i not in local]
        support = [local[i] for i in p.get("supporting_item_ids", []) if i in local]
        if unknown:
            issues.append(f"{block_id} proposition {n} : identifiant(s) inconnu(s) ignoré(s) — {', '.join(unknown)}")
        if not support:
            rejected.append({**p, "block_id": block_id, "reason": "NO_EVIDENCE : aucun élément reçu identifiable"})
            continue
        evidence = index.resolve(support)
        counter_keys = [local[i] for i in p.get("counterexample_item_ids", []) if i in local]
        counter = index.resolve(counter_keys)
        scope = scope_of(p["proposition_type"], evidence["supporting_interview_ids"])
        accepted.append({
            "block_id": block_id, "label": p.get("label"), "proposition_type": p["proposition_type"],
            "formulation": p.get("formulation"), "supporting_item_keys": support, **evidence,
            "counterexamples": evidence["counterexamples"] + [
                {"interview_id": iid, "description": f"élément {key} cité comme contre-exemple", "source": key}
                for key in counter_keys for iid in index.items[key].get("interview_ids", [])],
            "counterexample_interview_ids": counter["supporting_interview_ids"],
            "scope": scope, "confidence": p.get("confidence"),
            "needs_review": bool(p.get("needs_review")) or scope == INDIVIDUAL or bool(unknown),
            "limits": p.get("limits")})
    status = "SUCCESS_WITH_WARNINGS" if issues or rejected else "SUCCESS"
    return {"propositions": accepted, "rejected": rejected, "issues": issues, "status": status}


def synthesis_message(propositions: dict[str, dict], coverage: dict) -> str:
    shown = [{"id": pid, "type": p["proposition_type"], "category": p.get("label"), "formulation": p.get("formulation"),
              "n_interviews": len(p["supporting_interview_ids"]), "scope": p["scope"],
              "counterexamples": [c.get("description") for c in p["counterexamples"]][:5],
              **({"limits": p["limits"]} if p.get("limits") else {})} for pid, p in propositions.items()]
    return tb.SYNTHESIS_TEMPLATE.format(
        n_included=coverage["n_included"], n_expected=coverage["expected_interviews"],
        propositions_json="\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in shown))


def _merge_evidence(parts: list[dict]) -> dict:
    def union(key):
        return list(dict.fromkeys(v for p in parts for v in p.get(key, [])))
    counter, seen = [], set()
    for p in parts:
        for c in p.get("counterexamples", []):
            marker = (c.get("interview_id"), c.get("description"), c.get("source"))
            if marker not in seen:
                seen.add(marker)
                counter.append(c)
    return {"supporting_interview_ids": sorted(set(union("supporting_interview_ids"))),
            "supporting_cross_claim_ids": union("supporting_cross_claim_ids"),
            "supporting_trajectory_claim_ids": union("supporting_trajectory_claim_ids"),
            "representative_episode_ids": union("representative_episode_ids")[:MAX_EPISODES], "counterexamples": counter}


def validate_synthesis(output: dict | None, propositions: dict[str, dict]) -> dict:
    """Propositions fusionnées : preuves héritées des propositions des blocs ; aucune proposition perdue."""
    claims, issues, used = [], [], set()
    for n, draft in enumerate((output or {}).get("theory_claims", []), start=1):
        known = [pid for pid in draft.get("merged_from", []) if pid in propositions]
        unknown = [pid for pid in draft.get("merged_from", []) if pid not in propositions]
        if unknown:
            issues.append(f"synthèse {n} : proposition(s) inconnue(s) ignorée(s) — {', '.join(unknown)}")
        if not known:
            issues.append(f"synthèse {n} : aucune proposition reçue identifiable — rejetée")
            claims.append(None)
            continue
        used.update(known)
        evidence = _merge_evidence([propositions[pid] for pid in known])
        claims.append({"origin": "synthesis", "category": draft.get("category"),
                       "proposition_type": draft["proposition_type"], "formulation": draft.get("formulation"),
                       "merged_from": known, "level": draft.get("level"), "confidence": draft.get("confidence"),
                       "needs_review": bool(draft.get("needs_review")) or bool(unknown), "limits": draft.get("limits"),
                       **evidence})
    for pid, p in propositions.items():  # jamais une proposition perdue
        if pid not in used:
            claims.append({"origin": "not_merged_by_synthesis", "category": p.get("label"),
                           "proposition_type": p["proposition_type"], "formulation": p.get("formulation"),
                           "merged_from": [pid], "level": "secondary", "confidence": p.get("confidence"),
                           "needs_review": True, "limits": p.get("limits"),
                           **_merge_evidence([p])})
    numbering, final = {}, []
    for n, claim in enumerate(claims, start=1):
        if claim is None:
            continue
        claim["theory_claim_id"] = f"TH{len(final) + 1:03d}"
        claim["scope"] = scope_of(claim["proposition_type"], claim["supporting_interview_ids"])
        if claim["scope"] == INDIVIDUAL and claim["level"] != "hypothesis":
            issues.append(f"{claim['theory_claim_id']} : appuyée par moins de deux entretiens — niveau ramené à hypothesis")
            claim["level"] = "hypothesis"
        claim["needs_review"] = claim["needs_review"] or claim["scope"] == INDIVIDUAL
        if claim["origin"] == "synthesis":
            numbering[n] = claim["theory_claim_id"]
        final.append({"theory_claim_id": claim.pop("theory_claim_id"), **claim})
    relations = []
    for r in (output or {}).get("relations", []):
        source, target = numbering.get(r.get("source")), numbering.get(r.get("target"))
        if not source or not target or source == target:
            issues.append(f"relation {r.get('source')} → {r.get('target')} : proposition inconnue — ignorée")
            continue
        relations.append({"source": source, "target": target, "relation_type": r.get("relation_type"),
                          "description": r.get("description")})
    return {"theory_claims": final, "relations": relations, "issues": issues}


# --- Appels (cache, reprise) -------------------------------------------------------------------------------

def cache_key_fields(spec, run_id: str, settings, message: str) -> dict:
    return {"source_sha256": sha256_text(f"stage7:{run_id}"), "transcript_sha256": sha256_text(message),
            **spec.identity(), "model": settings.model, "request_params": settings.request_params()}


def _call(spec, message: str, label: str, run_id: str, runner, cache: AnalysisCache, settings) -> dict:
    """UN appel (ou le cache). Lève LLMError si le runtime local est injoignable (reprise à la relance)."""
    key_fields = cache_key_fields(spec, run_id, settings, message)
    record = {"cache_key": compute_cache_key(key_fields), "cache_hit": False, "status": None, "output": None,
              "error": None, "request_id": None}
    entry = cache.load(key_fields, spec.output_model)
    if entry is not None:
        record.update(cache_hit=True, status="CACHED", output=entry["output"])
        return record
    try:
        result = _run_coroutine(runner.complete_json(system_prompt=spec.system_prompt, user_content=message,
                                                     output_schema=spec.output_schema, response_model=spec.output_model,
                                                     label=label))
    except LLMError as error:
        if error.code in RUNTIME_ERROR_CODES and error.code != "PROMPT_TOO_LONG":
            raise
        record.update(status="FAILED", error=error.to_dict())
        return record
    except Exception as exc:  # noqa: BLE001 — isolé à ce bloc
        record.update(status="FAILED", error={"code": "UNEXPECTED_ERROR", "message": type(exc).__name__,
                                              "traceback": traceback.format_exc(limit=3)})
        return record
    record["output"] = result.data.model_dump(mode="json")
    cache.store(key_fields, record["output"], {"usage": result.usage, "attempts": result.attempts,
                                               "duration_seconds": result.duration_seconds,
                                               "request_id": result.request_id})
    record.update(status="SUCCESS", request_id=result.request_id)
    return record


# --- Exécution ---------------------------------------------------------------------------------------------

def coverage_of(metadata: dict, stage6: dict) -> dict:
    manifest = _read(Path(metadata["output_dir"]) / BATCH_MANIFEST)
    expected = (manifest or {}).get("total") or len((manifest or {}).get("interviews") or {}) or \
        len(eligible_files(metadata))
    included = stage6.get("interview_ids", [])
    return {"expected_interviews": expected, "n_included": len(included), "included_interview_ids": included,
            "excluded_interviews": stage6.get("excluded_interviews", []), "stage6_status": stage6.get("status"),
            "stage6_blocks": [{"block_id": b["block_id"], "interview_ids": b["interview_ids"], "status": b["status"]}
                              for b in stage6.get("blocks", [])]}


def run(metadata: dict, *, settings=None, runner=None, cache: AnalysisCache | None = None, log=print) -> dict:
    settings = settings or lp.runtime_settings()
    cache = cache or AnalysisCache()
    path = out_path(metadata)
    document = {"stage7_version": STAGE7_VERSION, "run_id": metadata.get("run_id"), "generated_at": _now(),
                "model": settings.model, "execution": "locale", "api_calls": 0, "status": None, "reason": None}

    def save() -> None:
        document["updated_at"] = _now()
        write_json_atomic(path, document)

    stage6 = _read(Path(metadata["output_dir"]) / STAGE6_PATH)
    if stage6 is None:
        document.update(status=lp.STAGE_BLOCKED, reason="Étape 6 absente : lancez d'abord trace_local.py stage6 <run>.")
        save()
        return document
    document["corpus_coverage"] = coverage_of(metadata, stage6)
    if stage6.get("status") == lp.STAGE_NOT_APPLICABLE:
        document.update(status=lp.STAGE_NOT_APPLICABLE, reason=STAGE6_NOT_APPLICABLE_REASON)
        save()
        return document
    if stage6.get("status") != lp.STAGE_COMPLETE:
        document.update(status=lp.STAGE_BLOCKED, reason=f"Étape 6 non terminée ({stage6.get('status')}"
                        + (f" — {stage6['reason']}" if stage6.get("reason") else "")
                        + ") : relancez trace_local.py stage6 <run> avant l'étape 7.")
        save()
        return document
    items = evidence_items(stage6)
    index = EvidenceIndex(metadata, items)
    coverage = document["corpus_coverage"]
    blocks = thematic_blocks(stage6, items)
    if not blocks:
        document.update(status=lp.STAGE_BLOCKED, reason="Étape 6 sans aucune affirmation validée : rien à théoriser.")
        save()
        return document
    if runner is None:
        runner = lp.make_runner(settings, journal_dir=Path(metadata["output_dir"]) / lp.LOCAL_RUNS_SUBDIR / "stage7")
    document["thematic_blocks"] = [{**b, "status": "PENDING"} for b in blocks]
    save()
    propositions: dict[str, dict] = {}
    rejected, issues = [], []
    try:
        for block in document["thematic_blocks"]:
            message, local = block_message(block, items, coverage)
            log(f"Étape 7 — bloc {block['block_id']} ({block['theme']}, {len(local)} élément(s))…")
            record = _call(tb.BLOCK_SPEC, message, f"{metadata['run_id']}/{tb.BLOCK_SPEC.name}/{block['block_id']}",
                           metadata["run_id"], runner, cache, settings)
            block.update(cache_hit=record["cache_hit"], call_status=record["status"], error=record["error"])
            if record["output"] is None:
                block["status"] = "FAILED"
                log(f"Étape 7 — bloc {block['block_id']} : ÉCHEC ({(record['error'] or {}).get('message')})")
            else:
                validated = validate_block(record["output"], local, index, block["block_id"])
                block.update(status=validated["status"], propositions=validated["propositions"],
                             rejected=validated["rejected"], issues=validated["issues"])
                log(f"Étape 7 — bloc {block['block_id']} : {validated['status']} ({len(validated['propositions'])} "
                    f"proposition(s){', cache' if record['cache_hit'] else ''})")
            save()
        for block in document["thematic_blocks"]:
            for proposition in block.get("propositions", []):
                propositions[f"P{len(propositions) + 1:02d}"] = proposition
            rejected += block.get("rejected", [])
            issues += block.get("issues", [])
        failed = [b["block_id"] for b in document["thematic_blocks"] if b["status"] == "FAILED"]
        synthesis = {"status": "SKIPPED", "cache_hit": False, "error": None}
        output = None
        if propositions:
            log(f"Étape 7 — synthèse finale ({len(propositions)} proposition(s) des blocs)…")
            record = _call(tb.SYNTHESIS_SPEC, synthesis_message(propositions, coverage),
                           f"{metadata['run_id']}/{tb.SYNTHESIS_SPEC.name}", metadata["run_id"], runner, cache,
                           settings)
            synthesis = {"status": record["status"], "cache_hit": record["cache_hit"], "error": record["error"]}
            output = record["output"]
            log(f"Étape 7 — synthèse : {record['status']}")
    except LLMError as error:
        document.update(status=lp.STAGE_FAILED, reason=f"runtime local indisponible ({error.code}) : relancez pour reprendre")
        save()
        raise
    merged = validate_synthesis(output, propositions) if propositions else {"theory_claims": [], "relations": [],
                                                                            "issues": []}
    issues += merged["issues"]
    for pid, p in propositions.items():
        p["proposition_id"] = pid
    claims = merged["theory_claims"]
    by_type = {name: [c["theory_claim_id"] for c in claims if c["proposition_type"] in types]
               for name, types in SECTION_TYPES.items()}
    categories: dict[str, list[str]] = {}
    for c in claims:
        categories.setdefault(c.get("category") or "?", []).append(c["theory_claim_id"])
    synthesis_failed = bool(propositions) and output is None
    limitations = [
        "Structure théorique produite par un modèle local à partir des seules comparaisons validées de l'étape 6 "
        "(aucune transcription) ; à relire avant toute rédaction.",
        "Les preuves (entretiens, affirmations des étapes 6 et 5, épisodes) sont reconstituées par TRACE à partir des "
        "éléments cités, jamais recopiées par le modèle.",
        "Une proposition appuyée par moins de deux entretiens est une hypothèse ou un cas individuel, jamais une "
        "régularité du corpus.",
        "Les comparaisons de l'étape 6 ont été menées par blocs d'entretiens : une régularité transversale aux blocs "
        "repose sur la synthèse de l'étape 7, à vérifier dans les preuves.",
    ]
    if coverage["n_included"] < coverage["expected_interviews"]:
        limitations.append(f"Corpus incomplet : {coverage['n_included']} entretien(s) inclus sur "
                           f"{coverage['expected_interviews']} prévu(s) ; exclus : "
                           + (", ".join(e["interview_id"] for e in coverage["excluded_interviews"]) or "—") + ".")
    if failed:
        limitations.append("Bloc(s) thématique(s) en échec : " + ", ".join(failed) + " — relancez pour ne rejouer qu'eux.")
    if synthesis_failed:
        limitations.append("Synthèse finale en échec : propositions des blocs conservées sans fusion ; relancez.")
    review = [f"{c['theory_claim_id']} ({c['proposition_type']}, {c['scope']}) : à revoir" for c in claims
              if c["needs_review"]] + issues + [f"proposition rejetée ({r['block_id']}) : {r['reason']}" for r in rejected]
    document.update(
        synthesis=synthesis, propositions=list(propositions.values()), rejected_propositions=rejected,
        theory_claims=claims, relations=merged["relations"],
        synthesis_summary=(output or {}).get("synthesis_summary"),
        main_categories=[{"category": name, "theory_claim_ids": ids} for name, ids in
                         sorted(categories.items(), key=lambda kv: -len(kv[1]))],
        structuring_regularities=[c["theory_claim_id"] for c in claims
                                  if c["level"] == "structuring" and c["scope"] == CORPUS],
        **by_type, needs_review=review, limitations=limitations,
        evidence_index={"cross_claims": {k: {f: items[k].get(f) for f in (
            "claim_type", "description", "interview_ids", "trajectory_claim_ids", "needs_review")}
            for k in sorted({k for c in claims for k in c["supporting_cross_claim_ids"]})},
            "trajectory_claims": index.trajectory_claims, "episodes": index.episodes},
        status=lp.STAGE_FAILED if failed or synthesis_failed else lp.STAGE_COMPLETE,
        reason=("bloc(s) en échec : " + ", ".join(failed)) if failed else
        ("synthèse finale en échec" if synthesis_failed else None))
    save()
    return document


def summary_lines(document: dict) -> list[str]:
    coverage = document.get("corpus_coverage") or {}
    lines = [f"Run {document['run_id']} — étape 7 (théorisation transversale) : {document['status']}"
             + (f" — {document['reason']}" if document.get("reason") else "")]
    if coverage:
        lines.append(f"  corpus : {coverage['n_included']} entretien(s) inclus sur {coverage['expected_interviews']} "
                     f"prévu(s) ; exclus : {len(coverage['excluded_interviews'])}")
    for block in document.get("thematic_blocks", []):
        lines.append(f"  {block['block_id']} {block['status']} — {block['theme']} : "
                     f"{len(block.get('propositions', []))} proposition(s)")
    if "theory_claims" in document:
        claims = document["theory_claims"]
        lines.append(f"  propositions théoriques : {len(claims)} (structurantes : "
                     f"{len(document['structuring_regularities'])}, hypothèses / cas individuels : "
                     f"{sum(c['scope'] == INDIVIDUAL for c in claims)}) ; relations : {len(document['relations'])}")
    return lines
