"""Étape 6 sur un run (lot de N entretiens) : comparaison inter-entretiens PAR BLOCS, puis fusion déterministe.

    python scripts/trace_local.py stage6 <run_id>

    <run>/batch/batch_manifest.json → entretiens `stage5_valid` (sinon : entretiens dont l'étape 5 est COMPLETE)
      → triplets de l'étape 5 déjà enregistrés (aucune relance des étapes 1 à 5) ; contrôle habituel du corpus
        (core/cross_interview_corpus.py : COMPLETE et SUCCESS_WITH_WARNINGS exploitables, exclusions documentées)
      → 0 exploitable : BLOCKED ; 1 : NOT_APPLICABLE_SINGLE_INTERVIEW ; ≥ 2 : comparaison
      → blocs d'au plus BLOCK_SIZE entretiens, équilibrés, dans l'ordre du run : chaque bloc est une étape 6
        ORDINAIRE (local_pipeline.run_stage6 : même prompt, même validateur, même cache, son propre dossier
        cross_interview/<corpus_id>/) — sauvegardé dès sa validation ; un bloc déjà analysé à l'identique n'est jamais
        rejoué ; un bloc en échec (réponse tronquée, invalide) ne fait rejouer que lui à la relance
      → fusion DÉTERMINISTE, sans appel : affirmations validées des blocs regroupées par catégorie (régularités,
        variations, tensions, cas négatifs et exceptions, différences de trajectoires), chacune avec ses entretiens,
        ses affirmations de l'étape 5 (`trajectory_claim_id`) et son bloc ; répartitions des configurations et des
        critères du métier d'étudiant sur TOUS les entretiens inclus, comptées directement dans les sorties de l'étape 5
        (seulement ≥ 2 entretiens : aucune généralisation sur un seul) ; needs_review et limites
      → <run>/stage6/stage6_corpus.json, réécrit après chaque bloc (reprise après interruption).

Taille des blocs (mesurée sans modèle sur la représentation habituelle) : ≈ 600 tokens par entretien synthétique,
≈ 1 300 pour un entretien long ; 6 entretiens ≈ 4 500 à 8 000 tokens de matériau, réponse sous la réserve de sortie du
comparateur (8 192) ; 17 entretiens → 3 blocs (6, 6, 5), chacun en mode comparatif (≥ 3 entretiens).

Pas de synthèse par le modèle entre les blocs : une régularité n'est affirmée qu'à l'intérieur d'un bloc, ou par les
comptes déterministes de l'étape 5 ; la limite est écrite dans la sortie. Aucun Grounding Checker, aucun benchmark.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from core import config, cross_interview
from core import cross_interview_corpus as corpus
from core import local_pipeline as lp
from core.analysis import eligible_files
from core.analysis_cache import write_json_atomic
from core.final_report import STAGE6_NOT_APPLICABLE_REASON
from core.llm_client import LLMError

STAGE6_BLOCKS_VERSION = "1.0"
BLOCK_SIZE = 6
OUT_DIRNAME = "stage6"
OUT_FILENAME = "stage6_corpus.json"
BATCH_MANIFEST = Path("batch") / "batch_manifest.json"
BLOCKED_REASON = "Aucun entretien exploitable à l'étape 5 : la comparaison inter-entretiens est impossible."
CATEGORIES = {
    "regularities": ("recurring_boundary", "recurring_accounting_move", "recurring_student_role_criterion",
                     "ordinary_zone_pattern", "contextual_association"),
    "variations": ("divergent_boundary", "divergent_accounting_move", "divergent_student_role_criterion"),
    "tensions": ("unresolved_cross_case_contrast",),
    "negative_cases_and_exceptions": ("negative_case", "exception_pattern"),
    "trajectory_differences": ("temporal_pattern", "minority_configuration"),
}
DONE = set(cross_interview.DONE_STATUSES)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- Entrées : entretiens retenus et exclus --------------------------------------------------------------

def selection(metadata: dict) -> dict:
    """Entretiens retenus (`stage5_valid` du manifeste du lot, sinon étape 5 COMPLETE) et exclus, avec la raison."""
    path = Path(metadata["output_dir"]) / BATCH_MANIFEST
    infos = {f["ingestion"]["interview_id"]: f for f in eligible_files(metadata)}
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        valid = [iid for iid in manifest.get("stage5_valid") or [] if iid in infos]
        excluded = []
        for iid, entry in (manifest.get("interviews") or {}).items():
            if iid in valid:
                continue
            error = next((s.get("error") for s in (entry.get("stages") or {}).values() if s.get("error")),
                         entry.get("ingestion_error"))
            excluded.append({"interview_id": iid, "source": "batch_manifest",
                             "reason": f"lot : {entry.get('final_status')}" + (f" — {error}" if error else "")})
        return {"source": "batch_manifest", "interview_ids": valid, "excluded": excluded}
    valid, excluded = [], []
    for iid, info in infos.items():
        state = lp.stage_state("5", info)
        if state["status"] == "COMPLETE":
            valid.append(iid)
        else:
            excluded.append({"interview_id": iid, "source": "run",
                             "reason": f"étape 5 {state['status']}" + (f" — {' '.join(state.get('reasons') or [])}"
                                                                         if state.get("reasons") else "")})
    return {"source": "run", "interview_ids": valid, "excluded": excluded}


def uploads_for(metadata: dict, interview_ids: list[str]) -> list[tuple[str, bytes]]:
    names = {f"{iid}_{kind}" for iid in interview_ids for kind in corpus.KINDS}
    return [u for u in corpus.run_stage5_uploads(metadata) if u[0] in names]


def plan_blocks(interview_ids: list[str], size: int = BLOCK_SIZE) -> list[list[str]]:
    """Blocs équilibrés d'au plus `size` entretiens, dans l'ordre (17, 6 → 6, 6, 5 ; 7, 6 → 4, 3)."""
    if not interview_ids:
        return []
    count = math.ceil(len(interview_ids) / size)
    base, extra = divmod(len(interview_ids), count)
    blocks, start = [], 0
    for index in range(count):
        length = base + (1 if index < extra else 0)
        blocks.append(interview_ids[start:start + length])
        start += length
    return blocks


# --- Fusion (déterministe) ---------------------------------------------------------------------------------

def _distribution(documents: dict[str, dict], extract) -> list[dict]:
    """{valeur: entretiens} sur tous les entretiens inclus ; seulement les valeurs portées par ≥ 2 entretiens."""
    groups: dict[str, list[str]] = {}
    for iid, doc in documents.items():
        for value in dict.fromkeys(extract(doc)):
            if value:
                groups.setdefault(value, []).append(iid)
    return [{"value": value, "interview_ids": ids, "n_interviews": len(ids)}
            for value, ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(ids) >= 2]


def merge(blocks: list[dict], documents: dict[str, dict]) -> dict:
    categories = {name: [] for name in CATEGORIES}
    rejected, review = 0, []
    for block in blocks:
        comparison = block.get("comparison")
        if not comparison:
            continue
        for claim in comparison.get("cross_case_claims", []):
            if not claim.get("usable_for_next_stages"):
                rejected += 1
                continue
            category = next((name for name, types in CATEGORIES.items() if claim.get("claim_type") in types), None)
            if category is None:
                continue
            support = claim.get("support", [])
            item = {"block_id": block["block_id"], "cross_claim_id": claim.get("cross_claim_id"),
                    "claim_type": claim.get("claim_type"), "description": claim.get("description"),
                    "criterion_label": claim.get("criterion_label"),
                    "interview_ids": sorted({s.get("interview_id") for s in support if s.get("interview_id")}),
                    "trajectory_claim_ids": [c for s in support for c in s.get("claim_ids", [])],
                    "student_role_criterion_ids": [c for s in support for c in s.get("criterion_ids", [])],
                    "counterexamples": [{"interview_id": c.get("interview_id"), "relation": c.get("relation"),
                                         "description": c.get("description"),
                                         "trajectory_claim_ids": c.get("claim_ids", [])}
                                        for c in claim.get("counterexamples", [])
                                        if c.get("validation_status") == "supported"],
                    "n_supporting_interviews": claim.get("n_supporting_interviews"),
                    "needs_review": bool(claim.get("needs_review"))}
            categories[category].append(item)
            if item["needs_review"]:
                review.append(f"{block['block_id']} {item['cross_claim_id']} ({item['claim_type']}) : à revoir")
    configurations = _distribution(documents, lambda d: [d.get("configuration_type")])
    criteria = _distribution(documents, lambda d: [c.get("criterion") for c in d.get("student_role_criteria", [])
                                                  if c.get("usable_for_next_stages", True)])
    return {**categories, "recurring_configurations": configurations, "recurring_student_role_criteria": criteria,
            "configuration_by_interview": {iid: d.get("configuration_type") for iid, d in documents.items()},
            "rejected_block_claims": rejected, "claims_needing_review": review}


# --- Exécution ---------------------------------------------------------------------------------------------

def out_path(metadata: dict) -> Path:
    return Path(metadata["output_dir"]) / OUT_DIRNAME / OUT_FILENAME


def run(metadata: dict, *, settings=None, base_dir: Path | None = None, block_size: int = BLOCK_SIZE,
        log=print) -> dict:
    """Étape 6 du run, par blocs. Renvoie le document (aussi écrit dans <run>/stage6/stage6_corpus.json).
    Lève LLMError si le runtime local est injoignable (document partiel enregistré : relance = reprise)."""
    settings = settings or lp.runtime_settings()
    chosen = selection(metadata)
    uploads = uploads_for(metadata, chosen["interview_ids"])
    checked = corpus.check_corpus(uploads)
    usable = [iid for iid in chosen["interview_ids"] if iid in checked["usable"]]
    excluded = chosen["excluded"] + [{"interview_id": r["interview_id"], "source": "corpus_check",
                                      "reason": " ".join(r["reasons"])}
                                     for r in checked["rows"] if r["status"] == corpus.STATUS_EXCLUDED]
    documents = {iid: checked["usable"][iid][corpus.KIND_TRAJECTORY][2] for iid in usable}
    document = {"stage6_blocks_version": STAGE6_BLOCKS_VERSION, "run_id": metadata.get("run_id"),
                "generated_at": _now(), "model": settings.model, "execution": "locale", "api_calls": 0,
                "selection_source": chosen["source"], "interview_ids": usable, "excluded_interviews": excluded,
                "n_usable": len(usable), "block_size": block_size, "blocks": [], "status": None, "reason": None}
    path = out_path(metadata)

    def save() -> None:
        document["updated_at"] = _now()
        write_json_atomic(path, document)

    if len(usable) == 0:
        document.update(status=lp.STAGE_BLOCKED, reason=BLOCKED_REASON)
    elif len(usable) == 1:
        document.update(status=lp.STAGE_NOT_APPLICABLE, reason=STAGE6_NOT_APPLICABLE_REASON)
    if document["status"]:
        save()
        return document
    plan = plan_blocks(usable, block_size)
    document["blocks"] = [{"block_id": f"B{n:02d}", "interview_ids": ids, "status": "PENDING"}
                          for n, ids in enumerate(plan, start=1)]
    save()
    for block in document["blocks"]:
        ids = block["interview_ids"]
        log(f"Étape 6 — bloc {block['block_id']} ({len(ids)} entretien(s) : {', '.join(ids)})…")
        try:
            result = lp.run_stage6(uploads_for(metadata, ids), settings=settings, base_dir=base_dir)
        except LLMError:
            block["status"] = "INTERRUPTED"
            document.update(status=lp.STAGE_FAILED, reason="runtime local indisponible : relancez pour reprendre")
            save()
            raise
        status, manifest = result["status"], result["manifest"]
        outputs = cross_interview.read_outputs(Path(result["corpus_dir"]))
        comparison = json.loads(outputs["comparison"]) if outputs.get("comparison") else None
        block.update(status=status["status"], corpus_id=status["corpus_id"], corpus_dir=status["corpus_dir"],
                     skipped=status.get("skipped", False), stage6_status=manifest.get("status"),
                     mode=status.get("mode"), error=status.get("reason"),
                     validation_warning_count=manifest.get("validation_warning_count"),
                     comparison=comparison)
        save()
        log(f"Étape 6 — bloc {block['block_id']} : {status['status']} ({manifest.get('status')})"
            + (" — déjà analysé, non rejoué" if block["skipped"] else ""))
    failed = [b for b in document["blocks"] if b["status"] != lp.STAGE_COMPLETE]
    merged = merge(document["blocks"], documents)
    limitations = [
        f"Comparaison menée à l'intérieur de blocs d'au plus {block_size} entretiens : une régularité n'est affirmée "
        "que dans un bloc (entretiens cités) ; aucune synthèse par le modèle entre les blocs.",
        "Configurations et critères du métier d'étudiant : comptes déterministes des sorties validées de l'étape 5 sur "
        "tous les entretiens inclus, retenus seulement s'ils concernent au moins deux entretiens.",
        "Les comptes portent sur des entretiens, jamais des occurrences ; un élément non mentionné est non observé.",
        "Les affirmations ont été produites par un modèle local et validées de façon déterministe : à relire.",
    ]
    if failed:
        limitations.append("Bloc(s) en échec : " + ", ".join(b["block_id"] for b in failed)
                           + " — leurs entretiens ne sont pas comparés ; relancez pour ne rejouer que ces blocs.")
    if excluded:
        limitations.append(f"{len(excluded)} entretien(s) exclu(s) : voir excluded_interviews.")
    document.update(**merged, limitations=limitations,
                    needs_review=bool(failed or merged["claims_needing_review"] or any(
                        (b.get("comparison") or {}).get("needs_review") for b in document["blocks"])),
                    status=lp.STAGE_FAILED if failed else lp.STAGE_COMPLETE,
                    reason=("bloc(s) en échec : " + ", ".join(b["block_id"] for b in failed)) if failed else None)
    save()
    return document


def summary_lines(document: dict) -> list[str]:
    lines = [f"Run {document['run_id']} — étape 6 par blocs : {document['status']}"
             + (f" — {document['reason']}" if document.get("reason") else ""),
             f"  entretiens inclus : {document['n_usable']} ; exclus : {len(document['excluded_interviews'])}"]
    for block in document["blocks"]:
        lines.append(f"  {block['block_id']} {block['status']} ({block.get('stage6_status')}) : "
                     f"{', '.join(block['interview_ids'])}")
    for name in CATEGORIES:
        if name in document:
            lines.append(f"  {name} : {len(document[name])}")
    if "recurring_configurations" in document:
        lines.append("  configurations récurrentes : " + (", ".join(
            f"{c['value']} ({c['n_interviews']})" for c in document["recurring_configurations"]) or "aucune"))
    for item in document["excluded_interviews"]:
        lines.append(f"  exclu — {item['interview_id']} : {item['reason']}")
    return lines
