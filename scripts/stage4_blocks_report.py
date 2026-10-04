"""Étape 4 : coût des blocs de l'Accountability Episode Builder, estimé (sans Ollama) ou mesuré (journaux d'un run).

    python scripts/stage4_blocks_report.py plan RUN [--interview ID] [--block 4 --block 12]
        # AUCUN appel : blocs prévus, tokens d'entrée par bloc, sortie estimée (moyenne et pire cas), fenêtre et
        # réponse maximale réservée, risque de troncature, durée estimée — pour chaque taille de bloc demandée
    python scripts/stage4_blocks_report.py measure RUN
        # AUCUN appel : lit les journaux de la dernière exécution de l'étape 4 (local_runs/stage4/) — appels,
        # tokens d'entrée et de sortie réels par bloc, durée, tentatives, troncatures, réparations ciblées
    python scripts/stage4_blocks_report.py repairs RUN
        # AUCUN appel : réponses des blocs relues dans le cache TRACE ; identifiants étrangers retirés sans modèle et
        # réparations ciblées que la prochaine exécution demanderait (core/stage4_repair.py)

La mesure réelle s'obtient en exécutant d'abord l'étape 4 (`python scripts/trace_local.py run RUN --until 4`), puis
`measure`. Ce script ne lance jamais le modèle.

Estimation de la sortie : tokens par épisode mesurés sur le journal réel OTMANE_NACER du 2026-10-03 (11 épisodes,
7 217 tokens, 25 550 caractères : ≈ 545 tokens en moyenne, ≈ 1 080 pour le plus long ; les citations recopiées font
77 % du JSON). Vitesses par défaut : celles du même journal (qwen2.5:7b, GPU : lecture ≈ 1 900 tokens/s, génération
≈ 34 tokens/s) ; modifiables (--prompt-tps, --gen-tps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import accountability, config  # noqa: E402
from core import accountability_candidates as ac  # noqa: E402
from core import local_pipeline as lp  # noqa: E402
from core import stage4_repair  # noqa: E402
from core.analysis_cache import AnalysisCache, compute_cache_key  # noqa: E402
from core.llm_client import LLMError  # noqa: E402
from core.local_agent_runner import OUTPUT_RESERVE, context_plan, estimate_prompt_tokens  # noqa: E402
from core.run_manager import load_metadata  # noqa: E402
from scripts.trace_local import resolve_run  # noqa: E402

EPISODE_OUTPUT_TOKENS_MEAN = 545
EPISODE_OUTPUT_TOKENS_MAX = 1080
RESERVE = OUTPUT_RESERVE[accountability.SPEC.name]


def _files(metadata: dict, interview_ids: list[str] | None) -> list[dict]:
    files = accountability.eligible_files(metadata)
    return [f for f in files if not interview_ids or f["ingestion"]["interview_id"] in interview_ids]


def plan_rows(info: dict, block: int, num_ctx_max: int) -> list[dict]:
    original = ac.BLOCK_MAX_CANDIDATES
    ac.BLOCK_MAX_CANDIDATES = block
    try:
        prepared = accountability.prepare_stage4(info["ingestion"])
    finally:
        ac.BLOCK_MAX_CANDIDATES = original
    rows = []
    for request in prepared.requests:
        n = len(request["candidate_ids"])
        prompt = estimate_prompt_tokens([{"content": accountability.SPEC.system_prompt},
                                         {"content": request["user_message"]}])
        try:
            num_ctx, num_predict = context_plan(prompt, RESERVE, num_ctx_max)
        except LLMError:
            num_ctx, num_predict = None, 0
        rows.append({"chunk": request["chunk"], "candidates": n, "input_tokens": prompt, "num_ctx": num_ctx,
                     "num_predict": num_predict, "output_mean": n * EPISODE_OUTPUT_TOKENS_MEAN,
                     "output_max": n * EPISODE_OUTPUT_TOKENS_MAX,
                     "truncation_risk": n * EPISODE_OUTPUT_TOKENS_MAX > num_predict})
    return rows


def cmd_plan(args) -> int:
    metadata = load_metadata(resolve_run(args.run))
    settings = lp.runtime_settings()
    blocks = args.block or [ac.BLOCK_MAX_CANDIDATES, ac.SINGLE_CALL_MAX_CANDIDATES]
    print(f"Run {metadata['run_id']} — étape 4, ESTIMATION sans appel (modèle {settings.model}, fenêtre maximale "
          f"{settings.num_ctx}, réponse réservée {RESERVE} tokens, lecture {args.prompt_tps} t/s, génération "
          f"{args.gen_tps} t/s)")
    for info in _files(metadata, args.interview):
        print(f"\n{info['ingestion']['interview_id']}")
        for block in blocks:
            rows = plan_rows(info, block, settings.num_ctx)
            if not rows:
                print(f"  bloc ≤ {block} : aucun candidat, aucun appel")
                continue
            label = "un seul appel" if block >= ac.SINGLE_CALL_MAX_CANDIDATES else f"bloc ≤ {block} candidats"
            total_in = sum(r["input_tokens"] for r in rows)
            total_out = sum(r["output_mean"] for r in rows)
            duration = total_in / args.prompt_tps + total_out / args.gen_tps
            risky = sum(r["truncation_risk"] for r in rows)
            print(f"  {label} : {len(rows)} appel(s), {sum(r['candidates'] for r in rows)} candidats, entrée "
                  f"{total_in} tokens, sortie ≈ {total_out} tokens (pire cas {sum(r['output_max'] for r in rows)}), "
                  f"durée ≈ {duration:.0f} s, bloc(s) à risque de troncature : {risky}")
            for r in rows:
                print(f"    bloc {r['chunk']} : {r['candidates']} candidat(s) | entrée ≈ {r['input_tokens']} | sortie ≈ "
                      f"{r['output_mean']} (pire {r['output_max']}) | num_ctx {r['num_ctx']}, num_predict "
                      f"{r['num_predict']}{' | RISQUE DE TRONCATURE' if r['truncation_risk'] else ''}")
    return 0


def cmd_measure(args) -> int:
    run_dir = resolve_run(args.run)
    metadata = load_metadata(run_dir)
    journal_dir = lp.stage_dir(metadata, "4")
    status = lp.read_status(run_dir, "4")
    if status is None:
        print("Aucune exécution de l'étape 4 dans ce run.")
        return 1
    print(f"Run {metadata['run_id']} — étape 4, MESURE ({status['updated_at']}, modèle {status['model']}) : "
          f"{status['status']}")
    for info in _files(metadata, None):
        manifest_path = (Path(info["ingestion"]["output_dir"]) / config.ANALYSIS_SUBDIR
                         / config.ACCOUNTABILITY_MANIFEST_FILENAME)
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = manifest.get("chunks") or []
        print(f"\n{manifest['interview_id']} : {manifest['status']}, {manifest['candidate_count']} candidats, "
              f"{len(chunks)} bloc(s)")
        totals = {"calls": 0, "input": 0, "output": 0, "seconds": 0.0, "truncated": 0}
        for chunk in chunks:
            journal_path = journal_dir / f"{chunk.get('request_id')}.json"
            if chunk.get("cache_hit") or not journal_path.is_file():
                print(f"  bloc {chunk['chunk']} : {chunk['candidate_count']} candidat(s) — {chunk['status']}"
                      f"{' (repris du cache : 0 appel)' if chunk.get('cache_hit') else ' (journal absent)'}")
                continue
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            attempts = journal.get("attempt_log") or []
            truncated = sum(a.get("done_reason") == "length" for a in attempts)
            inputs = [a.get("prompt_eval_count") or 0 for a in attempts]
            outputs = [a.get("eval_count") or 0 for a in attempts]
            seconds = sum(a.get("duration_seconds") or 0 for a in attempts)
            totals.update(calls=totals["calls"] + len(attempts), input=totals["input"] + sum(inputs),
                          output=totals["output"] + sum(outputs), seconds=totals["seconds"] + seconds,
                          truncated=totals["truncated"] + truncated)
            print(f"  bloc {chunk['chunk']} : {chunk['candidate_count']} candidat(s) — {chunk['status']} | "
                  f"{len(attempts)} tentative(s) | entrée {inputs} | sortie {outputs} | {seconds:.0f} s | "
                  f"troncature(s) {truncated} | {journal.get('outcome')}")
        print(f"  Total : {totals['calls']} appel(s) au modèle, entrée {totals['input']} tokens, sortie "
              f"{totals['output']} tokens, {totals['seconds']:.0f} s, troncature(s) {totals['truncated']}")
        for repair in manifest.get("repairs") or []:
            print(f"  réparation {repair['kind']} {', '.join(repair['candidate_ids'])} : {repair['outcome']}"
                  f"{' (cache)' if repair.get('cache_hit') else ''}{' — ' + repair['reason'] if repair.get('reason') else ''}")
    return 0


def cmd_repairs(args) -> int:
    metadata = load_metadata(resolve_run(args.run))
    settings = lp.runtime_settings()
    cache = AnalysisCache()
    print(f"Run {metadata['run_id']} — étape 4, réparations prévues SANS appel (modèle {settings.model})")
    for info in _files(metadata, None):
        prepared = accountability.prepare_stage4(info["ingestion"])
        if prepared.blocked or not prepared.needs_llm:
            continue
        episodes, missing_blocks = [], []
        for request in prepared.requests:
            key = accountability.cache_key_fields(prepared, settings, request["user_message"])
            entry = cache.load(key, accountability.SPEC.output_model)
            if entry is None:
                missing_blocks.append(request["chunk"])
                continue
            episodes += ac.expand_ids(entry["output"], prepared.interview_id)["episodes"]
        print(f"\n{prepared.interview_id} : {prepared.candidate_count} candidats, {len(prepared.requests)} bloc(s)"
              + (f", bloc(s) absent(s) du cache : {missing_blocks} (rejoué(s) en entier)" if missing_blocks else ""))
        skip = {c for r in prepared.requests if r["chunk"] in missing_blocks for c in r["candidate_ids"]}
        for episode in episodes:
            cleaned, needed = stage4_repair.remove_foreign_ids(prepared, episode)
            removed = cleaned.get("removed_foreign_ids") or []
            if removed:
                print(f"  {', '.join(episode['candidate_ids'])} : identifiant(s) étranger(s) retiré(s) sans modèle — "
                      + ", ".join(removed))
        cleaned = [stage4_repair.remove_foreign_ids(prepared, e)[0] for e in episodes]
        tasks = stage4_repair.plan_repairs(prepared, cleaned, skip)
        for task in tasks:
            message = stage4_repair.repair_message(prepared, task)
            key = compute_cache_key(accountability.cache_key_fields(prepared, settings, message))
            done = cache.contains(accountability.SPEC.name, key)
            print(f"  réparation {task['kind']} {', '.join(task['candidate_ids'])} "
                  f"({'déjà en cache' if done else '1 appel'}, ≈ {ac.payload_estimate(message)} tokens) : "
                  + " | ".join(task["problems"]))
        if not tasks:
            print("  aucune réparation")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coût des blocs de l'étape 4 (estimé ou mesuré), sans appel.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="estimation sans Ollama")
    p.add_argument("run")
    p.add_argument("--interview", action="append")
    p.add_argument("--block", type=int, action="append", help="taille de bloc à évaluer (répétable)")
    p.add_argument("--prompt-tps", type=float, default=1900.0)
    p.add_argument("--gen-tps", type=float, default=34.0)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("measure", help="mesure réelle d'après les journaux de la dernière étape 4")
    p.add_argument("run")
    p.set_defaults(func=cmd_measure)
    p = sub.add_parser("repairs", help="réparations ciblées prévues, d'après les blocs en cache, sans appel")
    p.add_argument("run")
    p.set_defaults(func=cmd_repairs)
    args = parser.parse_args(argv)
    config.load_env_file()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
