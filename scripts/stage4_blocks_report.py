"""Étape 4 : coût des blocs de l'Accountability Episode Builder, estimé (sans Ollama) ou mesuré (journaux d'un run).

    python scripts/stage4_blocks_report.py plan RUN [--interview ID] [--block 4 --block 12]
        # AUCUN appel : blocs prévus, tokens d'entrée par bloc, sortie estimée (moyenne et pire cas), fenêtre et
        # réponse maximale réservée, risque de troncature, durée estimée — pour chaque taille de bloc demandée
    python scripts/stage4_blocks_report.py measure RUN
        # AUCUN appel : lit les journaux de la dernière exécution de l'étape 4 (local_runs/stage4/) — appels,
        # tokens d'entrée et de sortie réels par bloc, durée, tentatives, troncatures, réparations ciblées
    python scripts/stage4_blocks_report.py grounding RUN [--plan | --rescore] [--model M] [--solid E007,…] [--doubtful …] [--false …]
        # RAPPORT SEULEMENT : fondement de chaque épisode dans ses citations (Episode Grounding Checker, un appel par
        # épisode, réponses en cache) ; --plan : aucun appel (coût et avertissements formels) ; --rescore : aucun
        # appel, verdicts recalculés avec l'agrégation actuelle sur les réponses du dernier rapport. Modèle du seul
        # vérificateur : --model ou TRACE_GROUNDING_MODEL (le reste du pipeline garde TRACE_LOCAL_MODEL ; rapport,
        # journaux et cache propres à chaque modèle : deux modèles se comparent sur les mêmes épisodes). Aucune sortie de
        # l'étape 4 n'est modifiée ; comparaison facultative avec un audit manuel
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
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import accountability, config  # noqa: E402
from core import accountability_candidates as ac  # noqa: E402
from core import local_pipeline as lp  # noqa: E402
from core import stage4_grounding, stage4_repair  # noqa: E402
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


def _ids(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _turns(row: dict, quote_ids: list[str]) -> str:
    by_id = {q["quote_id"]: q["turn_id"].rsplit("_", 1)[-1] for q in row["quotes"]}
    return ", ".join(f"{q} ({by_id.get(q, '?')})" for q in quote_ids) or "—"


def print_grounding(result: dict) -> None:
    print(f"\n{result['interview_id']} : " + ", ".join(f"{k} {v}" for k, v in result["counts"].items() if v))
    print("  épisode | verdict | problème | moves concernés | citations concernées | raison")
    for row in result["rows"]:
        short = row["episode_id"].rsplit("_", 1)[-1]
        if row["verdict"] == stage4_grounding.NOT_CHECKED:
            print(f"  {short} | non vérifié | {(row['error'] or {}).get('message', '')} | — | — | —")
            continue
        findings = row["findings"] or [{"problem": "—", "claim_ids": [], "quote_ids": [], "reason": ""}]
        for n, finding in enumerate(findings):
            moves = ", ".join(c for c in finding["claim_ids"] if c.startswith("M")) or "—"
            head = f"{short} | {row['verdict']}" if n == 0 else "     |"
            print(f"  {head} | {finding['problem']} | {moves} | {_turns(row, finding['quote_ids'])} | "
                  f"{finding['reason'][:220]}")
        if row["formal_warnings"]:
            print(f"       | avertissements formels : {', '.join(row['formal_warnings'])}")


def cmd_grounding(args) -> int:
    metadata = load_metadata(resolve_run(args.run))
    settings = stage4_grounding.grounding_settings()
    if args.model:
        settings = dataclasses.replace(settings, model=args.model)
    files = _files(metadata, args.interview)
    dirs = [Path(f["ingestion"]["output_dir"]) for f in files]
    manual = {k: _ids(v) for k, v in (("solide", args.solid), ("douteux", args.doubtful), ("faux", args.false))
              if _ids(v)}
    pipeline_model = lp.runtime_settings().model
    print(f"Run {metadata['run_id']} — étape 4, contrôle sémantique en RAPPORT SEULEMENT (vérificateur : modèle "
          f"{settings.model}{'' if settings.model == pipeline_model else f' ; pipeline : {pipeline_model}'}) : "
          "aucune sortie de l'étape 4 n'est modifiée, aucune réparation")
    if args.plan:
        cache = AnalysisCache()
        for directory in dirs:
            planned = stage4_grounding.plan(directory)
            transcript = stage4_grounding.read_episodes(directory)[0] if planned else None
            todo = [p for p in planned if not cache.contains(stage4_grounding.SPEC.name, compute_cache_key(
                stage4_grounding.cache_key_fields(transcript, settings, p["request"]["message"])))]
            tokens = [estimate_prompt_tokens([{"content": stage4_grounding.SPEC.system_prompt},
                                              {"content": p["request"]["message"]}]) for p in planned]
            print(f"\n{directory.name} : {len(planned)} épisode(s), {len(todo)} appel(s) à faire (le reste en cache), "
                  f"entrée ≈ {min(tokens, default=0)}–{max(tokens, default=0)} tokens par appel, réponse réservée "
                  f"{OUTPUT_RESERVE[stage4_grounding.SPEC.name]}")
            for item in planned:
                if item["formal_warnings"]:
                    print(f"  {item['episode_id'].rsplit('_', 1)[-1]} : {', '.join(item['formal_warnings'])}")
        return 0
    if args.rescore:
        path = stage4_grounding.report_path(metadata, settings.model)
        if not path.is_file():
            print(f"Aucun rapport de contrôle sémantique pour le modèle {settings.model} dans ce run : lancez d'abord "
                  "le diagnostic.")
            return 1
        previous = json.loads(path.read_text(encoding="utf-8"))
        print(f"Recalcul SANS appel : réponses du rapport existant (vérificateur {previous.get('agent_version')}), "
              f"agrégation {stage4_grounding.GROUNDING_VERSION} ; le rapport n'est pas réécrit")
        results = stage4_grounding.rescore(previous, dirs)
        for result in results:
            print_grounding(result)
        if manual:
            print_comparison(stage4_grounding.compare([r for x in results for r in x["rows"]], manual))
        return 0
    runner = lp.make_runner(settings, journal_dir=stage4_grounding.journal_dir(metadata, settings.model))
    runner.require_ready()

    async def run_all() -> list[dict]:
        try:
            return [await stage4_grounding.check_interview(d, runner, settings) for d in dirs]
        finally:
            await runner.aclose()

    results = asyncio.run(run_all())
    document = stage4_grounding.report(metadata, results, settings, manual or None)
    for result in results:
        if result["available"]:
            print_grounding(result)
    comparison = document.get("manual_comparison")
    if comparison:
        print_comparison(comparison)
    print(f"\nRapport : {lp.display_path(stage4_grounding.report_path(metadata, settings.model))}")
    return 0


def print_comparison(comparison: dict) -> None:
    if comparison:
        print(f"\nComparaison avec l'audit manuel : {comparison['agreements']}/{comparison['compared']} en accord "
              "(solide ↔ supported ; douteux ↔ doubtful ; faux ↔ unsupported ou contradicted)")
        for line in comparison["lines"]:
            print(f"  {line['episode_id']} : manuel {line['manual']}, automatique {line['automatic']}"
                  f"{'' if line['agree'] else '  ← désaccord'}")
        if comparison["missing"]:
            print(f"  absents du run : {', '.join(comparison['missing'])}")


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
    p = sub.add_parser("grounding", help="contrôle sémantique des épisodes, rapport seulement")
    p.add_argument("run")
    p.add_argument("--interview", action="append")
    p.add_argument("--plan", action="store_true", help="aucun appel : coût et avertissements formels")
    p.add_argument("--model", help="modèle local du seul vérificateur (sinon TRACE_GROUNDING_MODEL, sinon "
                                   "TRACE_LOCAL_MODEL)")
    p.add_argument("--rescore", action="store_true",
                   help="aucun appel : verdicts recalculés avec l'agrégation actuelle sur le dernier rapport")
    p.add_argument("--solid", help="audit manuel : épisodes solides (E007,E010,…)")
    p.add_argument("--doubtful", help="audit manuel : épisodes douteux")
    p.add_argument("--false", help="audit manuel : épisodes faux")
    p.set_defaults(func=cmd_grounding)
    p = sub.add_parser("repairs", help="réparations ciblées prévues, d'après les blocs en cache, sans appel")
    p.add_argument("run")
    p.set_defaults(func=cmd_repairs)
    args = parser.parse_args(argv)
    config.load_env_file()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
