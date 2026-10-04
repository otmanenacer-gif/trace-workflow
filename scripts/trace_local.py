"""TRACE en ligne de commande, entièrement local (Ollama sur cet ordinateur), sans interface.

    python scripts/trace_local.py runtime                          # état du runtime : Ollama, modèle, exécution locale
    python scripts/trace_local.py ingest FICHIER [FICHIER ...]     # crée un run et l'ingère (étapes 1-2)
    python scripts/trace_local.py batch CORPUS --until 5           # LOT sans surveillance : ingestion puis étapes 3 → 5
                                                                   # entretien par entretien ; reprise automatique,
                                                                   # échec isolé, manifeste <run>/batch/batch_manifest.json
    python scripts/trace_local.py run RUN --until 5                # étapes 3 → 5 (chaque étape seulement si elle
                                                                   # n'est pas déjà complète et à jour)
    python scripts/trace_local.py stage6 RUN                       # étape 6 d'un run (lot) : entretiens stage5_valid,
                                                                   # blocs d'au plus 6 entretiens, fusion -> <run>/stage6/
    python scripts/trace_local.py stage6 SOURCE [SOURCE ...]       # étape 6 sur les sorties de l'étape 5 (runs,
                                                                   # dossiers de JSON ou fichiers JSON)
    python scripts/trace_local.py status RUN                       # état du run
    python scripts/trace_local.py stage7 RUN                       # étape 7 : théorisation transversale du corpus à partir
                                                                   # de <run>/stage6/stage6_corpus.json -> <run>/stage7/
    python scripts/trace_local.py stage8 RUN                       # étape 8 : BROUILLON du rapport corpus (étapes 6 et 7)
                                                                   # -> <run>/stage8/stage8_report_draft.json et .md
    python scripts/trace_local.py stage9 RUN                       # étape 9 : validation du brouillon (PASS / WARN / FAIL)
                                                                   # -> <run>/stage9/ (validation, rapport validé .json/.md)
    python scripts/trace_local.py stage10 RUN                      # étape 10 : LIVRAISON finale (aucun modèle, aucune
                                                                   # analyse) -> <run>/stage10/final_report.md/.json,
                                                                   # delivery_manifest.json
    python scripts/trace_local.py report RUN [--verify E005,E009]  # rapport individuel expérimental (Markdown + JSON) à partir
                                                                   # des sorties validées, SANS aucun appel au modèle ;
                                                                   # --verify : épisodes « interprétation à vérifier »
    python scripts/trace_local.py refilter RUN [--interview ID]    # réapplique la sélectivité de l'étape 3 aux réponses
                                                                   # déjà validées (cache), SANS aucun appel au modèle
    python scripts/trace_local.py job start RUN --until 5          # mêmes étapes EN ARRIÈRE-PLAN (processus détaché :
                                                                   # fermer le terminal ou Streamlit n'arrête rien)
    python scripts/trace_local.py job status RUN                   # progression : agent en cours, appels validés
    python scripts/trace_local.py job stop RUN                     # arrêt (résultats validés conservés ; relancer
                                                                   # `job start` reprend au premier appel non terminé)

RUN : identifiant du run (dossier data/outputs/<RUN>), chemin de son dossier, ou « latest ».
Modèle : TRACE_LOCAL_MODEL (défaut qwen2.5:7b) ; adresse : TRACE_OLLAMA_URL (défaut http://localhost:11434,
boucle locale uniquement). Aucune clé, aucune API externe, aucun repli.
Codes de sortie de `run` et `stage6` : 0 étape terminée, 5 échec ou analyse incomplète, 6 étape bloquée (étape
précédente en échec, absente ou périmée ; moins de deux entretiens exploitables), 7 runtime local indisponible
(Ollama arrêté, modèle absent) ; `runtime` : 0 prêt, 7 sinon ; 2 erreur d'utilisation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from core import cross_interview_corpus as corpus  # noqa: E402
from core import local_jobs  # noqa: E402
from core import local_pipeline as lp  # noqa: E402
from core.ingestion import ingest_run  # noqa: E402
from core.llm_client import LLMError  # noqa: E402
from core.run_manager import TraceError, init_run, list_runs, load_metadata  # noqa: E402

CLI = "python scripts/trace_local.py"
EXIT_CODES = {lp.STAGE_COMPLETE: 0, lp.STAGE_FAILED: 5, lp.STAGE_BLOCKED: 6, lp.STAGE_NOT_APPLICABLE: 0}
EXIT_RUNTIME = 7


def resolve_run(value: str) -> Path:
    if value == "latest":
        runs = list_runs(config.OUTPUTS_DIR)
        if not runs:
            raise TraceError(f"Aucun run dans {config.OUTPUTS_DIR}.")
        return config.OUTPUTS_DIR / runs[0]
    for candidate in (Path(value), config.OUTPUTS_DIR / value):
        if (candidate / config.METADATA_FILENAME).is_file():
            return candidate
    raise TraceError(f"Run introuvable : {value} (identifiant, chemin du dossier du run, ou « latest »).")


def print_runtime(status: dict) -> None:
    print(f"Exécution : {status['execution']} · Runtime : {status['runtime']} ({status['url']}) · "
          f"Modèle : {status['model']} · Données externes : {status['external_data']}")
    if status["available"]:
        print(f"Ollama {status['version']} actif · modèles installés : {', '.join(status['models']) or 'aucun'}")
    if status["ready"]:
        print("Runtime local prêt.")
    else:
        print(f"Runtime local NON prêt : {status['error']['message']}")
        if not status["available"]:
            print(f"Démarrer Ollama : {status['start_command']}")
        print(f"Installer le modèle : {status['install_command']}")


def cmd_runtime(args) -> int:
    status = lp.runtime_status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_runtime(status)
    return 0 if status["ready"] else EXIT_RUNTIME


def cmd_ingest(args) -> int:
    paths = [Path(p) for p in args.files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print("Fichier(s) introuvable(s) : " + ", ".join(missing), file=sys.stderr)
        return 2
    run = init_run([(p.name, p.read_bytes()) for p in paths], args.problematique or "",
                   config.INPUTS_DIR, config.OUTPUTS_DIR)
    run = ingest_run(run)
    print(f"Run {run['run_id']} — ingestion : {run['pipeline'][config.INGESTION_STEP]}")
    for f in run["files"]:
        ingestion = f["ingestion"]
        print(f"  {ingestion['interview_id']} : {ingestion['status']}, {ingestion['turn_count']} tour(s), "
              f"{ingestion['warning_count']} avertissement(s)")
    print(f"Dossier du run : {lp.display_path(Path(run['output_dir']))}")
    print(f"Étape suivante : {CLI} run {run['run_id']} --until 5")
    return 0


def print_status(status: dict) -> None:
    print(f"Run {status['run_id']} — étape {status['stage']} (exécution {status['execution']}, {status['runtime']}, "
          f"modèle {status['model']}, {status['api_calls']} appel API) : {status['status']} "
          f"({lp.STATUS_LABELS[status['status']]}) — {status['call_count']} appel(s) au modèle local, "
          f"{status['corrected_calls']} corrigé(s), {status['duration_seconds']} s")
    for i in status["interviews"]:
        agents = ", ".join(f"{name} {s}" for name, s in i["agent_statuses"].items()) or "aucun agent"
        print(f"  {i['interview_id']} : {i['status']} — {i['phase']} — {agents}")
        for line in i.get("errors") or []:
            print(f"    {line}")
        for warning in i.get("warnings") or []:
            print(f"    AVERTISSEMENT : {warning}")


def cmd_run(args) -> int:
    run_dir = resolve_run(args.run)
    stages = lp.STAGES[:lp.STAGES.index(args.until) + 1]
    metadata = load_metadata(run_dir)
    for stage in stages:
        result = lp.run_stage(stage, metadata, args.interview or None, force=args.force)
        metadata = result["metadata"]
        status = result["status"]
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print_status(status)
        if status["status"] != lp.STAGE_COMPLETE:
            return EXIT_CODES[status["status"]]
    if not args.json:
        print(f"\nSorties : {lp.display_path(run_dir / config.INTERVIEWS_SUBDIR)}/<entretien>/{config.ANALYSIS_SUBDIR}/")
    return 0


def stage6_uploads(sources: list[str]) -> list[tuple[str, bytes]]:
    """Fichiers de l'étape 5 à comparer : runs (identifiant, « latest » ou dossier), dossiers de JSON téléchargés,
    fichiers JSON."""
    uploads: list[tuple[str, bytes]] = []
    for source in sources:
        path = Path(source)
        if path.is_file():
            uploads.append((path.name, path.read_bytes()))
            continue
        try:
            run_dir = resolve_run(source)
        except TraceError:
            run_dir = None
        if run_dir is not None:
            uploads += corpus.run_stage5_uploads(load_metadata(run_dir))
        elif path.is_dir():
            uploads += [(f.name, f.read_bytes()) for f in sorted(path.rglob("*.json")) if f.is_file()]
        else:
            raise TraceError(f"Source introuvable : {source} (run, dossier ou fichier JSON).")
    if not uploads:
        raise TraceError("Aucun fichier de l'étape 5 à comparer.")
    return uploads


def cmd_stage6_run(run_dir: Path) -> int:
    """Étape 6 d'un run (lot) : entretiens `stage5_valid`, blocs, fusion (core/stage6_blocks.py)."""
    from core import stage6_blocks
    document = stage6_blocks.run(load_metadata(run_dir), log=lambda line: print(line, flush=True))
    for line in stage6_blocks.summary_lines(document):
        print(line)
    print(f"Sortie : {lp.display_path(stage6_blocks.out_path(load_metadata(run_dir)))}")
    return EXIT_CODES[document["status"]]


def cmd_stage6(args) -> int:
    if len(args.sources) == 1 and not args.force:
        try:
            run_dir = resolve_run(args.sources[0])
        except TraceError:
            run_dir = None
        if run_dir is not None:
            return cmd_stage6_run(run_dir)
    result = lp.run_stage6(stage6_uploads(args.sources), force=args.force)
    status = result["status"]
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return EXIT_CODES[status["status"]]
    print(f"Corpus {status['corpus_id']} — étape 6 (exécution {status['execution']}, {status['runtime']}, modèle "
          f"{status['model']}, {status['api_calls']} appel API) : {status['status']} "
          f"({lp.STATUS_LABELS[status['status']]}) — {status['phase']}")
    print(f"Dossier : {status['corpus_dir']} · mode {status['mode']} · {status['n_usable']} entretien(s) "
          f"exploitable(s) sur {status['n_total']} importé(s)")
    for row in status["rows"]:
        reasons = f" — {' '.join(row['reasons'])}" if row["reasons"] else ""
        print(f"  {row['interview_id']} : {row['status']}{reasons}")
    for item in status["unrecognized"]:
        print(f"  fichier ignoré — {item['file']} : {item['reason']}")
    if status.get("reason"):
        print(f"  {status['reason']}")
    for warning in status["warnings"]:
        print(f"  AVERTISSEMENT : {warning}")
    if status["status"] == lp.STAGE_COMPLETE:
        print(f"\nSorties de l'étape 6 : {status['corpus_dir']}/{config.ANALYSIS_SUBDIR}/")
    return EXIT_CODES[status["status"]]


def cmd_batch(args) -> int:
    from core import batch
    try:
        metadata = load_metadata(resolve_run(args.corpus))  # run existant : pas de nouvelle ingestion
    except TraceError:
        corpus = Path(args.corpus)
        if not corpus.is_dir():
            raise TraceError(f"Corpus introuvable : {args.corpus} (dossier d'entretiens .txt / .docx / .pdf, ou run).")
        metadata = batch.open_corpus(corpus, args.problematique or "")
    runner = batch.Batch(metadata, args.until, stage_retries=args.retries, runtime_wait=args.runtime_wait,
                         runtime_retries=args.runtime_retries, log=lambda line: print(line, flush=True))
    try:
        manifest = runner.run()
    except KeyboardInterrupt:
        print(f"Manifeste : {lp.display_path(runner.path)}")
        return 130
    print()
    for line in batch.summary_lines(manifest):
        print(line)
    print(f"Manifeste : {lp.display_path(runner.path)}")
    if len(manifest.get("stage5_valid") or []) >= 2:
        print(f"Étape 6 (entretiens dont l'étape 5 est valide) : {CLI} stage6 {manifest['run_id']}")
    if manifest["state"] == "ABORTED":
        return EXIT_RUNTIME
    return 5 if manifest["counts"].get(batch.FAILED) else 0


def cmd_stage7(args) -> int:
    from core import stage7_theory
    metadata = load_metadata(resolve_run(args.run))
    document = stage7_theory.run(metadata, log=lambda line: print(line, flush=True))
    for line in stage7_theory.summary_lines(document):
        print(line)
    print(f"Sortie : {lp.display_path(stage7_theory.out_path(metadata))}")
    return EXIT_CODES[document["status"]]


def cmd_stage8(args) -> int:
    from core import stage8_report
    metadata = load_metadata(resolve_run(args.run))
    draft = stage8_report.run(metadata, log=lambda line: print(line, flush=True))
    for line in stage8_report.summary_lines(draft):
        print(line)
    json_path, md_path = stage8_report.paths(metadata)
    print(f"Brouillon : {lp.display_path(md_path)}")
    print(f"Provenance : {lp.display_path(json_path)}")
    return EXIT_CODES[draft["status"]]


def cmd_stage9(args) -> int:
    from core import stage9_validation
    metadata = load_metadata(resolve_run(args.run))
    validation = stage9_validation.run(metadata, log=lambda line: print(line, flush=True))
    for line in stage9_validation.summary_lines(validation):
        print(line)
    out = stage9_validation.paths(metadata)
    print(f"Validation : {lp.display_path(out['validation'])}")
    if validation["status"] != stage9_validation.BLOCKED:
        print(f"Rapport validé : {lp.display_path(out['md'])}")
    return EXIT_CODES["BLOCKED"] if validation["status"] == stage9_validation.BLOCKED else 0


def cmd_stage10(args) -> int:
    from core import stage10_delivery
    metadata = load_metadata(resolve_run(args.run))
    manifest = stage10_delivery.run(metadata, log=lambda line: print(line, flush=True))
    out = stage10_delivery.paths(metadata)
    if manifest["status"] == stage10_delivery.BLOCKED:
        print("TRACE BLOCKED")
        print(f"Run: {manifest['run_id']}")
        print(f"Reason: {manifest['reason']}")
        print(f"Manifest: {lp.display_path(out['manifest'])}")
        return EXIT_CODES["BLOCKED"]
    counts = manifest["interview_counts"]
    print("TRACE COMPLETE")
    print(f"Run: {manifest['run_id']}")
    print(f"Interviews included: {counts['included_stage6']} / {counts['planned']}")
    print(f"Result: {manifest['status']}")
    print(f"Final report: {lp.display_path(out['md'])}")
    print(f"Manifest: {lp.display_path(out['manifest'])}")
    return 0


def cmd_report(args) -> int:
    from core import final_report
    result = final_report.generate(load_metadata(resolve_run(args.run)), args.verify)
    report = result["report"]
    print(f"Run {report['run_id']} — rapport individuel expérimental ({report['status']}, {report['interview_count']} "
          "entretien(s), aucun appel au modèle)")
    print(f"Étape 6 : {report['stage6']['status']}" + (f" — {report['stage6']['reason']}" if report["stage6"]["reason"]
                                                         else ""))
    if report["interpretations_to_verify"]:
        print("Interprétation à vérifier : " + ", ".join(report["interpretations_to_verify"]))
    if report["unknown_episode_ids"]:
        print("Épisode(s) inconnu(s) dans ce run (ignorés) : " + ", ".join(report["unknown_episode_ids"]))
    for item in report["interviews"]:
        if item["missing"]:
            print(f"  {item['interview_id']} : étapes incomplètes — {' ; '.join(item['missing'])}")
    print(f"Markdown : {lp.display_path(result['md_path'])}")
    print(f"JSON : {lp.display_path(result['json_path'])}")
    return 0


def cmd_status(args) -> int:
    run_dir = resolve_run(args.run)
    metadata = load_metadata(run_dir)
    print(f"Run {metadata['run_id']} ({lp.display_path(run_dir)})")
    for step, state in metadata["pipeline"].items():
        print(f"  {step} : {state}")
    for stage in lp.STAGES:
        status = lp.read_status(run_dir, stage)
        if status:
            print()
            print_status(status)
    return 0


def print_refilter(report: dict) -> None:
    print(f"Run {report['run_id']} — resélection de l'étape 3 sans modèle ({report['model_calls']} appel au modèle)")
    for i in report["interviews"]:
        print(f"\n{i['interview_id']}")
        print(f"  Pratiques : {i['practices_kept']} conservée(s) sur {i['practices_before']} auparavant ; "
              f"{len(i['practices_set_aside'])} écartée(s)")
        for row in i["previous_practices"]:
            if row["fate"] != "kept":
                print(f"    - {row['previous_id']} écartée ({row.get('reason') or row['fate']}) :{row['label']} "
                      f"— « {' / '.join(q or '' for q in row['quotes'])[:160]} »")
        print(f"  Signaux : {i['signals_kept']} conservé(s) sur {i['signals_before']} auparavant ; "
              f"{len(i['signals_set_aside'])} écarté(s) au total (anciennes règles comprises)")
        for row in i["previous_signals"]:
            if row["fate"] != "kept":
                print(f"    - {row['previous_id']} écarté ({row.get('reason') or row['fate']}) : {row['label']} "
                      f"— {', '.join(row['turn_ids'])} — « {' / '.join(q or '' for q in row['quotes'])[:160]} »")
        normalized = i["non_use_reason_normalized"]
        print(f"  non_use_reason normalisés : {len(normalized)} "
              + ", ".join(f"{n['use_status']} {n['from']} → {n['to']}" for n in normalized))
        print(f"  NON_USE_REASON_MISMATCH restants : {', '.join(i['remaining_non_use_reason_mismatch']) or 'aucun'}")


def cmd_refilter(args) -> int:
    run_dir = resolve_run(args.run)
    result = lp.refilter_stage3(load_metadata(run_dir), args.interview or None)
    if args.json:
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    else:
        print_refilter(result["report"])
        print(f"\nRapport : {lp.display_path(lp.stage_dir(result['metadata'], '3') / lp.REFILTER_REPORT_FILENAME)}. "
              f"L'étape 4 sera rejouée par la garde habituelle ({CLI} run {run_dir.name} --until 4).")
    return EXIT_CODES[result["status"]["status"]]


def print_job(job: dict | None) -> None:
    if not job:
        print("Aucune exécution en arrière-plan pour ce run.")
        return
    print(f"Exécution {job['job_id']} — étape {job.get('stage')} — {local_jobs.JOB_LABELS.get(job['state'], job['state'])}"
          f" — {job['elapsed_seconds']} s" + (f" — {job['message']}" if job.get("message") else ""))
    current = job.get("current")
    if current and job["alive"]:
        print(f"  en cours : {current['agent_label']} ({current['label']}), tentative {current['attempt']}, "
              f"{current.get('elapsed_seconds')} s, {current.get('output_tokens') or 0} tokens générés, contexte "
              f"{current.get('num_ctx')}")
    for item in job.get("checklist") or []:
        details = f" — {item['duration_seconds']} s" if item.get("duration_seconds") is not None else ""
        print(f"  {local_jobs.ITEM_ICONS[item['status']]} {item['agent_label']} — {item['interview_id']} "
              f"({local_jobs.ITEM_LABELS[item['status']]}){details}")


def cmd_job(args) -> int:
    run_dir = resolve_run(args.run)
    job_dir = local_jobs.run_job_dir(run_dir)
    if args.action == "status":
        print_job(local_jobs.read_job(job_dir))
        return 0
    if args.action == "stop":
        stopped = local_jobs.stop(job_dir)
        print("Exécution arrêtée (résultats validés conservés)." if stopped else "Aucune exécution détachée en cours.")
        return 0
    lp.make_runner(lp.runtime_settings()).require_ready()  # Ollama absent : erreur claire avant tout lancement
    try:
        job = local_jobs.start_run_job(run_dir, args.until, args.interview or None, single=False, force=args.force)
    except local_jobs.JobAlreadyRunning as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    print(f"Exécution lancée en arrière-plan ({job['job_id']}) : étapes 3 → {args.until}. Suivi : "
          f"{CLI} job status {run_dir.name} (journal : {lp.display_path(job_dir / local_jobs.LOG_FILENAME)}).")
    if args.wait:
        while (job := local_jobs.read_job(job_dir)) and job["alive"]:
            time.sleep(2)
        print_job(job)
        return 0 if job and job["state"] == local_jobs.COMPLETE else 5
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRACE local (étapes 1 à 6 avec Ollama, sans API ni clé).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("runtime", help="état du runtime local (Ollama, modèle)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_runtime)
    p = sub.add_parser("ingest", help="crée un run et ingère les entretiens (déterministe)")
    p.add_argument("files", nargs="+")
    p.add_argument("--problematique", default="")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("run", help="étapes 3 → --until avec le modèle local (étape déjà complète : non rejouée)")
    p.add_argument("run")
    p.add_argument("--until", choices=lp.STAGES, default="5")
    p.add_argument("--interview", action="append", help="entretien à traiter (répétable ; défaut : tous)")
    p.add_argument("--force", action="store_true",
                   help="rejouer même les étapes déjà complètes (rend les étapes suivantes périmées)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("batch", help="lot sans surveillance : ingestion puis étapes 3 → --until, entretien par "
                                     "entretien, reprise automatique, échecs isolés, manifeste final")
    p.add_argument("corpus", help="dossier des entretiens (.txt, .docx, .pdf) ou run existant")
    p.add_argument("--until", choices=lp.STAGES, default="5")
    p.add_argument("--problematique", default="")
    p.add_argument("--retries", type=int, default=1, help="nouvelles tentatives d'une étape en échec (défaut 1)")
    p.add_argument("--runtime-wait", type=float, default=60, help="attente si Ollama est injoignable, en s (défaut 60)")
    p.add_argument("--runtime-retries", type=int, default=30, help="attentes successives avant arrêt (défaut 30)")
    p.set_defaults(func=cmd_batch)
    p = sub.add_parser("stage6", help="étape 6 sur un corpus de sorties de l'étape 5 (corpus déjà analysé à "
                                      "l'identique : non rejoué)")
    p.add_argument("sources", nargs="+", help="runs (identifiant, « latest », dossier), dossiers ou fichiers JSON")
    p.add_argument("--force", action="store_true", help="rejouer même si ce corpus est déjà analysé à l'identique")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stage6)
    p = sub.add_parser("status", help="état du run")
    p.add_argument("run")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("stage7", help="étape 7 : théorisation transversale du corpus (à partir de l'étape 6 du run)")
    p.add_argument("run")
    p.set_defaults(func=cmd_stage7)
    p = sub.add_parser("stage8", help="étape 8 : brouillon du rapport corpus (à partir des étapes 6 et 7 du run)")
    p.add_argument("run")
    p.set_defaults(func=cmd_stage8)
    p = sub.add_parser("stage9", help="étape 9 : validation du brouillon du rapport corpus (PASS / WARN / FAIL)")
    p.add_argument("run")
    p.set_defaults(func=cmd_stage9)
    p = sub.add_parser("stage10", help="étape 10 : livraison finale (rapport validé -> rapport final, sans modèle)")
    p.add_argument("run")
    p.set_defaults(func=cmd_stage10)
    p = sub.add_parser("report", help="rapport individuel expérimental à partir des sorties validées (aucun appel)")
    p.add_argument("run")
    p.add_argument("--verify", help="épisodes dont l'interprétation est à vérifier (ex. E005,E006,E009,E011) ; "
                                    "par défaut, la liste de la génération précédente")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("refilter", help="réapplique la sélectivité de l'étape 3 aux réponses du cache, sans modèle")
    p.add_argument("run")
    p.add_argument("--interview", action="append")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_refilter)
    p = sub.add_parser("job", help="étapes 3 → --until en arrière-plan (indépendant du terminal et de Streamlit)")
    p.add_argument("action", choices=("start", "status", "stop"))
    p.add_argument("run")
    p.add_argument("--until", choices=lp.STAGES, default="5")
    p.add_argument("--interview", action="append")
    p.add_argument("--force", action="store_true")
    p.add_argument("--wait", action="store_true", help="attendre la fin et afficher le bilan")
    p.set_defaults(func=cmd_job)
    args = parser.parse_args(argv)
    config.load_env_file()  # mêmes réglages (.env) que l'application
    # une ligne par appel au modèle local : agent, tokens d'entrée et de sortie, durée, vitesse, contexte, tentatives
    log = logging.getLogger("trace.local")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    try:
        return args.func(args)
    except LLMError as exc:
        print(f"Runtime local indisponible : {exc.user_message}", file=sys.stderr)
        return EXIT_RUNTIME
    except (TraceError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
