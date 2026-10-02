"""Workflow Claude Code de TRACE — commandes déterministes (aucun appel API, aucune clé).

Claude Code exécute ces commandes ; les agents sémantiques sont joués par Claude Code lui-même, à partir
des prompts du dépôt (voir TRACE_WORKFLOW.md). Étapes 3 et 4 pour l'instant.

    python scripts/trace_workflow.py ingest FICHIER [FICHIER ...]   # crée un run et l'ingère (étapes 1-2)
    python scripts/trace_workflow.py run RUN --until 4              # un passage jusqu'à l'étape 4 : étape 3 si
                                                                    # elle n'est pas complète, puis étape 4
    python scripts/trace_workflow.py stage3 RUN [--interview ID]    # un passage de l'étape 3 : consomme les
                                                                    # réponses, écrit les tâches suivantes
    python scripts/trace_workflow.py stage4 RUN [--interview ID]    # un passage de l'étape 4
    python scripts/trace_workflow.py tasks RUN [--stage 4]          # tâches en attente / à corriger
    python scripts/trace_workflow.py check DOSSIER_DE_TACHE         # valide response.json (validateurs existants)
    python scripts/trace_workflow.py status RUN                     # état du run

RUN : identifiant du run (dossier data/outputs/<RUN>), chemin de son dossier, ou « latest ».
Codes de sortie de `run`, `stage3`, `stage4` : 0 étape terminée, 3 tâches en attente, 4 réponses à corriger,
5 échec, 6 étape bloquée (étape précédente en échec ou absente). `check` : 0 aucune anomalie bloquante, 1 sinon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import claude_code_workflow as wf  # noqa: E402
from core import config  # noqa: E402
from core.ingestion import ingest_run  # noqa: E402
from core.run_manager import TraceError, init_run, list_runs, load_metadata  # noqa: E402

EXIT_CODES = {wf.STAGE_COMPLETE: 0, wf.STAGE_AWAITING: 3, wf.STAGE_INVALID: 4, wf.STAGE_FAILED: 5,
              wf.STAGE_BLOCKED: 6}


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


def _print_tasks(status: dict, run: str, next_command: str | None = None) -> None:
    if status["invalid"]:
        print("\nRéponses à CORRIGER (check signale une anomalie bloquante) :")
        for t in status["invalid"]:
            print(f"  - [{t['agent_label']}] {t['task_dir']}\n      {t['error']}")
    if status["pending"]:
        print("\nTâches EN ATTENTE (une tâche = un sous-agent trace-agent, sans lire les autres tâches) :")
        for t in status["pending"]:
            print(f"  - [{t['agent_label']}] {t['task_file']}")
    if status["pending"] or status["invalid"]:
        print(f"\nPour chaque tâche : réponse dans response.json, puis {wf.CLI} check <dossier de la tâche>.")
        stage = status.get("stage", "3")
        print(f"Ensuite : {next_command or f'{wf.CLI} stage{stage} {run}'}")


def print_status(status: dict, run: str, next_command: str | None = None) -> None:
    stage = status.get("stage", "3")
    print(f"Run {status['run_id']} — étape {stage} (workflow Claude Code, {status['api_calls']} appel API) : "
          f"{status['status']} ({wf.STATUS_LABELS[status['status']]})")
    print(f"Tâches : {status['answered_count']}/{status['task_count']} avec une réponse conforme au schéma.")
    for i in status["interviews"]:
        agents = ", ".join(f"{name} {s}" for name, s in i["agent_statuses"].items()) or "agents non lancés"
        print(f"  {i['interview_id']} : {i['status']} — phase {i['phase']} — {agents}")
        if i.get("reason"):
            print(f"    {i['reason']}")
    _print_tasks(status, run, next_command)


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
    print(f"Dossier du run : {wf._display_path(Path(run['output_dir']))}")
    print(f"Étape suivante : {wf.CLI} run {run['run_id']} --until 4   (ou stage3 {run['run_id']})")
    return 0


def _report(result: dict, run_dir: Path, args, next_command: str | None = None) -> int:
    status = result["status"]
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_status(status, args.run, next_command)
        if status["status"] == wf.STAGE_COMPLETE:
            print(f"\nSorties de l'étape {status['stage']} : {wf._display_path(run_dir / config.INTERVIEWS_SUBDIR)}"
                  f"/<entretien>/{config.ANALYSIS_SUBDIR}/")
    return EXIT_CODES[status["status"]]


def cmd_stage3(args) -> int:
    run_dir = resolve_run(args.run)
    return _report(wf.run_stage3(load_metadata(run_dir), args.interview or None), run_dir, args)


def cmd_stage4(args) -> int:
    run_dir = resolve_run(args.run)
    return _report(wf.run_stage4(load_metadata(run_dir), args.interview or None), run_dir, args)


def cmd_run(args) -> int:
    run_dir = resolve_run(args.run)
    result = wf.run_until(load_metadata(run_dir), args.until, args.interview or None)
    return _report(result, run_dir, args, f"{wf.CLI} run {args.run} --until {args.until}")


def cmd_tasks(args) -> int:
    status = wf.read_status(resolve_run(args.run), args.stage)
    if status is None:
        print(f"Aucun passage de l'étape {args.stage} pour ce run : lancez {wf.CLI} stage{args.stage} {args.run}")
        return 1
    if args.json:
        print(json.dumps({"pending": status["pending"], "invalid": status["invalid"]}, ensure_ascii=False, indent=2))
    else:
        _print_tasks(status, args.run)
        if not status["pending"] and not status["invalid"]:
            print("Aucune tâche en attente.")
    return 0


def cmd_check(args) -> int:
    report = wf.check_task(Path(args.task))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Tâche {report['task_id']} ({report['agent_label']}) : "
              f"{'OK — aucune anomalie bloquante' if report['ok'] else 'À CORRIGER'}")
        if report["counts"]:
            print("  " + ", ".join(f"{k} : {v}" for k, v in report["counts"].items()))
        for title, lines in (("Anomalies BLOQUANTES (à corriger)", report["blocking"]),
                             ("Avertissements (à corriger si la consigne de l'agent le permet)", report["warnings"]),
                             ("Informations", report["info"])):
            if lines:
                print(f"{title} :")
                for line in lines:
                    print(f"  - {line}")
    return 0 if report["ok"] else 1


def cmd_status(args) -> int:
    run_dir = resolve_run(args.run)
    metadata = load_metadata(run_dir)
    print(f"Run {metadata['run_id']} ({wf._display_path(run_dir)})")
    for step, state in metadata["pipeline"].items():
        print(f"  {step} : {state}")
    for stage in wf.STAGES:
        status = wf.read_status(run_dir, stage)
        if status:
            print()
            print_status(status, args.run)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Workflow Claude Code de TRACE (étapes 3 et 4, sans appel API).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="crée un run et ingère les entretiens (déterministe)")
    p.add_argument("files", nargs="+")
    p.add_argument("--problematique", default="")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("stage3", help="rejoue l'étape 3 : consomme les réponses, écrit les tâches suivantes")
    p.add_argument("run")
    p.add_argument("--interview", action="append", help="entretien à traiter (répétable ; défaut : tous)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stage3)
    p = sub.add_parser("stage4", help="rejoue l'étape 4 : consomme les réponses, écrit les tâches suivantes")
    p.add_argument("run")
    p.add_argument("--interview", action="append", help="entretien à traiter (répétable ; défaut : tous)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stage4)
    p = sub.add_parser("run", help="un passage jusqu'à l'étape --until (l'étape 3 seulement si elle n'est pas "
                                   "complète, puis l'étape 4)")
    p.add_argument("run")
    p.add_argument("--until", choices=wf.STAGES, default="4")
    p.add_argument("--interview", action="append", help="entretien à traiter (répétable ; défaut : tous)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("tasks", help="tâches en attente ou à corriger")
    p.add_argument("run")
    p.add_argument("--stage", choices=wf.STAGES, default="3")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_tasks)
    p = sub.add_parser("check", help="valide response.json d'une tâche (sans rien enregistrer)")
    p.add_argument("task", help="dossier de la tâche (ou son task.json)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("status", help="état du run")
    p.add_argument("run")
    p.set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    config.load_env_file()  # mêmes tailles de blocs (TRACE_*_CHUNK_TOKENS) que l'application
    try:
        return args.func(args)
    except (TraceError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
