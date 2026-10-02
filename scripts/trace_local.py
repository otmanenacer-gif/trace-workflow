"""TRACE en ligne de commande, entièrement local (Ollama sur cet ordinateur), sans interface.

    python scripts/trace_local.py runtime                          # état du runtime : Ollama, modèle, exécution locale
    python scripts/trace_local.py ingest FICHIER [FICHIER ...]     # crée un run et l'ingère (étapes 1-2)
    python scripts/trace_local.py run RUN --until 5                # étapes 3 → 5 (chaque étape seulement si elle
                                                                   # n'est pas déjà complète et à jour)
    python scripts/trace_local.py stage6 SOURCE [SOURCE ...]       # étape 6 sur les sorties de l'étape 5 (runs,
                                                                   # dossiers de JSON ou fichiers JSON)
    python scripts/trace_local.py status RUN                       # état du run

RUN : identifiant du run (dossier data/outputs/<RUN>), chemin de son dossier, ou « latest ».
Modèle : TRACE_LOCAL_MODEL (défaut qwen2.5:14b) ; adresse : TRACE_OLLAMA_URL (défaut http://localhost:11434,
boucle locale uniquement). Aucune clé, aucune API externe, aucun repli.
Codes de sortie de `run` et `stage6` : 0 étape terminée, 5 échec ou analyse incomplète, 6 étape bloquée (étape
précédente en échec, absente ou périmée ; moins de deux entretiens exploitables), 7 runtime local indisponible
(Ollama arrêté, modèle absent) ; `runtime` : 0 prêt, 7 sinon ; 2 erreur d'utilisation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from core import cross_interview_corpus as corpus  # noqa: E402
from core import local_pipeline as lp  # noqa: E402
from core.ingestion import ingest_run  # noqa: E402
from core.llm_client import LLMError  # noqa: E402
from core.run_manager import TraceError, init_run, list_runs, load_metadata  # noqa: E402

CLI = "python scripts/trace_local.py"
EXIT_CODES = {lp.STAGE_COMPLETE: 0, lp.STAGE_FAILED: 5, lp.STAGE_BLOCKED: 6}
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


def cmd_stage6(args) -> int:
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
    p = sub.add_parser("stage6", help="étape 6 sur un corpus de sorties de l'étape 5 (corpus déjà analysé à "
                                      "l'identique : non rejoué)")
    p.add_argument("sources", nargs="+", help="runs (identifiant, « latest », dossier), dossiers ou fichiers JSON")
    p.add_argument("--force", action="store_true", help="rejouer même si ce corpus est déjà analysé à l'identique")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stage6)
    p = sub.add_parser("status", help="état du run")
    p.add_argument("run")
    p.set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    config.load_env_file()  # mêmes réglages (.env) que l'application
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
