"""Test réel (« smoke test ») de l'étape 3 sur l'entretien SYNTHÉTIQUE.

⚠️ Ce script appelle réellement l'API Anthropic et CONSOMME DES TOKENS
(2 appels pour l'entretien de l'étape 3 ; 3 pour celui de l'étape 3.5, dont
1 audit des locuteurs ; sauf si le résultat est déjà en cache). Il ne
s'exécute qu'avec l'option explicite --confirm-api-cost.

Usage (depuis la racine du projet) :
    python scripts/smoke_test_stage3.py                     # affiche ce qui serait fait, sans appel
    python scripts/smoke_test_stage3.py --confirm-api-cost  # appel réel
    python scripts/smoke_test_stage3.py --confirm-api-cost --force   # ignore le cache
    python scripts/smoke_test_stage3.py --confirm-api-cost --stage35 # entretien synthétique de l'étape 3.5

Il crée un run normal (data/inputs|outputs/<run_id>/, non versionnés) avec
l'entretien fictif de tests/synthetic_interviews.py, lance les deux agents,
puis vérifie les attentes qualitatives de la section 18 (affect explicite
« scrupules », autocorrection, minimisation, référence à la professeure,
contradiction, absence de vocabulaire interprétatif, citations exactes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config  # noqa: E402
from core.analysis import analyze_run  # noqa: E402
from core.ingestion import ingest_run  # noqa: E402
from core.llm_client import LLMError, LLMSettings  # noqa: E402
from core.run_manager import init_run  # noqa: E402
from tests import synthetic_interviews as si  # noqa: E402


def check(label: str, ok: bool) -> bool:
    print(f"  [{'OK' if ok else 'À VÉRIFIER'}] {label}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-api-cost", action="store_true", help="autorise les appels API payants")
    parser.add_argument("--force", action="store_true", help="ignore le cache TRACE (nouveaux appels)")
    parser.add_argument("--stage35", action="store_true",
                        help="entretien synthétique de l'étape 3.5 (non-usages, métadiscours, tour mal attribué)")
    args = parser.parse_args()
    filename, text = (si.STAGE35_FILES[0][0], si.STAGE35_TEXT) if args.stage35 else (si.FILENAME, si.TEXT)
    expected_calls = 3 if args.stage35 else 2

    config.load_env_file()
    settings = LLMSettings.from_env()
    if not settings.enabled:
        print(settings.disabled_reason())
        return 2
    print(f"Modèle : {settings.model} · clé API : détectée (non affichée)")
    if not args.confirm_api_cost:
        print(f"Aucun appel effectué. Relancer avec --confirm-api-cost pour lancer {expected_calls} appels réels "
              "(audit des locuteurs s'il y a des tours suspects, Practice Extractor, Interaction Signal Reader) "
              "sur l'entretien synthétique.")
        return 0

    run = ingest_run(init_run([(filename, text.encode("utf-8"))], "Smoke test étape 3 (entretien synthétique)"))
    try:
        run = analyze_run(run, settings=settings, force=args.force)
    except LLMError as exc:
        print(f"Échec : {exc.user_message}")
        return 1

    summary = run["files"][0]["analysis"]
    out = Path(summary["analysis_dir"])
    usage = run["last_analysis"]["usage"]
    print(f"\nSorties : {out}")
    print(f"{usage['api_calls']} appel(s) API — {usage['input_tokens']} tokens entrée — "
          f"{usage['output_tokens']} tokens sortie — {usage['cached_results']} résultat(s) en cache")
    for name, agent in summary["agents"].items():
        error = f" — {agent['error']['message']}" if agent.get("error") else ""
        print(f"  {name} : {agent['status']} ({agent.get('item_count') or 0} objet(s)){error}")
    if any(a["status"] == "FAILED" for a in summary["agents"].values()):
        return 1

    if args.stage35:
        return 0 if all(check_stage35(out)) else 3

    practices = json.loads((out / "practice_extractor.json").read_text(encoding="utf-8"))["practices"]
    signals = json.loads((out / "interaction_signals.json").read_text(encoding="utf-8"))["signals"]
    validation = json.loads((out / config.EVIDENCE_VALIDATION_FILENAME).read_text(encoding="utf-8"))
    kinds = {s["signal_type"] for s in signals}
    affects = " ".join((s["explicit_affect"] or "").casefold() for s in signals)
    issues = [i for a in validation["agents"].values() for i in a.get("issues", [])]

    print("\nAttentes qualitatives (section 18) :")
    results = [
        check("toutes les citations sont exactes", validation["total_invalid_evidence"] == 0),
        check("pratiques : usage, non-usage et usage hypothétique",
              {"use", "non_use", "hypothetical"} <= {p["use_status"] for p in practices}),
        check("signal : affect explicite « scrupules »", "scrupule" in affects),
        check("signal : autocorrection", "self_correction" in kinds),
        check("signal : minimisation", "minimization" in kinds),
        check("signal : référence au jugement de la professeure", "reference_to_teacher_judgment" in kinds),
        check("signal : contradiction entre tours", "cross_turn_contradiction" in kinds),
        check("aucun vocabulaire interprétatif signalé",
              not [i for i in issues if i["code"] == "INTERPRETIVE_VOCABULARY"]),
        check("aucun affect non présent dans les citations",
              not [i for i in issues if i["code"] == "AFFECT_NOT_IN_QUOTES"]),
    ]
    print("\nRelisez aussi les fichiers produits : ce script ne remplace pas une lecture humaine.")
    return 0 if all(results) else 3


def check_stage35(out: Path) -> list[bool]:
    """Attentes qualitatives de l'étape 3.5 sur l'entretien synthétique (lecture humaine toujours nécessaire)."""
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8"))  # noqa: E731
    practices, signals = load("practice_extractor.json")["practices"], load("interaction_signals.json")["signals"]
    audit, validation = load("speaker_attribution_audit.json"), load(config.EVIDENCE_VALIDATION_FILENAME)
    statuses = {p["use_status"] for p in practices}
    kinds = {s["signal_type"] for s in signals}
    issues = [i for a in validation["agents"].values() for i in a.get("issues", [])]
    suspect = [i for i in audit["items"] if i["turn_id"] == si.STAGE35_MISATTRIBUTED_TURN]
    print("\nAttentes qualitatives (étape 3.5) :")
    return [
        check("toutes les citations sont exactes", validation["total_invalid_evidence"] == 0),
        check("pratiques : usage ET non-usage ou refus distincts", "use" in statuses and bool({"non_use", "refusal"} & statuses)),
        check("pratiques : refus explicite (refusal)", "refusal" in statuses),
        check("signal : préférence (preference_statement)", "preference_statement" in kinds),
        check("signal : évaluation métadiscursive", "metadiscursive_self_evaluation" in kinds),
        check("aucune alerte « réparation » (lave-vaisselle)",
              not [i for i in issues if i["code"] == "INTERPRETIVE_VOCABULARY" and i.get("term") == "réparation"]),
        check("audit : tour mal attribué détecté", bool(suspect)),
        check("audit : suggestion « enquete » pour ce tour", bool(suspect) and suspect[0]["suggested_speaker"] == "enquete"),
        check("audit : transcription non modifiée", audit["transcript_modified"] is False),
    ]


if __name__ == "__main__":
    sys.exit(main())
