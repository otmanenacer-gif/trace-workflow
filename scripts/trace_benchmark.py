"""Mesure des performances de l'étape 3 avec le modèle local (Ollama), sur un entretien représentatif.

    python scripts/trace_benchmark.py plan                 # SANS Ollama : appels prévus, taille de chaque requête,
                                                           # fenêtre de contexte et réponse maximale par appel
    python scripts/trace_benchmark.py run                  # étape 3 complète, mesurée (Ollama + modèle installés)
    python scripts/trace_benchmark.py run --legacy         # même mesure avec l'ANCIEN réglage (contexte fixe
                                                           # TRACE_OLLAMA_NUM_CTX pour tous les appels, génération non
                                                           # bornée, 2 corrections méthodologiques) : « avant »
    python scripts/trace_benchmark.py run --compare-format # en plus : un même bloc avec le JSON Schema strict, puis
                                                           # avec le simple mode JSON (diagnostic seulement)

Entretien par défaut : benchmarks/entretien_synthetique_1h.txt (SYNTHÉTIQUE, ≈ 56 000 caractères, 349 tours, ≈ 1 h
d'entretien). Autre fichier : `--file CHEMIN` (txt, docx, pdf).

Le benchmark travaille dans un dossier temporaire (run, cache, journaux) : aucun résultat en cache n'est réutilisé et
le cache de TRACE n'est pas modifié. Il n'utilise QUE le modèle configuré (TRACE_LOCAL_MODEL, défaut qwen2.5:7b) :
aucun téléchargement, aucune API, aucun repli. Les prompts, schémas et validateurs sont ceux de TRACE, inchangés ; le
mode `--compare-format` ne sert qu'à mesurer le coût du schéma strict (sa réponse n'est jamais utilisée).

Rapport : tableau par appel (agent, tokens d'entrée et de sortie, durée, tentatives, contexte, vitesse), totaux,
estimation pour N entretiens (`--interviews 17`), et JSON complet dans benchmarks/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from core import local_pipeline as lp  # noqa: E402
from core.agent_checks import AGENT_SPECS, MethodChecker  # noqa: E402
from core.ingestion import ingest_run  # noqa: E402
from core.llm_client import LLMError, LLMSettings, parse_json_output  # noqa: E402
from core.local_agent_runner import (LocalAgentRunner, OllamaTransport, context_plan,  # noqa: E402
                                     estimate_prompt_tokens)
from core.run_manager import init_run  # noqa: E402

DEFAULT_FILE = ROOT / "benchmarks" / "entretien_synthetique_1h.txt"
RESULTS_DIR = ROOT / "benchmarks" / "results"
KV_BYTES_PER_TOKEN = {"qwen2.5:7b": 57_344, "qwen2.5:14b": 196_608}  # cache K/V en f16 (couches × têtes K/V × dim × 2)


# --- Variantes du runner -----------------------------------------------------------------------------------

class LegacyRunner(LocalAgentRunner):
    """Réglage d'AVANT ce diagnostic : même fenêtre (TRACE_OLLAMA_NUM_CTX) pour tous les appels, génération non
    bornée, réponse d'un bloc (sans flux). Seuls les paramètres d'exécution diffèrent ; prompts, schémas et
    validateurs sont identiques."""

    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        self._ctx_floor = settings.num_ctx  # fenêtre fixe : celle de TRACE_OLLAMA_NUM_CTX pour chaque appel

    async def _chat(self, messages, output_schema, prepare, on_progress=None):
        async with self._semaphore().slot(messages[0]["content"]):
            prepare()
            payload = {"model": self.settings.model, "messages": messages, "format": output_schema,
                       "stream": False,
                       "options": {"temperature": self.settings.temperature, "num_ctx": self.settings.num_ctx}}
            return await asyncio.to_thread(self.transport.chat, payload)


class DryTransport:
    """Faux Ollama du mode `plan` : renvoie pour chaque agent l'objet minimal conforme à son schéma (aucun modèle)."""

    def __init__(self, model: str):
        self.model = model

    def version(self) -> str:
        return "plan"

    def models(self) -> list[str]:
        return [self.model]

    def chat(self, payload: dict, on_progress=None) -> dict:
        text = json.dumps(minimal_instance(payload["format"], payload["format"]), ensure_ascii=False)
        return {"model": self.model, "message": {"role": "assistant", "content": text}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 0, "eval_count": 0}


def minimal_instance(schema: dict, root: dict):
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[1]
        return minimal_instance(root.get("$defs", {})[name], root)
    if "anyOf" in schema:
        options = schema["anyOf"]
        return None if {"type": "null"} in options else minimal_instance(options[0], root)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {key: minimal_instance(schema["properties"][key], root) for key in schema.get("required", [])}
    if kind == "array":
        return [minimal_instance(schema["items"], root) for _ in range(schema.get("minItems", 0))]
    return {"string": "", "integer": 0, "number": 0, "boolean": False, "null": None}.get(kind)


# --- Exécution -----------------------------------------------------------------------------------------------

def _workspace(keep: bool) -> Path:
    root = Path(tempfile.mkdtemp(prefix="trace_benchmark_"))
    config.CACHE_DIR = root / "cache"
    config.INPUTS_DIR = root / "inputs"
    config.OUTPUTS_DIR = root / "outputs"
    config.CROSS_INTERVIEW_DIR = root / "cross_interview"
    return root


def _ingest(path: Path) -> dict:
    run = init_run([(path.name, path.read_bytes())], "", config.INPUTS_DIR, config.OUTPUTS_DIR)
    run = ingest_run(run)
    info = run["files"][0]["ingestion"]
    if info["status"] == "FAIL":
        raise SystemExit(f"Ingestion impossible : {info}")
    return run


def run_stage3(run: dict, settings: LLMSettings, runner_class, transport=None, checker=True) -> tuple[dict, list]:
    ids = [f["ingestion"]["interview_id"] for f in run["files"] if "ingestion" in f]
    method = MethodChecker({f["ingestion"]["interview_id"]: f["ingestion"]["output_dir"] for f in run["files"]
                            if "ingestion" in f}) if checker else None
    runner = runner_class(settings, transport=transport, checker=method, specs=AGENT_SPECS,
                          journal_dir=Path(run["output_dir"]) / "local_runs" / "stage3")
    started = time.monotonic()
    result = lp.run_stage("3", run, ids, settings=settings, runner=runner)
    result["status"]["wall_seconds"] = round(time.monotonic() - started, 1)
    records = sorted(runner.records.values(), key=lambda r: r["started_at"] or "")
    return result, records


def _fmt(value, digits=0) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and digits:
        return f"{value:.{digits}f}"
    return f"{int(round(value)):,}".replace(",", " ")


def print_table(records: list[dict], plan: bool = False) -> None:
    if plan:
        print("| Agent | Appel | caractères envoyés | tokens estimés (entrée) | contexte | réponse max |")
        print("|---|---|---:|---:|---:|---:|")
        for r in records:
            print(f"| {r['agent_label']} | {r['label'].split('/', 1)[1]} | {_fmt(r['prompt_chars'])} | "
                  f"{_fmt(r['estimated_input_tokens'])} | {_fmt(r['num_ctx'])} | {_fmt(r['num_predict'])} |")
        return
    print("| Agent | Appel | tokens entrée | tokens sortie | durée (s) | tentatives | contexte | tok/s | corrections |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in records:
        print(f"| {r['agent_label']} | {r['label'].split('/', 1)[1]} | {_fmt(r['input_tokens'])} | "
              f"{_fmt(r['output_tokens'])} | {_fmt(r['duration_seconds'], 1)} | {r['attempts']} | "
              f"{_fmt(r['num_ctx'])} | {_fmt(r['tokens_per_second'], 1)} | "
              f"{', '.join(r['correction_reasons']) or '—'} |")


def totals(records: list[dict], status: dict) -> dict:
    return {"calls": len(records), "attempts": sum(r["attempts"] for r in records),
            "corrected_calls": sum(r["attempts"] > 1 for r in records),
            "input_tokens": sum(r["input_tokens"] or 0 for r in records),
            "output_tokens": sum(r["output_tokens"] or 0 for r in records),
            "generation_seconds": round(sum(r["generation_seconds"] or 0 for r in records), 1),
            "load_seconds": round(sum(r["load_seconds"] or 0 for r in records), 1),
            "stage3_seconds": status["wall_seconds"], "stage3_status": status["status"],
            "max_num_ctx": max((r["num_ctx"] or 0 for r in records), default=0)}


def compare_format(first: dict, settings: LLMSettings) -> dict:
    """Un même bloc : JSON Schema strict (TRACE) puis mode JSON simple. Diagnostic : la seconde réponse est
    seulement validée contre le schéma (Pydantic), jamais utilisée."""
    transport = OllamaTransport(settings.ollama_url, settings.timeout_seconds)
    messages = [{"role": "system", "content": first["system"]}, {"role": "user", "content": first["user"]}]
    prompt = estimate_prompt_tokens(messages)
    num_ctx, num_predict = context_plan(prompt, 8192, settings.num_ctx)
    out = {}
    for name, fmt in (("schema", first["schema"]), ("json", "json")):
        payload = {"model": settings.model, "messages": messages, "format": fmt, "stream": False,
                   "keep_alive": settings.keep_alive,
                   "options": {"temperature": settings.temperature, "num_ctx": num_ctx, "num_predict": num_predict}}
        started = time.monotonic()
        reply = transport.chat(payload)
        text = (reply.get("message") or {}).get("content") or ""
        try:
            parse_json_output(text, first["model"])
            valid = True
        except LLMError as error:
            valid = f"{error.code}"
        out[name] = {"seconds": round(time.monotonic() - started, 1), "eval_count": reply.get("eval_count"),
                     "prompt_eval_count": reply.get("prompt_eval_count"),
                     "tokens_per_second": round(reply["eval_count"] / (reply["eval_duration"] / 1e9), 1)
                     if reply.get("eval_count") and reply.get("eval_duration") else None,
                     "response_chars": len(text), "schema_valid": valid, "done_reason": reply.get("done_reason")}
    return out


class CapturingRunner(LocalAgentRunner):
    first: dict | None = None

    async def complete_json(self, **kwargs):
        if CapturingRunner.first is None and "practice" in kwargs.get("label", ""):
            CapturingRunner.first = {"system": kwargs["system_prompt"], "user": kwargs["user_content"],
                                     "schema": kwargs["output_schema"], "model": kwargs["response_model"]}
        return await super().complete_json(**kwargs)


def kv_cache_gb(model: str, num_ctx: int) -> float | None:
    per_token = KV_BYTES_PER_TOKEN.get(model)
    return round(per_token * num_ctx / 1024 ** 3, 2) if per_token else None


def cmd_plan(args) -> int:
    settings = LLMSettings.from_env()
    root = _workspace(args.keep)
    try:
        run = _ingest(Path(args.file))
        info = run["files"][0]["ingestion"]
        _, records = run_stage3(run, settings, LocalAgentRunner, transport=DryTransport(settings.model), checker=False)
        print(f"Entretien : {Path(args.file).name} — {info['turn_count']} tours, "
              f"{len(Path(args.file).read_text(encoding='utf-8', errors='replace')):,} caractères".replace(",", " "))
        print(f"Modèle : {settings.model} · fenêtre maximale : {settings.num_ctx} (TRACE_OLLAMA_NUM_CTX)\n")
        print_table(records, plan=True)
        print(f"\nAppels prévus (hors corrections) : {len(records)}")
        print("Remarque : avec des réponses vides, la lecture à longue distance peut ne pas être déclenchée ; avec "
              "le vrai modèle elle ajoute au plus un appel.")
        windows = sorted({r["num_ctx"] for r in records})
        print(f"Fenêtres utilisées : {windows} (avant : {settings.num_ctx} pour chaque appel)")
        for window in (*windows, settings.num_ctx):
            size = kv_cache_gb(settings.model, window)
            if size is not None:
                print(f"  cache K/V {settings.model} à {window} tokens ≈ {size} Go (en plus des poids du modèle)")
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    return 0


def cmd_run(args) -> int:
    settings = LLMSettings.from_env()
    if args.legacy:
        settings = LLMSettings(**{**settings.__dict__, "max_method_corrections": settings.max_corrections})
    root = _workspace(args.keep)
    try:
        lp.make_runner(settings).require_ready()
    except LLMError as error:
        print(f"Runtime local indisponible : {error.user_message}", file=sys.stderr)
        return 7
    try:
        run = _ingest(Path(args.file))
        info = run["files"][0]["ingestion"]
        label = "AVANT (réglage d'origine)" if args.legacy else "APRÈS (contexte par appel, génération bornée)"
        print(f"Benchmark étape 3 — {label}")
        print(f"Entretien : {Path(args.file).name} — {info['turn_count']} tours · modèle {settings.model} · "
              f"Ollama {settings.ollama_url}\n")
        runner_class = LegacyRunner if args.legacy else CapturingRunner
        result, records = run_stage3(run, settings, runner_class)
        print_table(records)
        summary = totals(records, result["status"])
        print(f"\nDurée totale de l'étape 3 : {summary['stage3_seconds']} s ({summary['stage3_seconds'] / 60:.1f} min) "
              f"— statut {summary['stage3_status']}")
        print(f"Appels : {summary['calls']} · tentatives : {summary['attempts']} · appels corrigés : "
              f"{summary['corrected_calls']} · tokens générés : {summary['output_tokens']} · génération : "
              f"{summary['generation_seconds']} s · chargements du modèle : {summary['load_seconds']} s")
        n = args.interviews
        estimate = summary["stage3_seconds"] * n
        print(f"Estimation étape 3 pour {n} entretiens de cette longueur : {estimate / 3600:.1f} h "
              f"({estimate / 60:.0f} min), séquentiellement, modèle chargé en continu.")
        report = {"label": label, "legacy": args.legacy, "file": str(args.file), "turns": info["turn_count"],
                  "model": settings.model, "num_ctx_max": settings.num_ctx, "created_at": datetime.now().isoformat(),
                  "summary": summary, "estimate_seconds": estimate, "interviews": n, "calls": records}
        if args.compare_format and CapturingRunner.first:
            print("\nCoût du JSON Schema strict (même bloc du Practice Extractor, diagnostic) :")
            comparison = compare_format(CapturingRunner.first, settings)
            for name, values in comparison.items():
                print(f"  format={name:6s} : {values['seconds']} s, {values['eval_count']} tokens générés, "
                      f"{values['tokens_per_second']} tok/s, conforme au schéma : {values['schema_valid']}")
            report["format_comparison"] = comparison
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"benchmark_{'avant' if args.legacy else 'apres'}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nRapport complet : {out}")
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark de l'étape 3 avec le modèle local (Ollama).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("plan", cmd_plan), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--file", default=str(DEFAULT_FILE))
        p.add_argument("--keep", action="store_true", help="garder le dossier temporaire (run, journaux)")
        p.set_defaults(func=func)
        if name == "run":
            p.add_argument("--legacy", action="store_true", help="réglage d'origine (mesure « avant »)")
            p.add_argument("--compare-format", action="store_true", help="schéma strict contre mode JSON (diagnostic)")
            p.add_argument("--interviews", type=int, default=17)
    args = parser.parse_args(argv)
    config.load_env_file()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
