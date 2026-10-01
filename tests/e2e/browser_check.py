"""Test navigateur de bout en bout (Playwright + Chromium), SANS appel API réel.

    python tests/e2e/browser_check.py [--screenshots DOSSIER] [--chromium CHEMIN]

Deux scénarios, chacun avec un serveur Streamlit local :
A. app.py sans clé API : l'ingestion fonctionne, l'analyse IA est désactivée
   avec un message explicite ;
B. tests/e2e/fake_llm_app.py (LLM simulé) : import de l'entretien synthétique,
   ingestion, puis étape 3 en mode test (2 appels simulés), statuts, comptes,
   citation inventée détectée, second lancement repris du cache, mode corpus
   soumis à confirmation ;
C. étape 3.5 (LLM simulé) : entretien synthétique dont un tour est mal attribué,
   1 appel d'audit + 2 agents, avertissement affiché, transcription inchangée ;
D. étape 3.6 (LLM simulé) : entretien long synthétique (349 tours), annonce des blocs
   avant exécution, 5 blocs + 1 lecture à longue distance, bilan des blocs,
   téléchargement de interaction_signals.json fusionné (citations valides).

Non collecté par pytest (nécessite Playwright et un navigateur). Les runs
créés dans data/ pendant le test sont supprimés à la fin.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import json

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests import synthetic_interviews as si  # noqa: E402
from tests import synthetic_long_interview as long_interview  # noqa: E402

TIMEOUT_MS = 30_000


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def streamlit_server(script: Path, env: dict):
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(script), "--server.headless", "true",
         "--server.port", str(port), "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false",
         "--server.fileWatcherType", "none"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost : pas de proxy
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if opener.open(f"{url}/_stcore/health", timeout=1).read() == b"ok":
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Le serveur Streamlit n'a pas démarré.")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def upload_and_ingest(page, interview_path: Path) -> None:
    page.locator("input[type=file]").set_input_files(str(interview_path))
    expect(page.get_by_text(interview_path.name).first).to_be_visible(timeout=TIMEOUT_MS)
    page.get_by_role("button", name="Lancer l'analyse", exact=True).click()
    expect(page.get_by_text("Structuration des entretiens terminée")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_role("heading", name="Analyse IA — Étape 3")).to_be_visible(timeout=TIMEOUT_MS)


def scenario_without_key(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    expect(page.get_by_role("heading", name="TRACE", exact=True)).to_be_visible(timeout=TIMEOUT_MS)
    upload_and_ingest(page, interview)
    expect(page.get_by_text("Analyse IA désactivée").first).to_be_visible(timeout=TIMEOUT_MS)
    assert page.get_by_role("button", name="Lancer les deux analyses IA").count() == 0
    page.screenshot(path=str(shots / "A_sans_cle.png"), full_page=True)
    print("A. sans clé : ingestion OK, analyse IA désactivée avec message explicite")


def scenario_fake_llm(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    upload_and_ingest(page, interview)
    expect(page.get_by_text("2 appels LLM par entretien non présent dans le cache")).to_be_visible()
    launch = page.get_by_role("button", name="Lancer les deux analyses IA (2 appel(s) API payant(s))")
    expect(launch).to_be_enabled()
    page.screenshot(path=str(shots / "B1_avant_lancement.png"), full_page=True)

    launch.click()
    expect(page.get_by_text("Analyse IA terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("2 appel(s) API — 4 580 tokens entrée — 3 210 tokens sortie")).to_be_visible()
    table = page.locator("[data-testid=stTable]").last
    expect(table).to_contain_text("SUCCESS_WITH_WARNINGS")   # Practice Extractor : citation inventée
    expect(table).to_contain_text("✅ SUCCESS")              # Interaction Reader
    row = table.locator("tbody tr").first
    cells = [c.strip() for c in row.locator("td").all_inner_texts()]
    print("   ligne de résultats :", cells)
    assert cells[1:4] == ["✅ SUCCESS", "0", "0"], cells  # audit des locuteurs : aucun tour suspect, aucun appel
    assert cells[6:9] == ["6", "10", "1"], cells  # pratiques, signaux, citations invalides
    assert cells[-1] == "1", cells  # entretien court : un seul bloc (un appel Interaction Reader)
    page.get_by_text("Analyse IA — ENTRETIEN_SYNTHETIQUE — aperçu").click()
    expect(page.get_by_text("Anomalies de validation")).to_be_visible()
    for name in ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json"):
        expect(page.get_by_role("button", name=f"Télécharger {name}")).to_be_visible()
    expect(page.get_by_text("(IA) — terminé (1/1 entretien(s))").first).to_be_visible()
    assert_no_exception(page)
    page.screenshot(path=str(shots / "B2_resultats.png"), full_page=True)
    print("B. LLM simulé : 2 agents SUCCESS / SUCCESS_WITH_WARNINGS, citation inventée détectée, téléchargements OK")

    # Second lancement : tout vient du cache
    relaunch = page.get_by_role("button", name="Lancer les deux analyses IA (0 appel(s) API payant(s))")
    expect(relaunch).to_be_visible(timeout=TIMEOUT_MS)
    relaunch.click()
    expect(page.locator("[data-testid=stTable]").last).to_contain_text("CACHED", timeout=TIMEOUT_MS)
    expect(page.get_by_text("0 appel(s) API — 0 tokens entrée — 0 tokens sortie")).to_be_visible()
    print("   second lancement : CACHED, 0 appel API")

    # Mode corpus : confirmation obligatoire
    page.get_by_text("Corpus complet").click()
    corpus_button = page.get_by_role("button", name="Lancer les deux analyses IA (0 appel(s) API payant(s))")
    expect(corpus_button).to_be_disabled(timeout=TIMEOUT_MS)
    page.get_by_text("Je confirme lancer l'analyse IA sur tout le corpus.").click()
    expect(corpus_button).to_be_enabled(timeout=TIMEOUT_MS)
    page.screenshot(path=str(shots / "B3_mode_corpus.png"), full_page=True)
    print("   mode corpus : bouton désactivé tant que la case de confirmation n'est pas cochée")


def scenario_stage35(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    upload_and_ingest(page, interview)
    expect(page.get_by_text("1 tour(s) suspect(s), 1 appel(s) d'audit").first).to_be_visible(timeout=TIMEOUT_MS)
    launch = page.get_by_role("button", name="Lancer les deux analyses IA (au plus 3 appel(s) API payant(s))")
    expect(launch).to_be_enabled()
    launch.click()
    expect(page.get_by_text("Analyse IA terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("3 appel(s) API — 4 900 tokens entrée — 2 250 tokens sortie")).to_be_visible()
    title = f"Audit d'attribution des locuteurs — {si.STAGE35_INTERVIEW_ID} — 1 tour(s) suspect(s), 1 à vérifier"
    page.get_by_text(title).click()
    expect(page.get_by_text("Ces suggestions ne modifient pas la transcription originale.")).to_be_visible()
    expect(page.get_by_role("button", name="Télécharger speaker_attribution_audit.json")).to_be_visible()
    page.screenshot(path=str(shots / "C_etape_3_5_audit_locuteurs.png"), full_page=True)
    print("C. étape 3.5 : 1 appel d'audit + 2 agents, tour mal attribué signalé, transcription non modifiable")


def assert_no_exception(page) -> None:
    assert page.locator("[data-testid=stException]").count() == 0, "exception Streamlit affichée"


def scenario_long_interview(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    upload_and_ingest(page, interview)
    expect(page.get_by_text("entretien long : analyse en 5 blocs").first).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("soit au plus 6 appels Interaction Reader").first).to_be_visible()
    launch = page.get_by_role("button", name="Lancer les deux analyses IA (au plus 7 appel(s) API payant(s))")
    expect(launch).to_be_enabled()
    page.screenshot(path=str(shots / "D1_entretien_long_avant.png"), full_page=True)
    launch.click()
    expect(page.get_by_text("Analyse IA terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("7 appel(s) API").first).to_be_visible()
    table = page.locator("[data-testid=stTable]").last
    row = [c.strip() for c in table.locator("tbody tr").first.locator("td").all_inner_texts()]
    print("   ligne de résultats :", row)
    assert row[5] == "✅ SUCCESS" and row[-1] == "5/5", row
    page.get_by_text(f"Analyse IA — {long_interview.LONG_INTERVIEW_ID} — aperçu").click()
    expect(page.get_by_text("blocs réussis : 5/5").first).to_be_visible()
    with page.expect_download() as info:
        page.get_by_role("button", name="Télécharger interaction_signals.json").click()
    document = json.loads(Path(info.value.path()).read_text(encoding="utf-8"))
    chunking = document["chunking"]
    assert document["status"] == "SUCCESS" and document["analysis_complete"] is True
    assert chunking["chunking_used"] and chunking["chunk_count"] == 5 and chunking["chunks_succeeded"] == 5
    assert chunking["signals_after_dedup"] == len(document["signals"]) < chunking["signals_before_dedup"]
    assert all(e["validation"]["valid"] for s in document["signals"] for e in s["evidence"])
    assert [s["signal_type"] for s in document["signals"]].count("cross_turn_contradiction") == 1
    assert_no_exception(page)
    page.screenshot(path=str(shots / "D2_entretien_long_resultats.png"), full_page=True)
    print(f"D. entretien long : 5 blocs + 1 lecture à longue distance, {len(document['signals'])} signaux "
          f"(avant dédoublonnage : {chunking['signals_before_dedup']}), téléchargement OK")

    relaunch = page.get_by_role("button", name="Lancer les deux analyses IA (0 appel(s) API payant(s))")
    expect(relaunch).to_be_visible(timeout=TIMEOUT_MS)
    relaunch.click()
    expect(page.get_by_text("0 appel(s) API — 0 tokens entrée — 0 tokens sortie")).to_be_visible(timeout=TIMEOUT_MS)
    assert_no_exception(page)
    print("   second lancement : 0 appel API (blocs et lecture à longue distance en cache)")


def run_dirs() -> set[Path]:
    return {p for base in (ROOT / "data" / "inputs", ROOT / "data" / "outputs") for p in base.glob("run_*")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshots", type=Path, default=Path(tempfile.mkdtemp(prefix="trace_e2e_")))
    parser.add_argument("--chromium", default=os.environ.get("TRACE_CHROMIUM", "/opt/pw-browsers/chromium"))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="trace_e2e_data_"))
    interview = work / si.FILENAME
    interview.write_text(si.TEXT, encoding="utf-8")
    stage35 = work / si.STAGE35_FILES[0][0]
    stage35.write_bytes(si.STAGE35_FILES[0][1])
    long_file = work / long_interview.LONG_FILENAME
    long_file.write_text(long_interview.long_text(), encoding="utf-8")
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "TRACE_"))}
    before = run_dirs()
    try:
        with sync_playwright() as pw:
            executable = args.chromium if Path(args.chromium).exists() else None
            browser = pw.chromium.launch(executable_path=executable, args=["--no-proxy-server"])
            with streamlit_server(ROOT / "app.py", base_env) as url:
                scenario_without_key(browser, url, interview, args.screenshots)
            env = {**base_env, "TRACE_E2E_CACHE_DIR": str(work / "cache")}
            with streamlit_server(ROOT / "tests" / "e2e" / "fake_llm_app.py", env) as url:
                scenario_fake_llm(browser, url, interview, args.screenshots)
                scenario_stage35(browser, url, stage35, args.screenshots)
                scenario_long_interview(browser, url, long_file, args.screenshots)
            browser.close()
    finally:
        for path in run_dirs() - before:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    print(f"Test navigateur réussi. Captures : {args.screenshots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
