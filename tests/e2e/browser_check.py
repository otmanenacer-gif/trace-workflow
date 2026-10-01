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
   téléchargement de interaction_signals.json fusionné (citations valides) ;
E. étape 3.7 (LLM simulé) : entretien long synthétique saturé de remplisseurs (330 tours),
   Practice Extractor en 4 blocs et Interaction Reader en 3 blocs annoncés avant exécution,
   lecteur Interaction qui surcode, bilan (blocs, pratiques/signaux avant et après
   dédoublonnage, anomalies de validation), téléchargements JSON, relance depuis le cache ;
F. étape 4 (LLM simulé) : entretien synthétique de référence, étape 3 puis épisodes d'accountability
   (6 candidats annoncés, 1 appel), tableau accountability / ordinaires / incertains, aperçu,
   téléchargements JSON vérifiés, relance depuis le cache (0 appel) ;
G. étape 4 sur l'entretien long synthétique (320 tours) : 8 candidats pour 31 pratiques, 1 appel,
   5 épisodes, 3 pratiques ordinaires examinées, 20 sans marqueur ;
H. restauration (serveur neuf, cache vide, comme après un redéploiement) : l'entretien de l'étape 4 est
   réimporté, ses 4 JSON d'étape 3 (calculés au préalable par le LLM simulé) sont importés dans
   « Restaurer des résultats Stage 3 existants », puis l'étape 4 est lancée : 0 appel d'étape 3, 1 appel
   d'étape 4, résultats identiques ;
I. étape 4.1 : entretien de régression « OTMANE » synthétique (388 tours, longs tours à plusieurs pratiques) :
   26 candidats, représentation normalisée, 1 appel, 17 épisodes / 8 ordinaires examinées / 22 sans marqueur,
   aucune fusion de composantes déconnectées, aucun avertissement ;
J. étape 5 après l'étape 4 (entretien de référence, sans temporalité) : 1 appel, configuration contextuelle,
   aperçu, 3 téléchargements JSON vérifiés, relance depuis le cache (0 appel) ;
K. étape 5 sur un entretien à vraie temporalité (« Au lycée » / « Maintenant ») : trajectoire temporelle explicite ;
L. restauration des étapes 3 PUIS 4 (serveur neuf, cache vide) : « Stage 4 restauré depuis fichiers — 0 appel API »,
   puis étape 5 directement : 1 appel, aucune exécution des étapes 3 et 4 ;
M. étape 5 sur l'entretien long « OTMANE-like » (190 tours, 50 pratiques, 20 épisodes) : 1 appel, configuration mixte ;
N. étape 6, sans run : import de 2 triplets d'étape 5 → comparaison EXPLORATOIRE (1 appel) ;
O. étape 6 sur le corpus synthétique de 17 entretiens + un entretien invalide + un fichier illisible : tableau du corpus
   (18 entretiens, 17 exploitables, raison d'exclusion), 1 appel, cas négatifs et éléments à revoir visibles,
   3 téléchargements JSON vérifiés ;
P. relance de l'étape 6 sur le même corpus : 0 appel (cache).

`--only-stage6` n'exécute que les scénarios N à P.

Non collecté par pytest (nécessite Playwright et un navigateur). Les runs
créés dans data/ pendant le test sont supprimés à la fin.
"""

from __future__ import annotations

import argparse
import os
import re
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
from tests import synthetic_stage37 as stage37  # noqa: E402
from tests import synthetic_stage4 as stage4  # noqa: E402
from tests import synthetic_stage4_long as stage4_long  # noqa: E402
from tests import synthetic_stage4_otmane as otmane  # noqa: E402
from tests import synthetic_stage5 as stage5  # noqa: E402
from tests import synthetic_stage5_long as stage5_long  # noqa: E402
from tests import synthetic_stage6 as stage6  # noqa: E402

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


def file_input(page, label: str):
    """Champ d'import désigné par son libellé (la page en compte plusieurs : corpus, restaurations, étape 6)."""
    return page.get_by_test_id("stFileUploader").filter(has_text=label).locator("input[type=file]")


def upload_and_ingest(page, interview_path: Path) -> None:
    file_input(page, "Importer des entretiens").set_input_files(str(interview_path))
    expect(page.get_by_text(interview_path.name).first).to_be_visible(timeout=TIMEOUT_MS)
    wait_idle(page)  # l'import relance le script : cliquer seulement une fois le rendu terminé
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


def wait_idle(page) -> None:
    """Attend que Streamlit ait fini d'exécuter le script (indicateur « Running… » absent)."""
    page.wait_for_timeout(500)
    expect(page.locator("[data-testid=stStatusWidget]")).to_have_count(0, timeout=TIMEOUT_MS)
    page.wait_for_timeout(300)


def download_json(page, expander_label: str, name: str) -> dict:
    """Télécharge un JSON de l'aperçu d'un entretien. Un téléchargement relance le script et l'aperçu peut se
    refermer : attendre la fin du rendu, rouvrir l'aperçu au besoin, puis cliquer."""
    wait_idle(page)
    button = page.get_by_role("button", name=f"Télécharger {name}")
    if not button.is_visible():
        page.get_by_text(expander_label).click()
        wait_idle(page)
    with page.expect_download(timeout=TIMEOUT_MS) as info:
        button.click()
    return json.loads(Path(info.value.path()).read_text(encoding="utf-8"))


def assert_no_exception(page) -> None:
    assert page.locator("[data-testid=stException]").count() == 0, "exception Streamlit affichée"


def scenario_long_interview(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    upload_and_ingest(page, interview)
    expect(page.get_by_text("entretien long : analyse en 5 blocs").first).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("soit au plus 6 appels Interaction Reader").first).to_be_visible()
    # étape 3.7 : le Practice Extractor lit aussi l'entretien long par blocs (6 blocs)
    expect(page.get_by_text("soit au plus 6 appels Practice Extractor").first).to_be_visible()
    launch = page.get_by_role("button", name="Lancer les deux analyses IA (au plus 12 appel(s) API payant(s))")
    expect(launch).to_be_enabled()
    page.screenshot(path=str(shots / "D1_entretien_long_avant.png"), full_page=True)
    launch.click()
    expect(page.get_by_text("Analyse IA terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("12 appel(s) API").first).to_be_visible()
    wait_idle(page)  # la relance qui suit l'analyse remplace le tableau d'avancement par celui des résultats
    table = page.locator("[data-testid=stTable]").last
    row = [c.strip() for c in table.locator("tbody tr").first.locator("td").all_inner_texts()]
    print("   ligne de résultats :", row)
    assert row[5] == "✅ SUCCESS" and row[-1] == "5/5", row
    page.get_by_text(f"Analyse IA — {long_interview.LONG_INTERVIEW_ID} — aperçu").click()
    expect(page.get_by_text("blocs réussis : 5/5").first).to_be_visible()
    document = download_json(page, f"Analyse IA — {long_interview.LONG_INTERVIEW_ID} — aperçu", "interaction_signals.json")
    chunking = document["chunking"]
    assert document["status"] == "SUCCESS" and document["analysis_complete"] is True
    assert chunking["chunking_used"] and chunking["chunk_count"] == 5 and chunking["chunks_succeeded"] == 5
    assert chunking["signals_after_dedup"] < chunking["signals_before_dedup"]
    # étape 3.7 : les « Euh… » isolés du lecteur simulé sont écartés après la fusion, et conservés à part
    assert len(document["signals"]) + len(document["set_aside_signals"]) == chunking["signals_after_dedup"]
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


def scenario_stage37(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    upload_and_ingest(page, interview)
    expect(page.get_by_text("Practice Extractor — ENTRETIEN_ETAPE_3_7 : entretien long : analyse en 4 blocs").first
           ).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("Interaction Reader — ENTRETIEN_ETAPE_3_7 : entretien long : analyse en 3 blocs").first
           ).to_be_visible()
    launch = page.get_by_role("button", name="Lancer les deux analyses IA (au plus 9 appel(s) API payant(s))")
    expect(launch).to_be_enabled()
    page.screenshot(path=str(shots / "E1_etape_3_7_avant.png"), full_page=True)
    launch.click()
    expect(page.get_by_text("Analyse IA terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("9 appel(s) API").first).to_be_visible()
    wait_idle(page)
    table = page.locator("[data-testid=stTable]").last
    row = [c.strip() for c in table.locator("tbody tr").first.locator("td").all_inner_texts()]
    print("   ligne de résultats :", row)
    assert row[4] == "✅ SUCCESS" and row[5] == "✅ SUCCESS", row        # Practice, Interaction
    assert row[-3:] == ["0", "4/4", "3/3"], row                           # anomalies, blocs Practice, blocs Interaction
    page.get_by_text("Analyse IA — ENTRETIEN_ETAPE_3_7 — aperçu").click()
    expect(page.get_by_text("pratiques finales : 19 (avant dédoublonnage : 20)").first).to_be_visible()
    expect(page.get_by_text("anomalies de validation : 0").first).to_be_visible()
    documents = {name: download_json(page, "Analyse IA — ENTRETIEN_ETAPE_3_7 — aperçu", name)
                 for name in ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json")}
    practices, signals = documents["practice_extractor.json"], documents["interaction_signals.json"]
    assert practices["status"] == "SUCCESS" and practices["chunking"]["chunk_count"] == 4
    assert {"use", "non_use", "refusal", "past_use"} <= {p["use_status"] for p in practices["practices"]}
    markers = stage37.micro_marker_count()
    assert signals["chunking"]["signals_before_dedup"] > markers and len(signals["signals"]) * 20 < markers
    assert [s["signal_type"] for s in signals["signals"]].count("cross_turn_contradiction") == 2
    assert all(e["validation"]["valid"] for d in (practices, signals) for i in d[next(
        k for k in ("practices", "signals") if k in d)] for e in i["evidence"])
    assert documents["evidence_validation.json"]["total_invalid_evidence"] == 0
    assert_no_exception(page)
    page.screenshot(path=str(shots / "E2_etape_3_7_resultats.png"), full_page=True)
    print(f"E. étape 3.7 : Practice 4/4 blocs ({len(practices['practices'])} pratiques), Interaction 3/3 blocs "
          f"({len(signals['signals'])} signaux pour {markers} remplisseurs ; avant dédoublonnage : "
          f"{signals['chunking']['signals_before_dedup']}), 0 anomalie, téléchargements OK")

    relaunch = page.get_by_role("button", name="Lancer les deux analyses IA (0 appel(s) API payant(s))")
    expect(relaunch).to_be_visible(timeout=TIMEOUT_MS)
    relaunch.click()
    expect(page.get_by_text("0 appel(s) API — 0 tokens entrée — 0 tokens sortie")).to_be_visible(timeout=TIMEOUT_MS)
    assert_no_exception(page)
    print("   second lancement : 0 appel API (tous les blocs en cache)")


def stage4_row(page) -> list[str]:
    wait_idle(page)
    table = page.locator("[data-testid=stTable]").last
    return [c.strip() for c in table.locator("tbody tr").first.locator("td").all_inner_texts()]


def run_stage3_then_stage4(page, interview: Path, stage3_label: str, candidates: int) -> None:
    upload_and_ingest(page, interview)
    page.get_by_role("button", name=stage3_label).click()
    expect(page.get_by_text("Analyse IA terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    wait_idle(page)
    expect(page.get_by_role("heading", name="Étape 4 — Épisodes d'accountability")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text(f"Candidats d'épisodes (déterministes, sans IA) : {candidates}").first).to_be_visible()
    launch = page.get_by_role("button", name="Construire les épisodes d'accountability (1 appel(s) API payant(s))")
    expect(launch).to_be_enabled()
    launch.click()
    expect(page.get_by_text("Étape 4 terminée.")).to_be_visible(timeout=TIMEOUT_MS)


def scenario_stage4(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    run_stage3_then_stage4(page, interview, "Lancer les deux analyses IA (au plus 3 appel(s) API payant(s))", 6)
    expect(page.get_by_text("Dernière exécution (étape 4) : 1 appel(s) API").first).to_be_visible()
    row = stage4_row(page)
    print("   ligne étape 4 :", row)
    # Entretien, statut étape 3, étape 4, candidats, accountability, ordinaires examinées, sans marqueur, incertains,
    # rejetés, avertissements, appels
    assert row == [stage4.INTERVIEW_ID, "COMPLETE", "✅ SUCCESS", "6", "4", "1", "4", "1", "0", "0", "1"], row
    label = f"Épisodes d'accountability — {stage4.INTERVIEW_ID} — aperçu"
    page.get_by_text(label).click()
    expect(page.get_by_text("Épisodes (6) — 3 premiers")).to_be_visible(timeout=TIMEOUT_MS)
    page.screenshot(path=str(shots / "F1_etape_4_resultats.png"), full_page=True)
    episodes = download_json(page, label, "accountability_episodes.json")
    validation = download_json(page, label, "accountability_episode_validation.json")
    statuses = sorted(e["episode_status"] for e in episodes["episodes"])
    assert statuses == ["accountability_episode"] * 4 + ["ordinary_practice", "uncertain"], statuses
    assert episodes["analysis_complete"] is True and episodes["unmarked_practice_count"] == 4
    assert all(x["validation"]["valid"] for e in episodes["episodes"] for x in e["evidence"])
    [uncertain] = [e for e in episodes["episodes"] if e["episode_status"] == "uncertain"]
    assert uncertain["speaker_warnings"] and uncertain["needs_review"] is True
    assert validation["status"] == "SUCCESS" and validation["error_count"] == validation["warning_count"] == 0
    assert_no_exception(page)
    print("F. étape 4 : 6 candidats, 1 appel, 4 épisodes / 1 ordinaire examinée + 4 sans marqueur / 1 incertain, "
          "téléchargements JSON OK")

    relaunch = page.get_by_role("button", name="Construire les épisodes d'accountability (0 appel(s) API payant(s))")
    expect(relaunch).to_be_visible(timeout=TIMEOUT_MS)
    relaunch.click()
    expect(page.get_by_text("Dernière exécution (étape 4) : 0 appel(s) API").first).to_be_visible(timeout=TIMEOUT_MS)
    row = stage4_row(page)
    assert row[2] == "♻️ CACHED" and row[-1] == "0", row
    assert_no_exception(page)
    page.screenshot(path=str(shots / "F2_etape_4_cache.png"), full_page=True)
    print("   relance étape 4 : CACHED, 0 appel API")
    stage5_after_stage4(page, shots)


def scenario_stage4_long(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    run_stage3_then_stage4(page, interview, "Lancer les deux analyses IA (au plus 7 appel(s) API payant(s))", 8)
    row = stage4_row(page)
    print("   ligne étape 4 :", row)
    assert row == [stage4_long.INTERVIEW_ID, "COMPLETE", "✅ SUCCESS", "8", "5", "3", "20", "0", "0", "0", "1"], row
    assert_no_exception(page)
    page.screenshot(path=str(shots / "G_etape_4_long.png"), full_page=True)
    print("G. étape 4, entretien long : 31 pratiques → 8 candidats, 1 appel, 5 épisodes, 3 ordinaires examinées, "
          "20 sans marqueur")


def stage3_downloads(work: Path) -> list[Path]:
    """Calcule l'étape 3 de l'entretien de référence (LLM simulé, hors data/) et écrit les 4 JSON
    « téléchargés », nommés comme les boutons de téléchargement de l'interface."""
    from core.analysis import analyze_run
    from core.analysis_cache import AnalysisCache
    from tests.fake_llm import FakeTransport, fake_settings
    run = si.make_ingested_run(work / "precomputed", stage4.FILES)
    analyze_run(run, settings=fake_settings(), transport=FakeTransport(stage4.stage3_responders()),
                cache=AnalysisCache(work / "precomputed_cache"))
    out = Path(run["files"][0]["ingestion"]["output_dir"]) / "analysis"
    paths = []
    for name in ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json",
                 "speaker_attribution_audit.json"):
        target = work / "downloads" / f"{stage4.INTERVIEW_ID}_{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((out / name).read_bytes())
        paths.append(target)
    return paths


def scenario_restore(browser, url: str, interview: Path, downloads: list[Path], shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    upload_and_ingest(page, interview)
    expect(page.get_by_text("Restaurer des résultats Stage 3 existants")).to_be_visible(timeout=TIMEOUT_MS)
    restore = page.get_by_role("button", name="Restaurer l'étape 3 depuis ces fichiers (0 appel API)")
    expect(restore).to_be_disabled()
    restore_input = file_input(page, "Les 4 fichiers JSON de l'étape 3")
    expect(restore_input).to_have_count(1, timeout=TIMEOUT_MS)
    restore_input.set_input_files([str(p) for p in downloads])  # noms affichés tronqués : on attend le bouton
    wait_idle(page)
    expect(restore).to_be_enabled(timeout=TIMEOUT_MS)
    page.screenshot(path=str(shots / "H1_restauration_avant.png"), full_page=True)
    restore.click()
    expect(page.get_by_text("Stage 3 restauré depuis fichiers — 0 appel API").first).to_be_visible(timeout=TIMEOUT_MS)
    wait_idle(page)
    assert page.get_by_text("Dernière exécution :", exact=False).count() == 0  # aucune exécution de l'étape 3
    expect(page.get_by_text("Candidats d'épisodes (déterministes, sans IA) : 6").first).to_be_visible()
    launch = page.get_by_role("button", name="Construire les épisodes d'accountability (1 appel(s) API payant(s))")
    expect(launch).to_be_enabled()
    launch.click()
    expect(page.get_by_text("Étape 4 terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("Dernière exécution (étape 4) : 1 appel(s) API").first).to_be_visible()
    row = stage4_row(page)
    print("   ligne étape 4 :", row)
    assert row == [stage4.INTERVIEW_ID, "COMPLETE", "✅ SUCCESS", "6", "4", "1", "4", "1", "0", "0", "1"], row
    stage3_table = page.locator("[data-testid=stTable]").filter(has_text="Practice Extractor")
    expect(stage3_table).to_contain_text("SUCCESS")  # sorties restaurées affichées comme une étape 3 normale
    assert_no_exception(page)
    page.screenshot(path=str(shots / "H2_restauration_etape_4.png"), full_page=True)
    print("H. restauration : 4 JSON importés, « Stage 3 restauré depuis fichiers — 0 appel API », "
          "0 appel d'étape 3, étape 4 : 1 appel, 4 épisodes / 1 ordinaire + 4 sans marqueur / 1 incertain")


def scenario_otmane(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    run_stage3_then_stage4(page, interview, "Lancer les deux analyses IA (au plus 8 appel(s) API payant(s))", 26)
    row = stage4_row(page)
    print("   ligne étape 4 :", row)
    assert row == [otmane.INTERVIEW_ID, "COMPLETE", "✅ SUCCESS", "26", "17", "8", "22", "0", "0", "0", "1"], row
    label = f"Épisodes d'accountability — {otmane.INTERVIEW_ID} — aperçu"
    page.get_by_text(label).click()
    episodes = download_json(page, label, "accountability_episodes.json")
    assert episodes["payload_format"] == "2" and episodes["usable_episode_count"] == episodes["episode_count"] == 25
    assert not any("DISCONNECTED_MERGE" in e["review_reasons"] for e in episodes["episodes"])
    assert "coupable" not in json.dumps(episodes["episodes"], ensure_ascii=False)
    assert_no_exception(page)
    page.screenshot(path=str(shots / "I_etape_4_1_otmane.png"), full_page=True)
    print("I. étape 4.1 « OTMANE » synthétique : 26 candidats, 1 appel, 25 épisodes utilisables, aucune fusion "
          "déconnectée, mot de l'enquêteur non attribué")


# --- Étape 5 -------------------------------------------------------------------------------------------

STAGE5_LAUNCH = "Construire la configuration intra-entretien ({} appel(s) API payant(s))"


def stage5_row(page) -> list[str]:
    wait_idle(page)
    table = page.locator("[data-testid=stTable]").filter(has_text="Changements temporels explicites").last
    return [c.strip() for c in table.locator("tbody tr").first.locator("td").all_inner_texts()]


def launch_stage5(page) -> list[str]:
    expect(page.get_by_role("heading", name="Étape 5 — Configuration et trajectoire intra-entretien")).to_be_visible(
        timeout=TIMEOUT_MS)
    launch = page.get_by_role("button", name=STAGE5_LAUNCH.format(1))
    expect(launch).to_be_enabled(timeout=TIMEOUT_MS)
    launch.click()
    expect(page.get_by_text("Étape 5 terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("Dernière exécution (étape 5) : 1 appel(s) API").first).to_be_visible()
    assert_no_exception(page)
    return stage5_row(page)


def stage5_after_stage4(page, shots: Path) -> None:
    row = launch_stage5(page)
    print("   ligne étape 5 :", row)
    # Entretien, étape 4, étape 5, configuration, retenues, frontières, variations, temporels, exceptions, tensions,
    # zones ordinaires, critères, avertissements, appels
    assert row == [stage4.INTERVIEW_ID, "disponible", "✅ SUCCESS", "configuration contextuelle", "3", "0", "0", "0",
                   "1", "0", "1", "2", "0", "1"], row
    label = f"Configuration intra-entretien — {stage4.INTERVIEW_ID} — aperçu"
    page.get_by_text(label).click()
    expect(page.get_by_text("Affirmations retenues (3)")).to_be_visible(timeout=TIMEOUT_MS)
    page.screenshot(path=str(shots / "J1_etape_5_resultats.png"), full_page=True)
    document = download_json(page, label, "student_trajectory.json")
    validation = download_json(page, label, "student_trajectory_validation.json")
    manifest = download_json(page, label, "student_trajectory_manifest.json")
    assert document["configuration_type"] == "contextual_configuration" and document["explicit_temporal_changes"] == []
    assert len(document["exceptions"]) == len(document["ordinary_zones"]) == 1
    assert validation["error_count"] == validation["warning_count"] == 0
    assert manifest["api_calls"] == 1 and manifest["stage4_status"] == "COMPLETE"
    print("J. étape 5 après l'étape 4 : 1 appel, configuration contextuelle, 3 affirmations, 2 critères, "
          "téléchargements JSON OK")
    relaunch = page.get_by_role("button", name=STAGE5_LAUNCH.format(0))
    expect(relaunch).to_be_visible(timeout=TIMEOUT_MS)
    relaunch.click()
    expect(page.get_by_text("Dernière exécution (étape 5) : 0 appel(s) API").first).to_be_visible(timeout=TIMEOUT_MS)
    row = stage5_row(page)
    assert row[2] == "♻️ CACHED" and row[-1] == "0", row
    assert_no_exception(page)
    page.screenshot(path=str(shots / "J2_etape_5_cache.png"), full_page=True)
    print("   relance étape 5 : CACHED, 0 appel API")


def stage3_and_stage4(page, interview: Path) -> None:
    upload_and_ingest(page, interview)
    page.get_by_role("button", name=re.compile(r"^Lancer les deux analyses IA")).click()
    expect(page.get_by_text("Analyse IA terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    wait_idle(page)
    page.get_by_role("button", name="Construire les épisodes d'accountability (1 appel(s) API payant(s))").click()
    expect(page.get_by_text("Étape 4 terminée.")).to_be_visible(timeout=TIMEOUT_MS)
    wait_idle(page)


def scenario_stage5_temporal(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    stage3_and_stage4(page, interview)
    row = launch_stage5(page)
    print("   ligne étape 5 :", row)
    assert row[:4] == [stage5.TEMPORAL.interview_id, "disponible", "✅ SUCCESS", "trajectoire temporelle explicite"], row
    assert row[7] == "1", row  # un changement temporel explicite
    label = f"Configuration intra-entretien — {stage5.TEMPORAL.interview_id} — aperçu"
    page.get_by_text(label).click()
    expect(page.get_by_text("Affirmations retenues (1)")).to_be_visible(timeout=TIMEOUT_MS)
    document = download_json(page, label, "student_trajectory.json")
    [change] = document["explicit_temporal_changes"]
    assert document["configuration_type"] == "temporal_trajectory" and change.endswith("_TC001")
    [claim] = document["trajectory_claims"]
    assert [a["text"] for a in claim["validated_temporal_anchors"]] == ["Au lycée", "Maintenant"]
    assert_no_exception(page)
    page.screenshot(path=str(shots / "K_etape_5_temporalite.png"), full_page=True)
    print("K. étape 5, vraie temporalité : trajectoire temporelle explicite, ancrages « Au lycée » / « Maintenant »")


def scenario_stage5_long(browser, url: str, interview: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    stage3_and_stage4(page, interview)
    row = launch_stage5(page)
    print("   ligne étape 5 :", row)
    assert row == [stage5_long.INTERVIEW_ID, "disponible", "✅ SUCCESS",
                   "mixte (changement temporel explicite et variations contextuelles)", "10", "2", "2", "1", "1", "1",
                   "2", "5", "0", "1"], row
    page.screenshot(path=str(shots / "M_etape_5_long.png"), full_page=True)
    print("M. étape 5, entretien long OTMANE-like : 50 pratiques, 20 épisodes, 1 appel, configuration mixte")


def stage4_downloads(work: Path) -> list[Path]:
    """Calcule les étapes 3 et 4 de l'entretien de référence (LLM simulé, hors data/) et écrit les 6 JSON
    « téléchargés », nommés comme les boutons de téléchargement de l'interface."""
    run, _ = stage5.run_to_stage4(work / "precomputed4", stage4.FILES, stage4.stage3_responders(),
                                  stage4.REFERENCE_BUILDER)
    out = Path(run["files"][0]["ingestion"]["output_dir"]) / "analysis"
    paths = []
    for name in ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json",
                 "speaker_attribution_audit.json", "accountability_episodes.json",
                 "accountability_episode_validation.json"):
        target = work / "downloads4" / f"{stage4.INTERVIEW_ID}_{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((out / name).read_bytes())
        paths.append(target)
    return paths


def scenario_restore_stage4(browser, url: str, interview: Path, downloads: list[Path], shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    upload_and_ingest(page, interview)
    stage3_input = file_input(page, "Les 4 fichiers JSON de l'étape 3")
    expect(stage3_input).to_have_count(1, timeout=TIMEOUT_MS)
    stage3_input.set_input_files([str(p) for p in downloads[:4]])
    wait_idle(page)
    page.get_by_role("button", name="Restaurer l'étape 3 depuis ces fichiers (0 appel API)").click()
    expect(page.get_by_text("Stage 3 restauré depuis fichiers — 0 appel API").first).to_be_visible(timeout=TIMEOUT_MS)
    wait_idle(page)
    expect(page.get_by_text("Restaurer des résultats Stage 4 existants")).to_be_visible(timeout=TIMEOUT_MS)
    restore = page.get_by_role("button", name="Restaurer l'étape 4 depuis ces fichiers (0 appel API)")
    expect(restore).to_be_disabled()
    stage4_input = file_input(page, "Les 2 fichiers JSON de l'étape 4")
    expect(stage4_input).to_have_count(1, timeout=TIMEOUT_MS)
    stage4_input.set_input_files([str(p) for p in downloads[4:]])
    wait_idle(page)
    expect(restore).to_be_enabled(timeout=TIMEOUT_MS)
    page.screenshot(path=str(shots / "L1_restauration_etape_4_avant.png"), full_page=True)
    restore.click()
    expect(page.get_by_text("Stage 4 restauré depuis fichiers — 0 appel API").first).to_be_visible(timeout=TIMEOUT_MS)
    wait_idle(page)
    row = launch_stage5(page)
    print("   ligne étape 5 :", row)
    assert row == [stage4.INTERVIEW_ID, "disponible", "✅ SUCCESS", "configuration contextuelle", "3", "0", "0", "0",
                   "1", "0", "1", "2", "0", "1"], row
    assert page.get_by_text("Dernière exécution :", exact=False).count() == 0  # aucune exécution de l'étape 3
    assert page.get_by_text("Dernière exécution (étape 4)", exact=False).count() == 0  # ni de l'étape 4
    label = f"Configuration intra-entretien — {stage4.INTERVIEW_ID} — aperçu"
    page.get_by_text(label).click()
    expect(page.get_by_text("(restaurée depuis fichiers)", exact=False).first).to_be_visible(timeout=TIMEOUT_MS)
    assert_no_exception(page)
    page.screenshot(path=str(shots / "L2_restauration_etape_4_etape_5.png"), full_page=True)
    print("L. restauration étapes 3 et 4 : « Stage 4 restauré depuis fichiers — 0 appel API », étape 5 : 1 appel, "
          "0 appel d'étape 3 ou 4")


STAGE6_LAUNCH = "Lancer la comparaison inter-entretiens ({} appel(s) API payant(s))"
STAGE6_LABEL = "Sorties de l'étape 5"


def stage6_upload(page, paths: list[Path]) -> None:
    expect(page.get_by_role("heading", name="Étape 6 — Comparaison inter-entretiens")).to_be_visible(
        timeout=TIMEOUT_MS)
    wait_idle(page)
    file_input(page, STAGE6_LABEL).set_input_files([str(p) for p in paths])
    wait_idle(page)


def stage6_rows(page) -> list[list[str]]:
    table = page.locator("[data-testid=stTable]").filter(has_text="importé").last
    return [[c.strip() for c in row.locator("td").all_inner_texts()] for row in table.locator("tbody tr").all()]


def stage6_download(page, name: str) -> dict:
    wait_idle(page)
    with page.expect_download(timeout=TIMEOUT_MS) as info:
        page.get_by_role("button", name=f"Télécharger {name}").click()
    return json.loads(Path(info.value.path()).read_text(encoding="utf-8"))


def launch_stage6(page, calls: int) -> None:
    launch = page.get_by_role("button", name=STAGE6_LAUNCH.format(calls))
    expect(launch).to_be_enabled(timeout=TIMEOUT_MS)
    launch.click()
    expect(page.get_by_text(f"Dernière exécution (étape 6) : {calls} appel(s) API").first).to_be_visible(
        timeout=TIMEOUT_MS)
    wait_idle(page)
    assert_no_exception(page)


def scenario_stage6_exploratory(browser, url: str, corpus_dir: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    stage6_upload(page, [corpus_dir / name for iid in stage6.IDS_2 for name, _ in stage6.stage5_files(iid)])
    rows = stage6_rows(page)
    print("   corpus :", rows)
    assert [r[:2] for r in rows] == [["ENT_A", "✅ exploitable"], ["ENT_B", "✅ exploitable"]], rows
    expect(page.get_by_text("N exploitables : 2").first).to_be_visible()
    expect(page.get_by_text("la comparaison sera marquée EXPLORATOIRE", exact=False)).to_be_visible()
    launch_stage6(page, 1)
    expect(page.get_by_text("Comparaison EXPLORATOIRE", exact=False).first).to_be_visible()
    document = stage6_download(page, "cross_interview_comparison.json")
    assert document["exploratory"] is True and (document["corpus_n_total"], document["corpus_n_usable"]) == (2, 2)
    assert all(c["confidence"] != "high" for c in document["cross_case_claims"])
    page.screenshot(path=str(shots / "N_etape_6_exploratoire.png"), full_page=True)
    print("N. étape 6 exploratoire : 2 entretiens importés depuis fichiers, 1 appel, aucune confiance « high »")


def scenario_stage6_corpus(browser, url: str, corpus_dir: Path, shots: Path) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 1300})
    page.goto(url)
    garbage = corpus_dir / "notes_illisibles.json"
    garbage.write_text("{pas du json", encoding="utf-8")
    paths = [corpus_dir / name for iid in [*stage6.IDS_17, "ENT_X_INVALIDE"] for name, _ in stage6.stage5_files(iid)]
    stage6_upload(page, [*paths, garbage])
    rows = stage6_rows(page)
    assert len(rows) == 18 and [r[0] for r in rows] == [*stage6.IDS_17, "ENT_X_INVALIDE"], rows
    assert rows[-1][1] == "❌ exclu" and all(r[1] == "✅ exploitable" for r in rows[:-1]), rows
    assert rows[14][:4] == ["ENT_O", "✅ exploitable", "4", "2"], rows[14]  # claims, needs_review
    expect(page.get_by_text("N importés : 18").first).to_be_visible()
    expect(page.get_by_text("N exploitables : 17").first).to_be_visible()
    expect(page.get_by_text("ENT_X_INVALIDE exclu : validation_error_count = 2", exact=False)).to_be_visible()
    expect(page.get_by_text("Fichier ignoré — notes_illisibles.json : JSON invalide.")).to_be_visible()
    page.screenshot(path=str(shots / "O1_etape_6_corpus.png"), full_page=True)
    launch_stage6(page, 1)
    expect(page.get_by_text("Cas négatifs (3)")).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("cas négatifs : 3", exact=False).first).to_be_visible()
    expect(page.get_by_text("Avertissements de validation", exact=False).first).to_be_visible()
    page.get_by_text("Détail par catégorie", exact=False).click()
    expect(page.get_by_text("Frontières récurrentes (2)")).to_be_visible(timeout=TIMEOUT_MS)
    page.screenshot(path=str(shots / "O2_etape_6_resultats.png"), full_page=True)
    document = stage6_download(page, "cross_interview_comparison.json")
    validation = stage6_download(page, "cross_interview_validation.json")
    manifest = stage6_download(page, "cross_interview_manifest.json")
    assert (document["corpus_n_total"], document["corpus_n_usable"]) == (18, 17)
    assert sorted(n["interview_id"] for n in document["negative_cases"]) == ["ENT_D", "ENT_H", "ENT_L"]
    review = [c for c in document["cross_case_claims"] if "SUPPORTED_ONLY_BY_REVIEW_ITEMS" in c["review_reasons"]]
    assert len(review) == 1 and review[0]["confidence"] == "low" and review[0]["needs_review"] is True
    assert validation["error_count"] == 0 and validation["excluded_interviews"][0]["interview_id"] == "ENT_X_INVALIDE"
    assert manifest["api_calls"] == 1 and manifest["map_reduce_used"] is False
    assert manifest["upstream_stage_calls"] == 0 and manifest["estimated_input_tokens"] < 12_000
    print(f"O. étape 6 sur 17 entretiens (+1 invalide, +1 illisible) : 1 appel, "
          f"{manifest['estimated_input_tokens']} tokens estimés, {manifest['payload_chars']} caractères, "
          "3 cas négatifs (ENT_D, ENT_H, ENT_L), 1 pattern à revoir, téléchargements JSON OK")

    relaunch = page.get_by_role("button", name=STAGE6_LAUNCH.format(0))
    expect(relaunch).to_be_visible(timeout=TIMEOUT_MS)
    relaunch.click()
    expect(page.get_by_text("Dernière exécution (étape 6) : 0 appel(s) API").first).to_be_visible(timeout=TIMEOUT_MS)
    expect(page.get_by_text("repris du cache TRACE", exact=False).first).to_be_visible()
    assert_no_exception(page)
    page.screenshot(path=str(shots / "P_etape_6_cache.png"), full_page=True)
    print("P. relance de l'étape 6 : CACHED, 0 appel API")


def run_dirs() -> set[Path]:
    return {p for base in (ROOT / "data" / "inputs", ROOT / "data" / "outputs") for p in base.glob("run_*")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshots", type=Path, default=Path(tempfile.mkdtemp(prefix="trace_e2e_")))
    parser.add_argument("--chromium", default=os.environ.get("TRACE_CHROMIUM", "/opt/pw-browsers/chromium"))
    parser.add_argument("--only-stage6", action="store_true", help="scénarios N à P seulement")
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="trace_e2e_data_"))
    interview = work / si.FILENAME
    interview.write_text(si.TEXT, encoding="utf-8")
    stage35 = work / si.STAGE35_FILES[0][0]
    stage35.write_bytes(si.STAGE35_FILES[0][1])
    long_file = work / long_interview.LONG_FILENAME
    long_file.write_text(long_interview.long_text(), encoding="utf-8")
    stage37_file = work / stage37.FILENAME
    stage37_file.write_text(stage37.text(), encoding="utf-8")
    stage4_file = work / stage4.FILENAME
    stage4_file.write_text(stage4.TEXT, encoding="utf-8")
    stage4_long_file = work / stage4_long.FILENAME
    stage4_long_file.write_text(stage4_long.text(), encoding="utf-8")
    otmane_file = work / otmane.FILENAME
    otmane_file.write_text(otmane.text(), encoding="utf-8")
    temporal_file = work / stage5.TEMPORAL.filename
    temporal_file.write_text(stage5.TEMPORAL.text, encoding="utf-8")
    stage5_long_file = work / stage5_long.FILENAME
    stage5_long_file.write_text(stage5_long.text(), encoding="utf-8")
    stage6_dir = work / "stage6_corpus"
    stage6.write_corpus(stage6_dir, [*stage6.IDS_17, "ENT_X_INVALIDE"])
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "TRACE_"))}
    before = run_dirs()
    try:
        with sync_playwright() as pw:
            executable = args.chromium if Path(args.chromium).exists() else None
            browser = pw.chromium.launch(executable_path=executable, args=["--no-proxy-server"])
            env6 = {**base_env, "TRACE_E2E_CACHE_DIR": str(work / "cache_stage6"),
                    "TRACE_E2E_CROSS_DIR": str(work / "cross_interview")}
            with streamlit_server(ROOT / "tests" / "e2e" / "fake_llm_app.py", env6) as url:
                scenario_stage6_exploratory(browser, url, stage6_dir, args.screenshots)
                scenario_stage6_corpus(browser, url, stage6_dir, args.screenshots)
            if args.only_stage6:
                browser.close()
                print(f"Test navigateur (étape 6) réussi. Captures : {args.screenshots}")
                return 0
            with streamlit_server(ROOT / "app.py", base_env) as url:
                scenario_without_key(browser, url, interview, args.screenshots)
            env = {**base_env, "TRACE_E2E_CACHE_DIR": str(work / "cache")}
            with streamlit_server(ROOT / "tests" / "e2e" / "fake_llm_app.py", env) as url:
                scenario_fake_llm(browser, url, interview, args.screenshots)
                scenario_stage35(browser, url, stage35, args.screenshots)
                scenario_long_interview(browser, url, long_file, args.screenshots)
                scenario_stage37(browser, url, stage37_file, args.screenshots)
                scenario_stage4(browser, url, stage4_file, args.screenshots)
                scenario_stage4_long(browser, url, stage4_long_file, args.screenshots)
                scenario_otmane(browser, url, otmane_file, args.screenshots)
                scenario_stage5_temporal(browser, url, temporal_file, args.screenshots)
                scenario_stage5_long(browser, url, stage5_long_file, args.screenshots)
            downloads = stage3_downloads(work)
            downloads4 = stage4_downloads(work)
            env_restore = {**base_env, "TRACE_E2E_CACHE_DIR": str(work / "cache_after_redeploy")}
            with streamlit_server(ROOT / "tests" / "e2e" / "fake_llm_app.py", env_restore) as url:
                scenario_restore(browser, url, stage4_file, downloads, args.screenshots)
            env_restore4 = {**base_env, "TRACE_E2E_CACHE_DIR": str(work / "cache_after_redeploy_4")}
            with streamlit_server(ROOT / "tests" / "e2e" / "fake_llm_app.py", env_restore4) as url:
                scenario_restore_stage4(browser, url, stage4_file, downloads4, args.screenshots)
            browser.close()
    finally:
        for path in run_dirs() - before:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    print(f"Test navigateur réussi. Captures : {args.screenshots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
