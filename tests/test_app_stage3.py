"""Interface Streamlit de l'étape 3 (AppTest : sans navigateur, faux Ollama derrière le vrai runner local)."""

import pytest
from streamlit.testing.v1 import AppTest

from core import config
from tests import synthetic_interviews as si
from tests.fake_llm import AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, use_fake_runtime, text_response

APP = str(config.PROJECT_ROOT / "app.py")


def app_with_run(run):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    return at.run()


def texts(elements):
    return " ".join(str(e.value) for e in elements)


def simulate(monkeypatch, responders: dict):
    """Les agents sont exécutés par le vrai runner local, sur un faux Ollama (aucun modèle, aucun réseau)."""
    return use_fake_runtime(monkeypatch, responders)


@pytest.fixture
def fake_agents(monkeypatch):
    return simulate(monkeypatch, {PRACTICE: lambda p: text_response(si.GOOD_PRACTICES),
                                  INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)})


def test_app_starts_without_api_key():
    # Étapes 3 à 6 : exécution locale (Ollama), aucune clé ; aucune étape IA n'est désactivée.
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "désactivée" not in texts(at.markdown) and "ANTHROPIC" not in texts(at.markdown) + texts(at.warning)
    assert texts(at.markdown).count("prête, exécution locale (Ollama)") == 7  # étapes IA 3 à 9 du pipeline
    assert ("**Exécution :** locale · **Runtime :** Ollama · **Modèle :** `qwen2.5:7b` · **Données externes :** "
            "aucune") in texts(at.markdown)
    assert "Ollama ne répond pas" in texts(at.error)  # aucun vrai Ollama pendant les tests : état affiché clairement


def test_runtime_panel_shows_a_ready_local_ollama(fake_agents):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and "modèle `qwen2.5:7b` installé" in texts(at.success)
    assert "aucune donnée envoyée hors de cet ordinateur" in texts(at.success)


def test_missing_model_is_shown_with_the_install_command(monkeypatch):
    from tests.fake_llm import use_fake_runtime as fake
    fake(monkeypatch, {}, models=["llama3.2:3b"])
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and "`ollama pull qwen2.5:7b`" in texts(at.error)


def test_running_a_stage_with_ollama_stopped_shows_a_clear_error(tmp_path, monkeypatch):
    from tests.fake_llm import use_fake_runtime as fake
    ollama = fake(monkeypatch, {}, available=False)
    at = app_with_run(si.make_ingested_run(tmp_path))
    at.button(key="run_stage3").click().run()
    assert not at.exception and ollama.calls == []
    assert "Ollama ne répond pas" in texts(at.error) and "Aucun autre fournisseur" in texts(at.error)


def test_no_analysis_without_explicit_click(tmp_path, fake_agents):
    at = app_with_run(si.make_ingested_run(tmp_path))
    assert not at.exception
    assert fake_agents.calls == []
    assert "Analyse IA — Étape 3" in texts(at.header) and "Exécution locale" in texts(at.info)
    assert not at.button(key="run_stage3").disabled
    assert not [b for b in at.button if b.key in ("ai_launch", "ai_force", "ai_confirm_corpus")]  # anciens lanceurs


def test_test_mode_runs_both_agents_and_shows_results(tmp_path, fake_agents):
    at = app_with_run(si.make_ingested_run(tmp_path))
    at.button(key="run_stage3").click().run()
    assert not at.exception
    assert sorted(c["agent"] for c in fake_agents.calls) == [INTERACTION, PRACTICE]
    assert "Étape 3 terminée (exécution locale, modèle qwen2.5:7b, 0 appel API)." in texts(at.success)
    table = at.table[-1].value
    assert list(table["Practice Extractor"]) == ["✅ SUCCESS"] and list(table["Interaction Reader"]) == ["✅ SUCCESS"]
    assert list(table["Pratiques"]) == [6] and list(table["Signaux"]) == [10]
    assert list(table["Citations invalides"]) == [0]
    assert "0 appel(s) API — 0 tokens entrée — 0 tokens sortie" in texts(at.markdown)
    labels = [b.label for b in at.get("download_button")]
    for name in ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json"):
        assert f"Télécharger {name}" in labels
    assert "🟢 **3. Extraction des pratiques et analyse interactionnelle** (IA) — terminé (1/1 entretien(s))" in texts(at.markdown)

    # Second clic : l'étape est complète et à jour, elle n'est pas rejouée
    at.button(key="run_stage3").click().run()
    assert not at.exception and len(fake_agents.calls) == 2
    assert "Étape 3 terminée (exécution locale, modèle qwen2.5:7b, 0 appel API)." in texts(at.success)
    assert list(at.table[-1].value["Practice Extractor"]) == ["✅ SUCCESS"]


def test_corpus_mode_runs_every_interview(tmp_path, fake_agents):
    files = [(si.FILENAME, si.TEXT.encode("utf-8")), si.other_interview("Bob.txt", "Oui, pour traduire.")]
    at = app_with_run(si.make_ingested_run(tmp_path, files))
    at.radio(key="wf_mode").set_value("Corpus complet").run()
    at.button(key="run_stage3").click().run()
    assert not at.exception
    sent = " ".join(c["params"]["messages"][0]["content"] for c in fake_agents.calls)
    assert si.INTERVIEW_ID + "_T" in sent and "BOB_T" in sent  # les deux entretiens du run
    assert len(at.table[-1].value) == 2


def test_speaker_audit_is_shown_and_never_editable(tmp_path, monkeypatch):
    agents = simulate(monkeypatch, {PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
                                    INTERACTION: lambda p: text_response({"signals": [], "reading_notes": None}),
                                    AUDITOR: lambda p: text_response(si.AUDIT_ASSESSMENTS)})
    at = app_with_run(si.make_ingested_run(tmp_path, si.AUDIT_FILES))
    at.button(key="run_stage3").click().run()
    assert not at.exception and len(agents.calls) == 3 and agents.calls[0]["agent"] == AUDITOR
    table = at.table[-1].value
    assert list(table["Audit locuteurs"]) == ["✅ SUCCESS"]
    assert list(table["Tours suspects"]) == [2] and list(table["Locuteurs à vérifier"]) == [2]
    assert "Ces suggestions ne modifient pas la transcription originale." in texts(at.info)
    assert "Audit d'attribution des locuteurs — ENTRETIEN_LOCUTEURS — 2 tour(s) suspect(s), 2 à vérifier" in [
        e.label for e in at.expander]
    assert "Télécharger speaker_attribution_audit.json" in [b.label for b in at.get("download_button")]
    # aucun champ de saisie ne permet de modifier la transcription (seule la problématique est éditable ; le seul
    # champ texte est celui des épisodes signalés « interprétation à vérifier » du rapport final, étape 7 ; plus le
    # nom facultatif d'un corpus de l'analyse complète)
    assert [t.key for t in at.text_input] == ["full_label", "final_report_to_verify"]
    assert [t.key for t in at.text_area] == ["problematique"]


@pytest.fixture
def restore_project_modules():
    """Rétablit les modules du projet : les mocks des autres tests visent ces objets-là."""
    import sys
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] in ("core", "agents")}
    yield
    for name in [n for n in sys.modules if n.split(".")[0] in ("core", "agents")]:
        del sys.modules[name]
    sys.modules.update(saved)
    for name, module in saved.items():  # `from core import analysis` lit l'attribut du paquet, pas sys.modules
        package, _, attribute = name.rpartition(".")
        if package in saved:
            setattr(saved[package], attribute, module)


def test_app_reloads_stale_analysis_module_after_hot_redeploy(restore_project_modules):
    """Redéploiement à chaud : un ancien core.analysis (sans AUDITOR) reste en mémoire."""
    import sys
    import types
    from core import analysis
    stale = types.ModuleType("core.analysis")
    stale.__file__ = analysis.__file__
    stale._trace_loaded_at = 0.0  # chargé avant la dernière modification du fichier
    for name in ("AGENTS", "STATUS_PENDING", "STATUS_RUNNING", "STATUS_SUCCESS",
                 "STATUS_SUCCESS_WITH_WARNINGS", "STATUS_CACHED", "STATUS_FAILED"):
        setattr(stale, name, getattr(analysis, name))
    sys.modules["core.analysis"] = stale
    sys.modules["core"].analysis = stale

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert hasattr(sys.modules["core.analysis"], "AUDITOR")


# --- Étape 3.6 : entretien long lu par blocs ----------------------------------------------------

def long_interview_agents(monkeypatch, interaction=None):
    from tests import synthetic_long_interview as L
    return simulate(monkeypatch, {PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
                                  INTERACTION: interaction or L.simulated_chunk_reader,
                                  LONG_DISTANCE: L.simulated_long_distance_reader})


def test_long_interview_shows_chunk_results(tmp_path, monkeypatch):
    from tests import synthetic_long_interview as L
    agents = long_interview_agents(monkeypatch)
    at = app_with_run(si.make_ingested_run(tmp_path, L.long_files()))
    assert not at.exception and agents.calls == []
    at.button(key="run_stage3").click().run()
    # étape 3.7 : le Practice Extractor lit aussi l'entretien long par blocs (6 blocs) : 6 + 5 + 1 tâches
    assert not at.exception and len(agents.calls) == 12
    table = at.table[-1].value
    assert list(table["Interaction Reader"]) == ["✅ SUCCESS"] and list(table["Blocs (Interaction)"]) == ["5/5"]
    assert "entretien long : analyse en 5 blocs (chevauchement inclus) · blocs réussis : **5/5**" in texts(at.markdown)
    assert "Télécharger interaction_signals.json" in [b.label for b in at.get("download_button")]


def test_failed_block_is_shown_as_incomplete_and_listed_for_correction(tmp_path, monkeypatch):
    from tests import synthetic_long_interview as L

    def incomplete(params):
        if any(t["turn_id"] == L.ltid(200) for t in L.sent_turns(params)):
            return text_response('{"signals": [')  # réponse incomplète : JSON invalide
        return L.simulated_chunk_reader(params)

    long_interview_agents(monkeypatch, incomplete)
    at = app_with_run(si.make_ingested_run(tmp_path, L.long_files()))
    at.button(key="run_stage3").click().run()
    assert not at.exception
    table = at.table[-1].value
    assert list(table["Interaction Reader"]) == ["🟠 PARTIAL"] and list(table["Blocs (Interaction)"]) == ["4/5"]
    errors = texts(at.error)
    assert "Analyse INCOMPLÈTE : 4/5 bloc(s) réussi(s)" in errors and "JSON invalide" in errors


# --- Étape 3.7 : Practice Extractor par blocs, sélectivité, anomalies de validation -----------------

def stage37_agents(monkeypatch, practice=None):
    from tests import synthetic_stage37 as S
    return simulate(monkeypatch, {PRACTICE: practice or S.practice_reader, INTERACTION: S.naive_reader,
                                  LONG_DISTANCE: S.long_distance_reader,
                                  AUDITOR: lambda p: text_response({"assessments": [], "audit_notes": None})})


def test_stage37_long_interview_reports_both_chunked_agents(tmp_path, monkeypatch):
    from tests import synthetic_stage37 as S
    agents = stage37_agents(monkeypatch)
    at = app_with_run(si.make_ingested_run(tmp_path, S.files()))
    assert not at.exception and agents.calls == []
    at.button(key="run_stage3").click().run()
    assert not at.exception and len(agents.calls) == 9  # 1 audit + 4 + 3 + 1
    table = at.table[-1].value
    assert list(table["Practice Extractor"]) == ["✅ SUCCESS"] and list(table["Interaction Reader"]) == ["✅ SUCCESS"]
    assert list(table["Blocs (Practice)"]) == ["4/4"] and list(table["Blocs (Interaction)"]) == ["3/3"]
    assert list(table["Anomalies validation"]) == [0] and list(table["Pratiques"]) == [S.EXPECTED_PRACTICE_COUNT]
    markdown = texts(at.markdown)
    assert ("**Practice Extractor** — entretien long : analyse en 4 blocs (chevauchement inclus) · blocs réussis : "
            f"**4/4** · pratiques finales : **{S.EXPECTED_PRACTICE_COUNT}** (avant dédoublonnage : "
            f"{S.EXPECTED_PRACTICE_COUNT + 1}) · statut : SUCCESS") in markdown
    assert "**Interaction Reader** — entretien long : analyse en 3 blocs" in markdown
    assert "lecture à longue distance : SUCCESS · anomalies de validation : 0" in markdown
    assert "signal(aux) écarté(s) avant validation" in texts(at.caption)
    labels = [b.label for b in at.get("download_button")]
    for name in ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json"):
        assert f"Télécharger {name}" in labels
    at.button(key="run_stage3").click().run()  # relance : complète et à jour, non rejouée
    assert not at.exception and len(agents.calls) == 9


def test_stage37_failed_practice_block_is_shown_as_incomplete(tmp_path, monkeypatch):
    from tests import synthetic_stage37 as S
    from tests.synthetic_long_interview import sent_turns

    def incomplete(params):
        if any(t["turn_id"] == S.tid(140) for t in sent_turns(params)):
            return text_response('{"practices": [')
        return S.practice_reader(params)

    stage37_agents(monkeypatch, incomplete)
    at = app_with_run(si.make_ingested_run(tmp_path, S.files()))
    at.button(key="run_stage3").click().run()
    assert not at.exception
    table = at.table[-1].value
    assert list(table["Practice Extractor"]) == ["🟠 PARTIAL"] and list(table["Blocs (Practice)"]) == ["3/4"]
    assert "Analyse INCOMPLÈTE : 3/4 bloc(s) réussi(s)" in texts(at.error)
