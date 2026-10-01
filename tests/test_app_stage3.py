"""Tests de l'interface Streamlit pour l'étape 3 (AppTest : sans navigateur, LLM simulé)."""

import pytest
from streamlit.testing.v1 import AppTest

import core.llm_client as llm_client
from core import config
from tests import synthetic_interviews as si
from tests.fake_llm import INTERACTION, PRACTICE, FakeTransport, text_response

APP = str(config.PROJECT_ROOT / "app.py")


def app_with_run(run):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    return at.run()


def texts(elements):
    return " ".join(str(e.value) for e in elements)


@pytest.fixture
def fake_api(monkeypatch):
    """Clé et modèle factices + transport simulé : le parcours complet sans aucun appel réel."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    transport = FakeTransport({PRACTICE: lambda p: text_response(si.GOOD_PRACTICES),
                               INTERACTION: lambda p: text_response(si.GOOD_SIGNALS)})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    return transport


def test_app_starts_without_api_key():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "désactivée (clé API ou modèle absent)" in texts(at.markdown)


def test_ai_section_is_disabled_without_key(tmp_path):
    at = app_with_run(si.make_ingested_run(tmp_path))
    assert not at.exception
    assert "Analyse IA — Étape 3" in texts(at.header)
    assert "Analyse IA désactivée" in texts(at.warning) and "ANTHROPIC_API_KEY" in texts(at.warning)
    assert not [b for b in at.button if b.key == "ai_launch"]


def test_no_analysis_without_explicit_click(tmp_path, fake_api):
    at = app_with_run(si.make_ingested_run(tmp_path))
    assert not at.exception
    assert fake_api.calls == []
    assert "2 appels LLM par entretien non présent dans le cache" in texts(at.info)
    launch = at.button(key="ai_launch")
    assert "appel(s) API payant(s)" in launch.label and not launch.disabled
    assert "sk-ant-test-FAKE-KEY-000" not in texts(at.markdown) + texts(at.info)


def test_test_mode_runs_both_agents_and_shows_results(tmp_path, fake_api):
    at = app_with_run(si.make_ingested_run(tmp_path))
    at.button(key="ai_launch").click().run()
    assert not at.exception
    assert len(fake_api.calls) == 2
    assert "Analyse IA terminée." in texts(at.success)
    table = at.table[-1].value
    assert list(table["Practice Extractor"]) == ["✅ SUCCESS"] and list(table["Interaction Reader"]) == ["✅ SUCCESS"]
    assert list(table["Pratiques"]) == [6] and list(table["Signaux"]) == [10]
    assert list(table["Citations invalides"]) == [0]
    assert "2 appel(s) API — 2 400 tokens entrée — 600 tokens sortie" in texts(at.markdown)
    labels = [b.label for b in at.get("download_button")]
    for name in ("practice_extractor.json", "interaction_signals.json", "evidence_validation.json"):
        assert f"Télécharger {name}" in labels
    assert "🟢 **2. Extraction des pratiques** (IA) — terminé (1/1 entretien(s))" in texts(at.markdown)

    # Second clic : tout vient du cache, aucun nouvel appel
    at.button(key="ai_launch").click().run()
    assert len(fake_api.calls) == 2
    assert list(at.table[-1].value["Practice Extractor"]) == ["♻️ CACHED"]


def test_corpus_mode_requires_confirmation(tmp_path, fake_api):
    files = [(si.FILENAME, si.TEXT.encode("utf-8")), si.other_interview("Bob.txt", "Oui, pour traduire.")]
    at = app_with_run(si.make_ingested_run(tmp_path, files))
    at.radio(key="ai_mode").set_value("Corpus complet").run()
    assert at.button(key="ai_launch").disabled
    assert "4 appel(s) API" in at.button(key="ai_launch").label
    at.checkbox(key="ai_confirm_corpus").check().run()
    assert not at.button(key="ai_launch").disabled
    at.button(key="ai_launch").click().run()
    assert not at.exception and len(fake_api.calls) == 4
    # La confirmation ne vaut que pour un lancement
    assert not at.checkbox(key="ai_confirm_corpus").value and at.button(key="ai_launch").disabled


def test_force_option_is_reset_after_each_launch(tmp_path, fake_api):
    at = app_with_run(si.make_ingested_run(tmp_path))
    at.button(key="ai_launch").click().run()
    at.checkbox(key="ai_force").check().run()
    assert "(2 appel(s) API payant(s))" in at.button(key="ai_launch").label
    at.button(key="ai_launch").click().run()
    assert len(fake_api.calls) == 4 and not at.checkbox(key="ai_force").value
    assert "(0 appel(s) API payant(s))" in at.button(key="ai_launch").label


def test_speaker_audit_is_shown_and_never_editable(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    from tests.fake_llm import AUDITOR
    transport = FakeTransport({PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
                               INTERACTION: lambda p: text_response({"signals": [], "reading_notes": None}),
                               AUDITOR: lambda p: text_response(si.AUDIT_ASSESSMENTS)})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    at = app_with_run(si.make_ingested_run(tmp_path, si.AUDIT_FILES))
    assert "2 tour(s) suspect(s), 1 appel(s) d'audit" in texts(at.info)
    assert "(au plus 3 appel(s) API payant(s))" in at.button(key="ai_launch").label
    at.button(key="ai_launch").click().run()
    assert not at.exception and len(transport.calls) == 3
    table = at.table[-1].value
    assert list(table["Audit locuteurs"]) == ["✅ SUCCESS"]
    assert list(table["Tours suspects"]) == [2] and list(table["Locuteurs à vérifier"]) == [2]
    assert "Ces suggestions ne modifient pas la transcription originale." in texts(at.info)
    assert "Audit d'attribution des locuteurs — ENTRETIEN_LOCUTEURS — 2 tour(s) suspect(s), 2 à vérifier" in [
        e.label for e in at.expander]
    assert "Télécharger speaker_attribution_audit.json" in [b.label for b in at.get("download_button")]
    # aucun champ de saisie ne permet de modifier la transcription (seule la problématique est éditable)
    assert not at.text_input and [t.key for t in at.text_area] == ["problematique"]
    # relance : tout vient du cache, y compris l'audit
    assert "(0 appel(s) API payant(s))" in at.button(key="ai_launch").label
