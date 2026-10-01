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

def long_interview_api(monkeypatch, interaction=None):
    from tests import synthetic_long_interview as L
    from tests.fake_llm import LONG_DISTANCE
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    transport = FakeTransport({PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
                               INTERACTION: interaction or L.simulated_chunk_reader,
                               LONG_DISTANCE: L.simulated_long_distance_reader})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    return transport


def test_long_interview_announces_blocks_then_shows_chunk_results(tmp_path, monkeypatch):
    from tests import synthetic_long_interview as L
    transport = long_interview_api(monkeypatch)
    at = app_with_run(si.make_ingested_run(tmp_path, L.long_files()))
    assert not at.exception and transport.calls == []
    assert ("Interaction Reader — ENTRETIEN_LONG : entretien long : analyse en **5 blocs** (chevauchement inclus)"
            in texts(at.info))
    assert "au plus **6 appels** Interaction Reader" in texts(at.info)
    assert "Interaction Reader : au plus **6 appel(s)**" in texts(at.caption)
    # étape 3.7 : le Practice Extractor lit aussi l'entretien long par blocs (6 blocs) : 6 + 6 appels au plus
    assert "(au plus 12 appel(s) API payant(s))" in at.button(key="ai_launch").label
    at.button(key="ai_launch").click().run()
    assert not at.exception and len(transport.calls) == 12
    table = at.table[-1].value
    assert list(table["Interaction Reader"]) == ["✅ SUCCESS"] and list(table["Blocs (Interaction)"]) == ["5/5"]
    assert "entretien long : analyse en 5 blocs (chevauchement inclus) · blocs réussis : **5/5**" in texts(at.markdown)
    assert "Télécharger interaction_signals.json" in [b.label for b in at.get("download_button")]
    assert "(0 appel(s) API payant(s))" in at.button(key="ai_launch").label  # tout est en cache


def test_truncated_block_is_shown_as_incomplete(tmp_path, monkeypatch):
    from tests import synthetic_long_interview as L

    def truncating(params):
        if any(t["turn_id"] == L.ltid(200) for t in L.sent_turns(params)):
            return text_response('{"signals": [', stop_reason="max_tokens")
        return L.simulated_chunk_reader(params)

    long_interview_api(monkeypatch, truncating)
    at = app_with_run(si.make_ingested_run(tmp_path, L.long_files()))
    at.button(key="ai_launch").click().run()
    assert not at.exception
    table = at.table[-1].value
    assert list(table["Interaction Reader"]) == ["🟠 PARTIAL"] and list(table["Blocs (Interaction)"]) == ["4/5"]
    assert "analyse(s) incomplète(s)" in texts(at.warning)
    errors = texts(at.error)
    assert "Analyse INCOMPLÈTE : 4/5 bloc(s) réussi(s)" in errors and "Bloc(s) tronqué(s)" in errors
    assert "(au plus 2 appel(s) API payant(s))" in at.button(key="ai_launch").label  # bloc 3 + longue distance


# --- Étape 3.7 : Practice Extractor par blocs, sélectivité, anomalies de validation -----------------

def stage37_api(monkeypatch, practice=None):
    from tests import synthetic_stage37 as S
    from tests.fake_llm import AUDITOR, LONG_DISTANCE
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    transport = FakeTransport({PRACTICE: practice or S.practice_reader, INTERACTION: S.naive_reader,
                               LONG_DISTANCE: S.long_distance_reader,
                               AUDITOR: lambda p: text_response({"assessments": [], "audit_notes": None})})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    return transport


def test_stage37_long_interview_announces_and_reports_both_chunked_agents(tmp_path, monkeypatch):
    from tests import synthetic_stage37 as S
    transport = stage37_api(monkeypatch)
    at = app_with_run(si.make_ingested_run(tmp_path, S.files()))
    assert not at.exception and transport.calls == []
    infos = texts(at.info)
    assert f"Practice Extractor — {S.INTERVIEW_ID} : entretien long : analyse en **4 blocs**" in infos
    assert "soit au plus **4 appels** Practice Extractor" in infos
    assert f"Interaction Reader — {S.INTERVIEW_ID} : entretien long : analyse en **3 blocs**" in infos
    assert "Practice Extractor : au plus **4 appel(s)** · Interaction Reader : au plus **4 appel(s)**" in texts(at.caption)
    assert "(au plus 9 appel(s) API payant(s))" in at.button(key="ai_launch").label  # 1 audit + 4 + 3 + 1
    at.button(key="ai_launch").click().run()
    assert not at.exception and len(transport.calls) == 9
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
    assert "(0 appel(s) API payant(s))" in at.button(key="ai_launch").label  # relance : tout est en cache
    at.button(key="ai_launch").click().run()
    assert not at.exception and len(transport.calls) == 9


def test_stage37_truncated_practice_block_is_shown_as_incomplete(tmp_path, monkeypatch):
    from tests import synthetic_stage37 as S
    from tests.synthetic_long_interview import sent_turns

    def truncating(params):
        if any(t["turn_id"] == S.tid(140) for t in sent_turns(params)):
            return text_response('{"practices": [', stop_reason="max_tokens")
        return S.practice_reader(params)

    stage37_api(monkeypatch, truncating)
    at = app_with_run(si.make_ingested_run(tmp_path, S.files()))
    at.button(key="ai_launch").click().run()
    assert not at.exception
    table = at.table[-1].value
    assert list(table["Practice Extractor"]) == ["🟠 PARTIAL"] and list(table["Blocs (Practice)"]) == ["3/4"]
    assert "Practice Extractor — Bloc(s) tronqué(s)" in texts(at.error)
    assert "(1 appel(s) API payant(s))" in at.button(key="ai_launch").label  # seul le bloc tronqué est à refaire
