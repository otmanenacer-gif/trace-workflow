"""Interface Streamlit de l'étape 4 (AppTest : sans navigateur, LLM simulé, aucun appel réel)."""

import pytest
from streamlit.testing.v1 import AppTest

import core.llm_client as llm_client
from core import config
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S
from tests.fake_llm import ACCOUNTABILITY, INTERACTION, FakeTransport, server_error

APP = str(config.PROJECT_ROOT / "app.py")


def texts(elements):
    return " ".join(str(e.value) for e in elements)


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    monkeypatch.setenv("TRACE_STAGE3_BACKEND", "anthropic")  # ancien mode de l'étape 3 par API (LLM simulé)
    transport = FakeTransport({**S.stage3_responders(), ACCOUNTABILITY: S.REFERENCE_BUILDER})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    return transport


def app_with_run(run):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    return at.run()


def agents(transport):
    return [c["agent"] for c in transport.calls]


def test_stage4_waits_for_stage3(tmp_path, fake_api):
    at = app_with_run(si.make_ingested_run(tmp_path, S.FILES))
    assert not at.exception
    assert "Étape 4 — Épisodes d'accountability" in texts(at.header)
    assert "Lancez d'abord l'étape 3" in texts(at.info)
    assert not [b for b in at.button if b.key == "acc_launch"]


def test_stage4_full_flow_with_fake_llm_then_cache(tmp_path, fake_api):
    at = app_with_run(si.make_ingested_run(tmp_path, S.FILES))
    at.button(key="ai_launch").click().run()
    assert not at.exception and agents(fake_api).count(ACCOUNTABILITY) == 0  # l'étape 3 ne lance jamais l'étape 4
    assert "Candidats d'épisodes (déterministes, sans IA) : **6**" in texts(at.info)
    launch = at.button(key="acc_launch")
    assert launch.label == "Construire les épisodes d'accountability (1 appel(s) API payant(s))"
    stage3_calls = len(fake_api.calls)

    launch.click().run()
    assert not at.exception
    assert agents(fake_api)[stage3_calls:] == [ACCOUNTABILITY]
    assert "Étape 4 terminée." in texts(at.success)
    table = at.table[-1].value
    assert list(table["Étape 4"]) == ["✅ SUCCESS"] and list(table["Statut étape 3"]) == ["COMPLETE"]
    assert list(table["Candidats"]) == [6] and list(table["Épisodes accountability"]) == [4]
    assert list(table["Pratiques ordinaires (examinées)"]) == [1] and list(table["Pratiques sans marqueur"]) == [4]
    assert list(table["Incertains"]) == [1] and list(table["Avertissements"]) == [0]
    labels = [b.label for b in at.get("download_button")]
    for name in ("accountability_episodes.json", "accountability_episode_validation.json"):
        assert f"Télécharger {name}" in labels
    assert "Épisodes d'accountability — ENTRETIEN_ETAPE_4 — aperçu" in [e.label for e in at.expander]
    preview = [df for df in at.dataframe if "opérations" in df.value.columns]
    assert len(preview) == 1 and len(preview[0].value) == 3
    assert "Construction des épisodes d'accountability** (IA) — terminé (1/1 entretien(s))" in texts(at.markdown)

    # Relance : le résultat vient du cache, aucun nouvel appel
    assert at.button(key="acc_launch").label.endswith("(0 appel(s) API payant(s))")
    at.button(key="acc_launch").click().run()
    assert not at.exception and agents(fake_api)[stage3_calls:] == [ACCOUNTABILITY]
    assert list(at.table[-1].value["Étape 4"]) == ["♻️ CACHED"]


def test_stage4_blocked_when_stage3_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    monkeypatch.setenv("TRACE_STAGE3_BACKEND", "anthropic")  # ancien mode de l'étape 3 par API (LLM simulé)
    responders = {**S.stage3_responders(), INTERACTION: server_error(500), ACCOUNTABILITY: S.REFERENCE_BUILDER}
    transport = FakeTransport(responders)
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    at = app_with_run(si.make_ingested_run(tmp_path, S.FILES))
    at.button(key="ai_launch").click().run()
    assert "1 entretien(s) bloqué(s)" in texts(at.info)
    at.button(key="acc_launch").click().run()
    assert not at.exception and ACCOUNTABILITY not in agents(transport)
    assert list(at.table[-1].value["Étape 4"]) == ["⛔ BLOCKED"]
    assert "Étape 4 non exécutée" in texts(at.error)
    assert "Télécharger accountability_episodes.json" not in [b.label for b in at.get("download_button")]
