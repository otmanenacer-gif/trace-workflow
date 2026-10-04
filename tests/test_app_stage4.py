"""Interface Streamlit de l'étape 4 (AppTest : sans navigateur, faux Ollama derrière le vrai runner local)."""

import pytest
from streamlit.testing.v1 import AppTest

from core import config
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S
from tests.fake_llm import ACCOUNTABILITY, INTERACTION, agent_error, use_fake_runtime

APP = str(config.PROJECT_ROOT / "app.py")


def texts(elements):
    return " ".join(str(e.value) for e in elements)


def simulate(monkeypatch, responders: dict):
    """Les agents sont exécutés par le vrai runner local, sur un faux Ollama (aucun modèle, aucun réseau)."""
    return use_fake_runtime(monkeypatch, responders)


@pytest.fixture
def fake_agents(monkeypatch):
    return simulate(monkeypatch, {**S.stage3_responders(), ACCOUNTABILITY: S.REFERENCE_BUILDER})


def app_with_run(run):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    return at.run()


def agents(transport):
    return [c["agent"] for c in transport.calls]


def test_stage4_waits_for_stage3(tmp_path, fake_agents):
    at = app_with_run(si.make_ingested_run(tmp_path, S.FILES))
    assert not at.exception
    assert "Étape 4 — Épisodes d'accountability" in texts(at.header)
    assert "Lancez d'abord l'étape 3" in texts(at.info)
    assert not [b for b in at.button if b.key in ("run_stage4", "acc_launch")]


def test_stage4_full_flow_then_no_replay(tmp_path, fake_agents):
    at = app_with_run(si.make_ingested_run(tmp_path, S.FILES))
    at.button(key="run_stage3").click().run()
    assert not at.exception and agents(fake_agents).count(ACCOUNTABILITY) == 0  # l'étape 3 ne lance jamais l'étape 4
    stage3_calls = len(fake_agents.calls)

    at.button(key="run_stage4").click().run()
    assert not at.exception
    assert agents(fake_agents)[stage3_calls:] == [ACCOUNTABILITY] * 2  # 6 candidats : blocs de 4 et 2
    assert "Étape 4 terminée (exécution locale, modèle qwen2.5:7b, 0 appel API)." in texts(at.success)
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

    # Relance : l'étape 4 est complète et à jour, elle n'est pas rejouée
    at.button(key="run_stage4").click().run()
    assert not at.exception and agents(fake_agents)[stage3_calls:] == [ACCOUNTABILITY] * 2
    assert list(at.table[-1].value["Étape 4"]) == ["✅ SUCCESS"]


def test_stage4_blocked_when_stage3_failed(tmp_path, monkeypatch):
    responders = {**S.stage3_responders(), INTERACTION: agent_error(), ACCOUNTABILITY: S.REFERENCE_BUILDER}
    transport = simulate(monkeypatch, responders)
    at = app_with_run(si.make_ingested_run(tmp_path, S.FILES))
    at.button(key="run_stage3").click().run()
    assert "Étape 3 : échec" in texts(at.warning)  # l'Interaction Reader a échoué : étape 3 incomplète
    at.button(key="run_stage4").click().run()
    assert not at.exception and ACCOUNTABILITY not in agents(transport)
    assert list(at.table[-1].value["Étape 4"]) == ["⛔ BLOCKED"]
    assert "Étape 4 non exécutée" in texts(at.error)
    assert "Télécharger accountability_episodes.json" not in [b.label for b in at.get("download_button")]
