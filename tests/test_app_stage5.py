"""Interface Streamlit de l'étape 5 (AppTest : sans navigateur, LLM simulé, aucun appel réel)."""

import pytest
from streamlit.testing.v1 import AppTest

import core.llm_client as llm_client
from core import config
from core.stage3_restore import restore_stage3
from core.stage4_restore import restore_stage4
from tests import synthetic_interviews as si
from tests import synthetic_stage4 as S4
from tests import synthetic_stage5 as S5
from tests.fake_llm import ACCOUNTABILITY, TRAJECTORY, FakeTransport

APP = str(config.PROJECT_ROOT / "app.py")


def texts(elements):
    return " ".join(str(e.value) for e in elements)


def agents(transport):
    return [c["agent"] for c in transport.calls]


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    monkeypatch.setenv("TRACE_STAGE3_BACKEND", "anthropic")  # ancien mode de l'étape 3 par API (LLM simulé)
    monkeypatch.setenv("TRACE_STAGE4_BACKEND", "anthropic")  # ancien mode de l'étape 4 par API (LLM simulé)
    transport = FakeTransport({**S4.stage3_responders(), ACCOUNTABILITY: S4.REFERENCE_BUILDER,
                               TRAJECTORY: S5.scripted_mapper(S5.REFERENCE_PLAN)})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    return transport


def app_with_run(run):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["last_run"] = run
    return at.run()


def stage5_table(at):
    return next(t.value for t in reversed(at.table) if "Configuration" in t.value.columns)


def test_stage5_waits_for_stage4(tmp_path, fake_api):
    at = app_with_run(si.make_ingested_run(tmp_path, S4.FILES))
    assert not at.exception
    assert "Étape 5 — Configuration et trajectoire intra-entretien" in texts(at.header)
    assert "Lancez d'abord l'étape 4" in texts(at.info)
    assert not [b for b in at.button if b.key == "traj_launch"]
    # étape 3 absente : la restauration de l'étape 4 renvoie d'abord vers celle de l'étape 3
    assert "Restaurer des résultats Stage 4 existants" in [e.label for e in at.expander]
    assert "restaurez-la (ou lancez-la) d'abord" in texts(at.info)


def test_full_flow_stage3_stage4_stage5_then_cache(tmp_path, fake_api):
    at = app_with_run(si.make_ingested_run(tmp_path, S4.FILES))
    at.button(key="ai_launch").click().run()
    at.button(key="acc_launch").click().run()
    assert not at.exception and agents(fake_api).count(TRAJECTORY) == 0  # l'étape 4 ne lance jamais l'étape 5
    launch = at.button(key="traj_launch")
    assert launch.label == "Construire la configuration intra-entretien (1 appel(s) API payant(s))"
    before = len(fake_api.calls)

    launch.click().run()
    assert not at.exception and agents(fake_api)[before:] == [TRAJECTORY]
    assert "Étape 5 terminée." in texts(at.success)
    table = stage5_table(at)
    assert list(table["Étape 5"]) == ["✅ SUCCESS"] and list(table["Statut étape 4"]) == ["disponible"]
    assert list(table["Configuration"]) == ["configuration contextuelle"]
    assert (list(table["Affirmations retenues"]), list(table["Exceptions"]), list(table["Zones ordinaires"]),
            list(table["Changements temporels explicites"]), list(table["Critères (métier d'étudiant)"])) == (
        [3], [1], [1], [0], [2])
    assert f"Configuration intra-entretien — {S4.INTERVIEW_ID} — aperçu" in [e.label for e in at.expander]
    labels = [b.label for b in at.get("download_button")]
    for name in ("student_trajectory.json", "student_trajectory_validation.json", "student_trajectory_manifest.json"):
        assert f"Télécharger {name}" in labels
    claims = [df for df in at.dataframe if "ancrages temporels" in df.value.columns]
    assert len(claims) == 1 and len(claims[0].value) == 3
    criteria = [df for df in at.dataframe if "critère" in df.value.columns]
    assert len(criteria) == 1 and len(criteria[0].value) == 2
    assert "Configuration et trajectoire intra-entretien** (IA) — terminé (1/1 entretien(s))" in texts(at.markdown)

    assert at.button(key="traj_launch").label.endswith("(0 appel(s) API payant(s))")
    at.button(key="traj_launch").click().run()
    assert not at.exception and agents(fake_api)[before:] == [TRAJECTORY]
    assert list(stage5_table(at)["Étape 5"]) == ["♻️ CACHED"]


def test_restored_stage3_and_stage4_lead_directly_to_stage5(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    transport = FakeTransport({TRAJECTORY: S5.scripted_mapper(S5.REFERENCE_PLAN)})  # seule l'étape 5 répond
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    before, _ = S5.run_to_stage4(tmp_path / "before", S4.FILES, S4.stage3_responders(), S4.REFERENCE_BUILDER)
    out = S5.analysis_dir(before)
    stage3 = [(n, (out / n).read_bytes()) for n in ("practice_extractor.json", "interaction_signals.json",
                                                     "evidence_validation.json", "speaker_attribution_audit.json")]
    stage4 = [(n, (out / n).read_bytes()) for n in (config.ACCOUNTABILITY_EPISODES_FILENAME,
                                                     config.ACCOUNTABILITY_VALIDATION_FILENAME)]
    run = restore_stage3(si.make_ingested_run(tmp_path / "after", S4.FILES), S4.INTERVIEW_ID, stage3)
    at = app_with_run(run)
    assert not at.exception
    assert "Restaurer des résultats Stage 4 existants" in [e.label for e in at.expander]
    assert at.button(key="restore4_launch").disabled  # aucun fichier choisi
    assert not [b for b in at.button if b.key == "traj_launch"]

    at.session_state["last_run"] = restore_stage4(run, S4.INTERVIEW_ID, stage4)
    at.run()
    assert "Stage 4 restauré depuis fichiers — 0 appel API" in texts(at.success)
    assert "Restaurer des résultats Stage 4 existants" not in [e.label for e in at.expander]
    assert at.button(key="traj_launch").label == "Construire la configuration intra-entretien (1 appel(s) API payant(s))"
    at.button(key="traj_launch").click().run()
    assert not at.exception and agents(transport) == [TRAJECTORY]  # ni étape 3, ni audit, ni étape 4
    assert list(stage5_table(at)["Étape 5"]) == ["✅ SUCCESS"]
    assert "(restaurée depuis fichiers)" in texts(at.markdown)


def test_temporal_interview_shows_the_explicit_change(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    run, _ = S5.case_to_stage4(tmp_path, S5.TEMPORAL)
    transport = FakeTransport({TRAJECTORY: S5.TEMPORAL.mapper()})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    at = app_with_run(run)
    at.button(key="traj_launch").click().run()
    assert not at.exception
    table = stage5_table(at)
    assert list(table["Configuration"]) == ["trajectoire temporelle explicite"]
    assert list(table["Changements temporels explicites"]) == [1]
    claims = next(df.value for df in at.dataframe if "ancrages temporels" in df.value.columns)
    assert "« Au lycée » (T0002) · « Maintenant » (T0004)" in list(claims["ancrages temporels"])


def test_requalified_configuration_is_shown(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-FAKE-KEY-000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fake-model")
    run, _ = S5.case_to_stage4(tmp_path, S5.NOPATTERN)
    transport = FakeTransport({TRAJECTORY: S5.NOPATTERN.mapper("adversarial")})
    monkeypatch.setattr(llm_client, "AnthropicTransport", lambda settings: transport)
    at = app_with_run(run)
    at.button(key="traj_launch").click().run()
    assert not at.exception
    assert list(stage5_table(at)["Changements temporels explicites"]) == [0]
    assert "requalifiée par TRACE" in texts(at.warning)
    warnings = next(df.value for df in at.dataframe if "code" in df.value.columns and "gravité" in df.value.columns)
    assert "TEMPORAL_CHANGE_REQUALIFIED" in list(warnings["code"])
