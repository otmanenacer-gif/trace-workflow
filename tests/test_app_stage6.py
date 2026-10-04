"""Interface Streamlit de l'étape 6 (AppTest : sans navigateur, faux Ollama derrière le vrai runner local)."""

import pytest
from streamlit.testing.v1 import AppTest

from core import config
from tests import synthetic_stage5 as S5
from tests import synthetic_stage6 as S6
from tests.fake_llm import COMPARATOR, use_fake_runtime

APP = str(config.PROJECT_ROOT / "app.py")


def texts(elements):
    return " ".join(str(e.value) for e in elements)


@pytest.fixture
def fake_agents(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CROSS_INTERVIEW_DIR", tmp_path / "cross_interview")
    return use_fake_runtime(monkeypatch, {COMPARATOR: S6.scripted_comparator(S6.GOOD_PLAN)})


def app(run=None):
    at = AppTest.from_file(APP, default_timeout=60)
    if run is not None:
        at.session_state["last_run"] = run
    return at.run()


def upload(at, ids):
    at.file_uploader(key="stage6_files").set_value([(n, d, "application/json") for n, d in S6.uploads(ids)]).run()
    return at


def corpus_table(at):
    return next(t.value for t in at.table if "interview_id" in t.value.columns and "importé" in t.value.columns)


def test_stage6_section_is_available_without_a_run(fake_agents):
    at = app()
    assert not at.exception
    assert "Étape 6 — Comparaison inter-entretiens" in texts(at.header)
    assert "Constituer le corpus Stage 6" in texts(at.subheader)
    assert "Aucune sortie de l'étape 5 importée." in texts(at.info)
    assert not [b for b in at.button if b.key in ("run_stage6", "stage6_launch")]
    assert ("6. Comparaison transversale** (IA) — prête, exécution locale (Ollama), sur import des sorties "
            "de l'étape 5") in texts(at.markdown)


def test_two_interviews_exploratory_then_no_replay(fake_agents):
    at = upload(app(), S6.IDS_2)
    assert not at.exception
    table = corpus_table(at)
    assert list(table["interview_id"]) == ["ENT_A", "ENT_B"] and list(table["statut"]) == ["✅ exploitable"] * 2
    assert list(table.columns) == ["interview_id", "statut", "claims", "needs_review", "configuration", "importé"]
    assert "**N importés : 2** · **N exploitables : 2** · mode : **EXPLORATOIRE (deux entretiens)**" in texts(
        at.markdown)
    assert "comparaison sera marquée EXPLORATOIRE" in texts(at.warning)
    at.button(key="run_stage6").click().run()
    assert not at.exception and len(fake_agents.calls) == 1
    assert "Étape 6 terminée (exécution locale, modèle qwen2.5:7b, 0 appel API)." in texts(at.success)
    assert "Dernière exécution (étape 6) :** 0 appel(s) API" in texts(at.markdown)
    assert "Comparaison EXPLORATOIRE" in texts(at.warning)
    labels = [b.label for b in at.get("download_button")]
    for name in ("cross_interview_comparison.json", "cross_interview_validation.json", "cross_interview_manifest.json"):
        assert f"Télécharger {name}" in labels
    at.button(key="run_stage6").click().run()  # corpus identique, déjà complet : non rejoué
    assert not at.exception and len(fake_agents.calls) == 1
    assert "Comparaison EXPLORATOIRE" in texts(at.warning)


def test_invalid_interview_is_excluded_with_its_reason(fake_agents):
    at = upload(app(), S6.IDS_4 + ["ENT_X_INVALIDE"])
    table = corpus_table(at)
    assert list(table["statut"]) == ["✅ exploitable"] * 4 + ["❌ exclu"]
    assert "**N importés : 5** · **N exploitables : 4**" in texts(at.markdown)
    assert "ENT_X_INVALIDE exclu : validation_error_count = 2" in texts(at.error)
    at.button(key="run_stage6").click().run()
    assert not at.exception and "N importés : **5** · N exploitables : **4**" in texts(at.markdown)


def test_single_interview_makes_stage6_not_applicable(fake_agents):
    at = upload(app(), ["ENT_A"])
    message = texts(at.info)
    assert "NOT_APPLICABLE_SINGLE_INTERVIEW" in message
    assert "La comparaison inter-entretiens nécessite au moins deux entretiens exploitables." in message
    assert "Étape 6 bloquée" not in texts(at.error)
    assert not [b for b in at.button if b.key == "run_stage6"] and fake_agents.calls == []


def test_seventeen_interviews_show_negative_cases_and_review(fake_agents):
    at = upload(app(), S6.IDS_17)
    at.button(key="run_stage6").click().run()
    assert not at.exception
    assert "cas négatifs : **3**" in texts(at.markdown) and "frontières récurrentes : **2**" in texts(at.markdown)
    negative = next(df.value for df in at.dataframe if "complique" in df.value.columns)
    assert list(negative["entretien"]) == ["ENT_D", "ENT_H", "ENT_L"]
    claims = next(df.value for df in at.dataframe if "contre-exemples" in df.value.columns)
    assert "ENT_D (cas contraire)" in list(claims["contre-exemples"])
    review = [r for r in claims["à revoir"] if "SUPPORTED_ONLY_BY_REVIEW_ITEMS" in r]
    assert len(review) == 1
    criteria = next(df.value for df in at.dataframe if "refus explicite" in df.value.columns)
    assert list(criteria["refus explicite"]) == ["ENT_H"] and "ENT_B" in criteria["non observé"][0]
    configurations = next(df.value for df in at.dataframe if "configuration" in df.value.columns
                          and "entretiens" in df.value.columns)
    assert "2 entretien(s) sur 17 exploitable(s)" in list(configurations["entretiens"])


def test_stage5_outputs_of_the_current_run_can_be_included(fake_agents, tmp_path):
    run, cache = S5.case_to_stage4(tmp_path / "run", S5.EXCEPTION)
    run, _ = S5.run_stage5(run, cache, S5.EXCEPTION.mapper())
    at = app(run)
    assert at.checkbox(key="stage6_include_run").value is True
    assert "**N importés : 1** · **N exploitables : 1**" in texts(at.markdown)  # seul l'entretien du run
    upload(at, S6.IDS_2)
    assert "**N importés : 3** · **N exploitables : 3**" in texts(at.markdown)
    assert S5.EXCEPTION.interview_id in list(corpus_table(at)["interview_id"])


def test_without_key_the_corpus_is_checked_and_nothing_is_launched_before_the_click(fake_agents):
    at = upload(app(), S6.IDS_2)
    assert not at.exception and "N exploitables : 2" in texts(at.markdown)
    assert [b for b in at.button if b.key == "run_stage6"] and fake_agents.calls == []
    assert not [b for b in at.button if b.key == "stage6_launch"]  # aucun ancien lanceur par API
