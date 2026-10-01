"""Lanceur de TEST : exécute app.py tel quel, mais avec un LLM SIMULÉ.

    streamlit run tests/e2e/fake_llm_app.py

Aucun appel réel, aucun token consommé : la clé et le modèle sont factices et
le transport HTTP est remplacé par tests/fake_llm.FakeTransport. Sert au test
navigateur (tests/e2e/browser_check.py) et à une démonstration locale du
parcours complet sans clé API. Ne jamais utiliser pour de vraies analyses.

Pour l'entretien synthétique (tests/synthetic_interviews.py), les réponses
simulées sont celles de la section 18, avec UNE citation volontairement
inventée côté Practice Extractor, afin de montrer la détection des fausses
citations. Pour l'entretien synthétique de l'étape 3.5 (Entretien_etape_3_5.txt,
dont un tour est volontairement mal attribué), les trois réponses simulées
(audit des locuteurs, pratiques, signaux) sont celles de tests/test_stage3_5_synthetic.py.
Pour l'entretien long synthétique de l'étape 3.6 (Entretien_long.txt, 349 tours), chaque
bloc et la lecture à longue distance sont simulés par tests/synthetic_long_interview.py.
Pour l'entretien long synthétique de l'étape 3.7 (Entretien_etape_3_7.txt, 330 tours), les blocs
Practice Extractor et Interaction Reader (lecteur qui SURCODE : un signal par remplisseur) et la
lecture à longue distance sont simulés par tests/synthetic_stage37.py.
Pour les entretiens synthétiques de l'étape 4 (Entretien_etape_4.txt, 25 tours, et
Entretien_etape_4_long.txt, 320 tours), l'étape 3 et l'Accountability Episode Builder sont simulés par
tests/synthetic_stage4.py et tests/synthetic_stage4_long.py (lecteur déterministe des candidats envoyés).
Pour l'entretien de régression de l'étape 4.1 (Regression_otmane.txt, 388 tours), voir
tests/synthetic_stage4_otmane.py. Pour les entretiens synthétiques de l'étape 5 (tests/synthetic_stage5.py : A à H, et
tests/synthetic_stage5_long.py : Etape5_long.txt, 190 tours), les étapes 3 et 4 et le Trajectory Mapper sont simulés par ces
modules ; pour l'entretien de référence de l'étape 4, le Trajectory Mapper suit synthetic_stage5.REFERENCE_PLAN.
Pour tout autre entretien, les agents simulés de l'étape 3 ne renvoient rien, l'étape 4 simulée
classe chaque candidat « incertain » et l'étape 5 simulée ne décrit aucune configuration.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake-browser-test"  # factice : jamais envoyée
os.environ["ANTHROPIC_MODEL"] = "fake-model"
# Garde-fou : si un rechargement à chaud (fichier source modifié pendant le test, voir app.py) réimporte
# core.llm_client, le transport réel réapparaît ; il viserait alors une adresse locale fermée et échouerait
# sans jamais joindre l'API.
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"

import core.llm_client as llm_client  # noqa: E402
from core import config  # noqa: E402
from tests import synthetic_interviews as si  # noqa: E402
from tests import synthetic_long_interview as long_interview  # noqa: E402
from tests import synthetic_stage37 as stage37  # noqa: E402
from tests import synthetic_stage4 as stage4  # noqa: E402
from tests import synthetic_stage4_long as stage4_long  # noqa: E402
from tests import synthetic_stage4_otmane as otmane  # noqa: E402
from tests import synthetic_stage5 as stage5  # noqa: E402
from tests import synthetic_stage5_long as stage5_long  # noqa: E402
from tests.fake_llm import (ACCOUNTABILITY, AUDITOR, INTERACTION, LONG_DISTANCE, PRACTICE, TRAJECTORY,  # noqa: E402
                            FakeTransport, text_response)

if os.environ.get("TRACE_E2E_CACHE_DIR"):
    config.CACHE_DIR = Path(os.environ["TRACE_E2E_CACHE_DIR"])


def _is_synthetic(params: dict) -> bool:
    return si.tid(2) in params["messages"][0]["content"]


def _is_stage35(params: dict) -> bool:
    # le tour mal attribué figure dans l'entretien complet ET dans l'extrait envoyé à l'auditeur
    return si.STAGE35_MISATTRIBUTED_TURN in params["messages"][0]["content"]


def _is_stage37(params: dict) -> bool:
    return stage37.INTERVIEW_ID in params["messages"][0]["content"]


def _is_stage4(params: dict) -> bool:
    content = params["messages"][0]["content"]
    # étape 3 : identifiants complets ; étape 4 (représentation normalisée) : préfixe déclaré dans id_prefix
    return stage4.INTERVIEW_ID + "_T" in content or f'"id_prefix":"{stage4.INTERVIEW_ID}_"' in content


def _is_stage4_long(params: dict) -> bool:
    return stage4_long.INTERVIEW_ID in params["messages"][0]["content"]


def _is_otmane(params: dict) -> bool:
    return otmane.INTERVIEW_ID in params["messages"][0]["content"]


_STAGE4_REFERENCE = stage4.stage3_responders()
_OTMANE = otmane.stage3_responders()
_STAGE5_LONG = stage5_long.stage3_responders()


def _stage5_case(params: dict):
    """Cas synthétique de l'étape 5 (A à H) dont l'identifiant figure dans le message, sinon None."""
    content = params["messages"][0]["content"]
    return next((case for case in stage5.CASES.values() if case.interview_id + "_" in content), None)


def _is_stage5_long(params: dict) -> bool:
    return stage5_long.INTERVIEW_ID + "_" in params["messages"][0]["content"]


def _stage5_stage3(agent: str):
    def respond(params):
        if _is_stage5_long(params):
            return _STAGE5_LONG[agent](params)
        case = _stage5_case(params)
        return case.stage3_responders()[agent](params) if case else None
    return respond


def _practices(params):
    if (stage5_response := _stage5_stage3(PRACTICE)(params)) is not None:
        return stage5_response
    if _is_otmane(params):
        return _OTMANE[PRACTICE](params)
    if _is_stage4_long(params):
        return stage4_long.practice_reader(params)
    if _is_stage4(params):
        return _STAGE4_REFERENCE[PRACTICE](params)
    if _is_stage37(params):
        return stage37.practice_reader(params)
    if _is_stage35(params):
        return text_response(si.STAGE35_PRACTICES, input_tokens=2100, output_tokens=1200)
    if _is_synthetic(params):
        return text_response(si.with_fabricated_quote(si.GOOD_PRACTICES, "practices"), input_tokens=2210,
                             output_tokens=1480)
    return text_response({"practices": [], "extraction_notes": None}, input_tokens=300, output_tokens=20)


def _is_long(params: dict) -> bool:
    return long_interview.LONG_INTERVIEW_ID in params["messages"][0]["content"]


def _signals(params):
    if (stage5_response := _stage5_stage3(INTERACTION)(params)) is not None:
        return stage5_response
    if _is_otmane(params):
        return _OTMANE[INTERACTION](params)
    if _is_stage4_long(params):
        return stage4_long.signal_reader(params)
    if _is_stage4(params):
        return _STAGE4_REFERENCE[INTERACTION](params)
    if _is_stage37(params):
        return stage37.naive_reader(params)
    if _is_long(params):
        return long_interview.simulated_chunk_reader(params)
    if _is_stage35(params):
        return text_response(si.STAGE35_SIGNALS, input_tokens=2200, output_tokens=900)
    if _is_synthetic(params):
        return text_response(si.GOOD_SIGNALS, input_tokens=2370, output_tokens=1730)
    return text_response({"signals": [], "reading_notes": None}, input_tokens=300, output_tokens=20)


def _audit(params):
    if (stage5_response := _stage5_stage3(AUDITOR)(params)) is not None:
        return stage5_response
    if _is_stage4(params):
        return _STAGE4_REFERENCE[AUDITOR](params)
    if _is_stage35(params):
        return text_response(si.STAGE35_AUDIT, input_tokens=600, output_tokens=150)
    return text_response({"assessments": [], "audit_notes": None}, input_tokens=300, output_tokens=20)


def _long_distance(params):
    if _is_stage5_long(params):
        return _STAGE5_LONG[LONG_DISTANCE](params)
    if _is_otmane(params):
        return _OTMANE[LONG_DISTANCE](params)
    if _is_stage4_long(params):
        return stage4_long.long_distance_reader(params)
    if _is_stage37(params):
        return stage37.long_distance_reader(params)
    return long_interview.simulated_long_distance_reader(params)


_GENERIC_STAGE4 = stage4.scripted_builder({})


_OTMANE_BUILDER = otmane.builder("good")


def _accountability(params):
    if _is_stage5_long(params):
        return stage5_long.BUILDER(params)
    if (case := _stage5_case(params)) is not None:
        return case.builder()(params)
    if _is_otmane(params):
        return _OTMANE_BUILDER(params)
    if _is_stage4_long(params):
        return stage4_long.BUILDER(params)
    if _is_stage4(params):
        return stage4.REFERENCE_BUILDER(params)
    return _GENERIC_STAGE4(params)


_GENERIC_STAGE5 = stage5.scripted_mapper(stage5.generic_plan())
_REFERENCE_STAGE5 = stage5.scripted_mapper(stage5.REFERENCE_PLAN)


def _trajectory(params):
    if _is_stage5_long(params):
        return stage5_long.mapper()(params)
    if (case := _stage5_case(params)) is not None:
        return case.mapper()(params)
    if _is_stage4(params):
        return _REFERENCE_STAGE5(params)
    return _GENERIC_STAGE5(params)


llm_client.AnthropicTransport = lambda settings: FakeTransport(
    {PRACTICE: _practices, INTERACTION: _signals, AUDITOR: _audit, LONG_DISTANCE: _long_distance,
     ACCOUNTABILITY: _accountability, TRAJECTORY: _trajectory})

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
