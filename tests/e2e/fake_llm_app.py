"""Lanceur de TEST : exécute app.py tel quel, mais avec un LLM SIMULÉ.

    streamlit run tests/e2e/fake_llm_app.py

Aucun appel réel, aucun token consommé : la clé et le modèle sont factices et
le transport HTTP est remplacé par tests/fake_llm.FakeTransport. Sert au test
navigateur (tests/e2e/browser_check.py) et à une démonstration locale du
parcours complet sans clé API. Ne jamais utiliser pour de vraies analyses.

Pour l'entretien synthétique (tests/synthetic_interviews.py), les réponses
simulées sont celles de la section 18, avec UNE citation volontairement
inventée côté Practice Extractor, afin de montrer la détection des fausses
citations. Pour tout autre entretien, les agents simulés ne renvoient rien.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake-browser-test"  # factice : jamais envoyée
os.environ["ANTHROPIC_MODEL"] = "fake-model"

import core.llm_client as llm_client  # noqa: E402
from core import config  # noqa: E402
from tests import synthetic_interviews as si  # noqa: E402
from tests.fake_llm import INTERACTION, PRACTICE, FakeTransport, text_response  # noqa: E402

if os.environ.get("TRACE_E2E_CACHE_DIR"):
    config.CACHE_DIR = Path(os.environ["TRACE_E2E_CACHE_DIR"])


def _is_synthetic(params: dict) -> bool:
    return si.tid(2) in params["messages"][0]["content"]


def _practices(params):
    if _is_synthetic(params):
        return text_response(si.with_fabricated_quote(si.GOOD_PRACTICES, "practices"), input_tokens=2210,
                             output_tokens=1480)
    return text_response({"practices": [], "extraction_notes": None}, input_tokens=300, output_tokens=20)


def _signals(params):
    if _is_synthetic(params):
        return text_response(si.GOOD_SIGNALS, input_tokens=2370, output_tokens=1730)
    return text_response({"signals": [], "reading_notes": None}, input_tokens=300, output_tokens=20)


llm_client.AnthropicTransport = lambda settings: FakeTransport({PRACTICE: _practices, INTERACTION: _signals})

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
