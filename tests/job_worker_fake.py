"""Processus DÉTACHÉ de test : le vrai `core.local_jobs.execute`, avec le faux Ollama (aucun modèle, aucun réseau).

    python tests/job_worker_fake.py DOSSIER_DU_TRAVAIL DOSSIER_DU_CACHE DÉLAI_PAR_APPEL
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config, local_jobs, local_pipeline  # noqa: E402
from tests import synthetic_long_interview as L  # noqa: E402
from tests.fake_llm import (INTERACTION, LONG_DISTANCE, PRACTICE, FakeLocalAgentRunner, FakeOllama,  # noqa: E402
                            text_response)


def slow(handler, delay):
    def respond(params):
        time.sleep(delay)
        return handler(params)
    return respond


def responders(delay: float = 0.0) -> dict:
    return {PRACTICE: slow(lambda p: text_response({"practices": [], "extraction_notes": None}), delay),
            INTERACTION: slow(L.simulated_chunk_reader, delay),
            LONG_DISTANCE: slow(L.simulated_long_distance_reader, delay)}


def main() -> int:
    job_dir, cache_dir, delay = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    config.CACHE_DIR = cache_dir
    ollama = FakeOllama(responders(delay))
    local_pipeline.make_runner = lambda settings, journal_dir=None, checker=None: FakeLocalAgentRunner(
        settings=settings, ollama=ollama, checker=checker, journal_dir=journal_dir)
    local_jobs.execute(job_dir, log_to_file=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
