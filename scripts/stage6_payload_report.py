"""Mesure de la représentation envoyée au Cross-Interview Comparator (étape 6). AUCUN appel API.

    python scripts/stage6_payload_report.py

Corpus synthétiques de 2, 8 et 17 entretiens (tests/synthetic_stage6.py, sorties d'étape 5 produites par le vrai
orchestrateur avec un LLM simulé), puis projection : 17 entretiens de la taille de l'entretien long
« OTMANE-like » (tests/synthetic_stage5_long.py, étapes 3 à 5 simulées).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import cross_interview as X  # noqa: E402
from core import cross_interview_material as cm  # noqa: E402
from core.interaction_chunking import CHARS_PER_TOKEN  # noqa: E402
from tests import synthetic_stage5 as S5  # noqa: E402
from tests import synthetic_stage5_long as L  # noqa: E402
from tests import synthetic_stage6 as S6  # noqa: E402


def row(label: str, uploads) -> None:
    prepared = X.prepare_stage6(uploads)
    raw = cm.raw_size(prepared.checked["usable"])
    request = prepared.request
    summary = prepared.material["summary"]
    print(f"{label:>22} | {prepared.checked['n_usable']:>2} entretiens | {summary['claim_count']:>3} affirmations | "
          f"{summary['criterion_count']:>3} critères | représentation {request['payload_chars']:>6} car. | message "
          f"≈ {request['estimated_tokens']:>6} tokens | JSON étape 5 bruts {raw:>7} car. "
          f"({request['payload_chars'] / raw:.1%}) | appels : 1")


def otmane_like_chunk() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        run, cache = S5.run_to_stage4(Path(tmp), L.files(), L.stage3_responders(), L.BUILDER)
        run, _ = S5.run_stage5(run, cache, L.mapper())
        directory = S5.analysis_dir(run)
        uploads = [(f"{L.INTERVIEW_ID}_{n}", (directory / n).read_bytes()) for n in S6.STAGE5_NAMES]
    prepared = X.prepare_stage6(uploads + S6.uploads(["ENT_A"]))
    payload = cm.build_payload(prepared.material, [L.INTERVIEW_ID])
    return len(cm.serialize_payload(payload))


def main() -> int:
    for ids in (S6.IDS_2, S6.IDS_8, S6.IDS_17):
        row(f"{len(ids)} entretiens", S6.uploads(ids))
    chunk = otmane_like_chunk()
    projected = 17 * chunk
    print(f"Projection : 17 entretiens « OTMANE-like » ≈ {projected} car. ≈ {round(projected / CHARS_PER_TOKEN)} tokens "
          f"(représentation seule ; seuil d'un appel unique : {X.SINGLE_CALL_MAX_INPUT_TOKENS} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
