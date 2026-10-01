"""Correctif de segmentation des locuteurs (bug « OTMANE ») : marqueurs explicites en milieu de ligne.

Le vrai entretien ayant révélé le bug n'est PAS versionné : le fixture ci-dessous en reproduit seulement la
forme (deux très longs paragraphes contenant chacun de nombreux « Enquêteur : » / « Enquêté : »), avec un
texte entièrement synthétique. Aucun appel réseau, aucun appel API (LLM simulé).
"""

import re
import unicodedata
from pathlib import Path

import pytest

import core.transcript_structurer as structurer
from core import config, ingestion_validator
from core.analysis import analyze_run, plan_analysis, prepare_interview
from core.analysis_cache import AnalysisCache
from core.schemas import SPEAKER_INTERVIEWEE, SPEAKER_INTERVIEWER
from core.transcript_structurer import split_inline_markers, structure_transcript
from tests import synthetic_interviews as si
from tests.fake_llm import INTERACTION, PRACTICE, FakeTransport, fake_settings, text_response

OTMANE_FILENAME = "Entretien_paragraphes.txt"
OTMANE_ID = "ENTRETIEN_PARAGRAPHES"

QUESTIONS = ["Est-ce que tu utilises ChatGPT pour tes cours ?", "Et pour les dissertations, comment tu fais ?",
             "Tu peux me donner un exemple précis ?", "Et tes professeurs, ils en pensent quoi ?",
             "Est-ce qu'il t'arrive de vérifier ce qu'il répond ?", "Et en dehors des études ?"]
ANSWERS = ["Oui, surtout pour reformuler mes phrases quand elles sont trop lourdes.",
           "Je fais le plan moi-même, euh, et après je lui demande juste de relire.",
           "La semaine dernière, pour un exposé d'économie, je lui ai demandé des sources.",
           "Ma prof dit qu'on peut l'utiliser mais qu'il faut le citer, donc voilà.",
           "Ben oui, toujours, parce qu'il invente parfois des références.",
           "Pour les recettes de cuisine, honnêtement c'est pratique."]
# Variantes de libellés rencontrées dans un même document (casse, accents, anglicismes).
INTERVIEWER_MARKERS = ["Enquêteur :", "Enqueteur :", "ENQUÊTEUR :", "Interviewer :", "Enquêteur:"]
INTERVIEWEE_MARKERS = ["Enquêté :", "Enquete :", "enquêté :", "Interviewé :", "Enquêtée :"]
EXCHANGES = 40   # 40 échanges par paragraphe, 2 paragraphes : 160 prises de parole
MISLABELED = "Enquêteur : Moi personnellement je lui fais écrire mes introductions, je l'avoue."


def otmane_exchanges() -> list[tuple[str, str, str]]:
    """(marqueur, rôle attendu, texte exact) dans l'ordre du document."""
    items = []
    for paragraph in range(2):
        for k in range(EXCHANGES):
            n = paragraph * EXCHANGES + k
            items.append((INTERVIEWER_MARKERS[n % 5], SPEAKER_INTERVIEWER, QUESTIONS[n % 6]))
            items.append((INTERVIEWEE_MARKERS[n % 5], SPEAKER_INTERVIEWEE, ANSWERS[n % 6] + f" (échange {n + 1})"))
    return items


def otmane_text() -> str:
    """Deux paragraphes (deux lignes) : toute la conversation est écrite au fil du texte."""
    items = otmane_exchanges()
    half = len(items) // 2
    paragraphs = [" ".join(f"{m} {t}" for m, _, t in items[:half]),
                  " ".join(f"{m} {t}" for m, _, t in items[half:])]
    # un marqueur mal attribué (réponse à la première personne marquée « Enquêteur ») dans le second paragraphe
    paragraphs[1] = paragraphs[1].replace(f"{items[half + 1][0]} {items[half + 1][2]}", MISLABELED, 1)
    return "\n\n".join(paragraphs) + "\n"


def structure(text: str, interview_id: str = "TEST"):
    return structure_transcript([{"page": None, "text": text}], interview_id, "test.txt")


def non_blank(text: str) -> str:
    return re.sub(r"\s+", "", text)


# --- Le bug « OTMANE » ---------------------------------------------------------------------------

def old_behaviour(monkeypatch):
    """Ancien comportement : aucune coupure en milieu de ligne."""
    monkeypatch.setattr(structurer, "split_inline_markers", lambda line: [line])


def test_otmane_fixture_reproduces_the_bug_with_the_old_segmentation(monkeypatch):
    old_behaviour(monkeypatch)
    turns = structure(otmane_text())["turns"]
    assert len(turns) == 2 and all(len(t["text"]) > 4000 for t in turns)


def test_otmane_fixture_now_gives_one_turn_per_speaker_change():
    text = otmane_text()
    result = structure(text)
    turns = result["turns"]
    expected = otmane_exchanges()
    assert len(turns) == len(expected) == 4 * EXCHANGES  # 160 tours au lieu de 2
    assert [t["turn_id"] for t in turns] == [f"TEST_T{i:04d}" for i in range(1, len(turns) + 1)]
    # rôles d'après les libellés écrits ; textes recopiés à l'identique, dans l'ordre
    half = len(expected) // 2
    for i, (turn, (marker, role, sentence)) in enumerate(zip(turns, expected)):
        if i == half + 1:  # libellé « Enquêteur » écrit devant une réponse : conservé tel quel, jamais corrigé
            assert turn["speaker"] == SPEAKER_INTERVIEWER and turn["marker"] == "Enquêteur :"
            assert turn["text"] == MISLABELED.removeprefix("Enquêteur : ")
            continue
        assert turn["speaker"] == role, (i, turn["marker"])
        assert turn["marker"] == marker and turn["text"] == sentence
    # aucun contenu perdu ni ajouté : marqueurs + textes = texte d'origine (hors blancs)
    assert non_blank("".join(t["marker"] + t["text"] for t in turns)) == non_blank(text)
    # chaque tour garde la ligne (le paragraphe) d'où il vient
    assert {t["source"]["line_start"] for t in turns[:half]} == {1}
    assert {t["source"]["line_start"] for t in turns[half:]} == {3}
    warning = next(w for w in result["warnings"] if w["code"] == "INLINE_SPEAKER_LABELS_SPLIT")
    assert warning["severity"] == "info" and warning["count"] == len(expected) - 2


def test_otmane_old_segmentation_would_now_be_blocked_by_the_guard(tmp_path, monkeypatch):
    old_behaviour(monkeypatch)
    run = si.make_ingested_run(tmp_path, [(OTMANE_FILENAME, otmane_text().encode("utf-8"))])
    ingestion = run["files"][0]["ingestion"]
    assert ingestion["status"] == "FAIL"
    report = (Path(ingestion["output_dir"]) / config.INGESTION_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "OVERSIZED_TURN_WITH_INTERNAL_MARKERS" in report
    from core.analysis import eligible_files
    assert eligible_files(run) == []  # aucun tour géant n'est jamais envoyé à un modèle


def test_otmane_ingestion_now_passes_and_invalidates_stage3_caches(tmp_path, monkeypatch):
    files = [(OTMANE_FILENAME, otmane_text().encode("utf-8"))]
    empty = {PRACTICE: lambda p: text_response({"practices": [], "extraction_notes": None}),
             INTERACTION: lambda p: text_response({"signals": [], "reading_notes": None})}

    with monkeypatch.context() as m:  # ancienne segmentation, garde-fou désactivé pour comparer les caches
        old_behaviour(m)
        m.setattr(ingestion_validator, "OVERSIZED_TURN_CHARS", 10 ** 9)
        old_run = si.make_ingested_run(tmp_path / "old", files)
        old_prepared = prepare_interview(old_run["files"][0]["ingestion"])
        old_sha = (Path(old_run["files"][0]["ingestion"]["output_dir"]) /
                   config.STRUCTURED_TRANSCRIPT_FILENAME).read_bytes()
        cache = AnalysisCache(tmp_path / "cache")
        analyze_run(old_run, settings=fake_settings(), transport=FakeTransport(empty), cache=cache)

    run = si.make_ingested_run(tmp_path / "new", files)
    ingestion = run["files"][0]["ingestion"]
    assert ingestion["status"] == "PASS" and ingestion["turn_count"] == 4 * EXCHANGES
    new_transcript = (Path(ingestion["output_dir"]) / config.STRUCTURED_TRANSCRIPT_FILENAME).read_bytes()
    assert new_transcript != old_sha
    prepared = prepare_interview(ingestion)
    assert prepared.source_sha256 == old_prepared.source_sha256       # même fichier source…
    assert prepared.transcript_sha256 != old_prepared.transcript_sha256  # …mais nouvelle représentation
    plan = plan_analysis(run, [ingestion["interview_id"]], fake_settings(), cache)
    assert plan["cached"] == 0 and plan["calls"] >= 2  # l'ancien cache de l'étape 3 ne s'applique plus
    # le fichier source et le texte brut restent inchangés
    raw = (Path(ingestion["output_dir"]) / config.RAW_TEXT_FILENAME).read_text(encoding="utf-8")
    assert non_blank(raw) == non_blank(otmane_text())


# --- Règles de découpage -------------------------------------------------------------------------

def test_thirty_alternations_on_one_line_give_thirty_turns():
    line = " ".join(f"Enquêteur : Question {i} ? Enquêté : Réponse {i}." for i in range(1, 16))
    turns = structure(line)["turns"]
    assert len(turns) == 30
    assert [t["speaker"] for t in turns] == [SPEAKER_INTERVIEWER, SPEAKER_INTERVIEWEE] * 15
    assert [t["text"] for t in turns[:4]] == ["Question 1 ?", "Réponse 1.", "Question 2 ?", "Réponse 2."]


@pytest.mark.parametrize("label, speaker", [
    ("Enquêteur", SPEAKER_INTERVIEWER), ("Enqueteur", SPEAKER_INTERVIEWER), ("ENQUETEUR", SPEAKER_INTERVIEWER),
    ("enquêteur", SPEAKER_INTERVIEWER), ("Enquêtrice", SPEAKER_INTERVIEWER), ("Interviewer", SPEAKER_INTERVIEWER),
    ("Enquêté", SPEAKER_INTERVIEWEE), ("Enquete", SPEAKER_INTERVIEWEE), ("ENQUÊTÉE", SPEAKER_INTERVIEWEE),
    ("enquêté", SPEAKER_INTERVIEWEE), ("Interviewé", SPEAKER_INTERVIEWEE), ("Interviewee", SPEAKER_INTERVIEWEE),
])
def test_inline_label_variants_after_sentence_end(label, speaker):
    turns = structure(f"Enquêteur : Bonjour. {label} : Texte du tour.")["turns"]
    assert len(turns) == 2 and turns[1]["speaker"] == speaker and turns[1]["text"] == "Texte du tour."


def test_decomposed_accents_are_split_and_kept_verbatim():
    text = unicodedata.normalize("NFD", "Enquêteur : Ça va ? Enquêté : Oui, très bien.")
    turns = structure(text)["turns"]
    assert len(turns) == 2 and turns[1]["text"] == unicodedata.normalize("NFD", "Oui, très bien.")


def test_markers_at_line_start_and_after_line_breaks_still_work():
    text = "Enquêteur : Première question ?\n\nEnquêté : Réponse\nsur deux lignes.\n  Enquêteur : Suite ?\nEnquêté : Fin."
    turns = structure(text)["turns"]
    assert [t["text"] for t in turns] == ["Première question ?", "Réponse\nsur deux lignes.", "Suite ?", "Fin."]


def test_capitalized_marker_without_punctuation_is_split():
    turns = structure("Enquêté : ok bon voilà Enquêteur : Et ensuite ?")["turns"]
    assert [t["text"] for t in turns] == ["ok bon voilà", "Et ensuite ?"]


@pytest.mark.parametrize("sentence", [
    "Enquêté : le rôle de l'enquêteur : poser des questions.",       # nom commun, minuscule, sans fin de phrase
    "Enquêté : j'ai lu le guide de l'Enquêteur : très utile.",         # élision : jamais un marqueur
    "Enquêté : ma réponse : non. Question : pourquoi pas ?",            # libellés ambigus : jamais découpés
    "Enquêté : participant : moi aussi.",
])
def test_ordinary_words_followed_by_a_colon_are_not_split(sentence):
    assert len(structure(sentence)["turns"]) == 1


def test_split_pieces_rebuild_the_exact_line():
    line = "Enquêteur : A ? Enquêté : B. (rires) Enquêteur (Marie) : C ? Enquêté 2 : D."
    pieces = split_inline_markers(line)
    assert "".join(pieces) == line and len(pieces) == 4
    turns = structure(line)["turns"]
    assert [t["marker"] for t in turns] == ["Enquêteur :", "Enquêté :", "Enquêteur (Marie) :", "Enquêté 2 :"]


def test_oversized_turn_guard_needs_both_size_and_internal_markers():
    big = "mot " * 1200
    assert ingestion_validator.check_oversized_turns([{"turn_id": "X_T0001", "text": big}]) == []
    with_markers = big + " Question : a ? Réponse : b."
    [warning] = ingestion_validator.check_oversized_turns([{"turn_id": "X_T0001", "text": with_markers}])
    assert warning["code"] == "OVERSIZED_TURN_WITH_INTERNAL_MARKERS" and warning["severity"] == "error"
    short = "Question : a ? Réponse : b."
    assert ingestion_validator.check_oversized_turns([{"turn_id": "X_T0001", "text": short}]) == []
