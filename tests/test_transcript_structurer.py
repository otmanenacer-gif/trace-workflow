"""Tests du découpage en tours de parole (règles explicites, textes synthétiques)."""

import unicodedata

import pytest

from core.schemas import SPEAKER_INTERVIEWEE, SPEAKER_INTERVIEWER, SPEAKER_UNKNOWN
from core.transcript_structurer import make_interview_id, match_speaker_label, structure_transcript


def structure(text, page=None, interview_id="TEST", source="test.txt"):
    return structure_transcript([{"page": page, "text": text}], interview_id, source)


def codes(result):
    return [w["code"] for w in result["warnings"]]


# --- Alternance et variantes de libellés ------------------------------------------

def test_alternation_enqueteur_enquete():
    result = structure("Enquêteur : Bonjour.\nEnquêté : Salut.\nEnquêteur : Ça va ?\nEnquêté : Oui.")
    turns = result["turns"]
    assert [t["speaker"] for t in turns] == [SPEAKER_INTERVIEWER, SPEAKER_INTERVIEWEE] * 2
    assert [t["text"] for t in turns] == ["Bonjour.", "Salut.", "Ça va ?", "Oui."]
    assert turns[0]["speaker_raw"] == "Enquêteur" and turns[0]["marker"] == "Enquêteur :"
    assert result["warnings"] == []


@pytest.mark.parametrize("label, speaker", [
    ("Enquêteur", SPEAKER_INTERVIEWER), ("enquêteur", SPEAKER_INTERVIEWER), ("ENQUÊTEUR", SPEAKER_INTERVIEWER),
    ("Enqueteur", SPEAKER_INTERVIEWER), ("Enquêtrice", SPEAKER_INTERVIEWER), ("Intervieweur", SPEAKER_INTERVIEWER),
    ("Interviewer", SPEAKER_INTERVIEWER), ("Question", SPEAKER_INTERVIEWER), ("Q", SPEAKER_INTERVIEWER),
    ("Enquêté", SPEAKER_INTERVIEWEE), ("Enquêtée", SPEAKER_INTERVIEWEE), ("ENQUETE", SPEAKER_INTERVIEWEE),
    ("Interviewé", SPEAKER_INTERVIEWEE), ("Interviewee", SPEAKER_INTERVIEWEE), ("Participant", SPEAKER_INTERVIEWEE),
    ("Participante", SPEAKER_INTERVIEWEE), ("Réponse", SPEAKER_INTERVIEWEE), ("Reponse", SPEAKER_INTERVIEWEE),
    ("R", SPEAKER_INTERVIEWEE),
])
@pytest.mark.parametrize("separator", [" : ", ":", "  :  ", "\t: ", " ："])
def test_label_variants(label, speaker, separator):
    match = match_speaker_label(f"{label}{separator}texte")
    assert match is not None
    assert match["speaker"] == speaker
    assert match["speaker_raw"] == label
    assert match["rest"].strip() == "texte"


@pytest.mark.parametrize("line, speaker_raw", [
    ("Participant 2 : oui", "Participant 2"),
    ("Enquêteur (Marie) : oui", "Enquêteur (Marie)"),
    ("   Enquêté : oui", "Enquêté"),
])
def test_label_qualifiers_and_indentation(line, speaker_raw):
    assert match_speaker_label(line)["speaker_raw"] == speaker_raw


def test_decomposed_unicode_label():
    line = unicodedata.normalize("NFD", "Enquêté : déjà")
    match = match_speaker_label(line)
    assert match["speaker"] == SPEAKER_INTERVIEWEE
    assert match["rest"].strip() == unicodedata.normalize("NFD", "déjà")


@pytest.mark.parametrize("line", [
    "Il m'a dit : « fais-le toi-même ».",
    "À 14:30 j'avais cours.",
    "Remarque : c'était pas simple.",
    "Enquête : usages de l'IA",            # titre, pas un locuteur
    "Questionnaire : rempli",
    "La question : est-ce que c'est grave ?",
    "Bref: voilà",
])
def test_colon_in_content_is_not_a_label(line):
    assert match_speaker_label(line) is None


def test_colon_inside_turn_does_not_split():
    result = structure("Enquêteur : Et alors ?\nEnquêté : Il m'a dit : bon.\nRemarque : à 14:30 c'était fini.")
    assert len(result["turns"]) == 2
    assert result["turns"][1]["text"] == "Il m'a dit : bon.\nRemarque : à 14:30 c'était fini."


# --- Contenu préservé ---------------------------------------------------------------

def test_multiline_turn_preserved_verbatim():
    text = "Enquêteur : Question ?\nEnquêté : Première ligne\n\n   deuxième   ligne\ntroisième\nEnquêteur : Ok."
    turns = structure(text)["turns"]
    assert turns[1]["text"] == "Première ligne\n\n   deuxième   ligne\ntroisième"
    assert turns[1]["source"]["line_start"] == 2 and turns[1]["source"]["line_end"] == 5


def test_orality_accents_apostrophes_hesitations_repetitions_kept():
    oral = "Euh… bah j’sais pas, j'sais pas, genre c'était — euh — chiant, putain, ouais ouais."
    turns = structure(f"Enquêteur : Pourquoi ?\nEnquêté : {oral}")["turns"]
    assert turns[1]["text"] == oral


def test_text_before_first_label():
    result = structure("Entretien du 3 mars\nLieu : café\n\nDurée : 1h\nEnquêteur : Bonjour.\nEnquêté : Salut.")
    turns = result["turns"]
    assert [t["speaker"] for t in turns] == [SPEAKER_UNKNOWN, SPEAKER_UNKNOWN, SPEAKER_INTERVIEWER, SPEAKER_INTERVIEWEE]
    assert turns[0]["text"] == "Entretien du 3 mars\nLieu : café"
    assert turns[1]["text"] == "Durée : 1h"
    assert turns[0]["speaker_raw"] is None and turns[0]["marker"] is None
    assert "TEXT_BEFORE_FIRST_LABEL" in codes(result)


def test_no_label_at_all():
    result = structure("Un texte libre.\nSans locuteur.\n\nDeuxième paragraphe.")
    assert [t["speaker"] for t in result["turns"]] == [SPEAKER_UNKNOWN, SPEAKER_UNKNOWN]
    assert codes(result) == ["NO_SPEAKER_LABEL_DETECTED"]


def test_recurrent_unknown_label_is_unknown_speaker():
    result = structure("Enquêteur : Tu fais quoi ?\nEloïse : Bah rien.\nEnquêteur : Vraiment ?\nEloïse : Ouais.")
    turns = result["turns"]
    assert [t["speaker"] for t in turns] == [SPEAKER_INTERVIEWER, SPEAKER_UNKNOWN] * 2
    assert turns[1]["speaker_raw"] == "Eloïse" and turns[1]["text"] == "Bah rien."
    warning = next(w for w in result["warnings"] if w["code"] == "UNRECOGNIZED_SPEAKER_LABEL")
    assert warning["count"] == 2 and warning["turn_ids"] == ["TEST_T0002", "TEST_T0004"]


def test_single_unknown_label_does_not_split():
    turns = structure("Enquêté : Bon.\nMarc : il m'a dit ça.")["turns"]
    assert len(turns) == 1 and turns[0]["text"] == "Bon.\nMarc : il m'a dit ça."


def test_inline_label_is_flagged_not_split():
    result = structure("Enquêteur : Tu viens ? Enquêté : Oui.\nEnquêté : Enfin non.")
    assert len(result["turns"]) == 2
    assert "POSSIBLE_INLINE_SPEAKER_LABEL" in codes(result)


def test_empty_turn_kept():
    result = structure("Enquêteur :\nEnquêté : Oui.")
    assert result["turns"][0]["text"] == "" and "EMPTY_TURN" in codes(result)


def test_single_role_warning():
    assert "SINGLE_ROLE_ONLY" in codes(structure("Enquêté : a\nEnquêté : b"))


# --- Traçabilité -----------------------------------------------------------------------

def test_turn_ids_unique_ordered_and_citable():
    text = "\n".join(f"{'Enquêteur' if i % 2 == 0 else 'Enquêté'} : réplique {i}" for i in range(12))
    turns = structure(text, interview_id="ELOISE")["turns"]
    ids = [t["turn_id"] for t in turns]
    assert ids[0] == "ELOISE_T0001" and ids[-1] == "ELOISE_T0012"
    assert len(set(ids)) == len(ids) and ids == sorted(ids)
    assert [t["index"] for t in turns] == list(range(1, 13))


def test_structuring_is_deterministic():
    text = "Intro\nEnquêteur : a\nEnquêté : b\nEloïse : c\nEloïse : d"
    assert structure(text) == structure(text)


def test_source_file_and_pages():
    pages = [
        {"page": 1, "text": "Enquêteur : Bonjour.\nEnquêté : Début de réponse"},
        {"page": 2, "text": "suite de la réponse.\nEnquêteur : Merci."},
    ]
    result = structure_transcript(pages, "X", "Entretien_X.pdf")
    turns = result["turns"]
    assert all(t["source"]["file"] == "Entretien_X.pdf" for t in turns)
    assert (turns[1]["source"]["page"], turns[1]["source"]["page_end"]) == (1, 2)
    assert turns[1]["text"] == "Début de réponse\nsuite de la réponse."
    assert turns[2]["source"]["page"] == 2 and turns[2]["source"]["line_start"] == 4
    assert result["lines"] == [(1, "Enquêteur : Bonjour."), (1, "Enquêté : Début de réponse"),
                               (2, "suite de la réponse."), (2, "Enquêteur : Merci.")]


@pytest.mark.parametrize("filename, expected", [
    ("Eloise.pdf", "ELOISE"),
    ("Éloïse_Franzmann.pdf", "ELOISE_FRANZMANN"),
    ("entretien 3 (v2).docx", "ENTRETIEN_3_V2"),
    ("___.txt", "ENTRETIEN"),
])
def test_make_interview_id(filename, expected):
    assert make_interview_id(filename) == expected
