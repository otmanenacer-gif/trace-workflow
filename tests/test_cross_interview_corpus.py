"""Étape 6 — constitution du corpus : reconnaissance, regroupement, contrôles d'intégrité. Aucun appel."""

import json

from core import config
from core import cross_interview_corpus as corpus
from tests import synthetic_stage5 as S5
from tests import synthetic_stage6 as S6

TRAJ, VAL, MAN = corpus.KIND_TRAJECTORY, corpus.KIND_VALIDATION, corpus.KIND_MANIFEST


def files(iid: str) -> dict[str, bytes]:
    """{kind: octets} pour un entretien synthétique."""
    return {name.removeprefix(f"{iid}_"): data for name, data in S6.stage5_files(iid)}


def tampered(iid: str, kind: str, change) -> list[tuple[str, bytes]]:
    triplet = files(iid)
    document = json.loads(triplet[kind])
    change(document)
    triplet[kind] = json.dumps(document, ensure_ascii=False).encode()
    return [(f"{iid}_{k}", v) for k, v in triplet.items()]


def row(checked, iid):
    return next(r for r in checked["rows"] if r["interview_id"] == iid)


def test_seventeen_genuine_triplets_are_recognized_and_usable():
    checked = corpus.check_corpus(S6.uploads(S6.IDS_17))
    assert (checked["n_total"], checked["n_usable"], checked["mode"]) == (17, 17, corpus.MODE_COMPARATIVE)
    assert [r["interview_id"] for r in checked["rows"]] == S6.IDS_17
    a = row(checked, "ENT_A")
    assert a["status"] == corpus.STATUS_USABLE and a["reasons"] == [] and a["claims"] == 5
    assert a["configuration"] == "contextual_configuration" and set(a["files"]) == {TRAJ, VAL, MAN}
    assert row(checked, "ENT_O")["needs_review"] == 2  # affirmation SINGLE_SUPPORT_PATTERN + critère à revoir
    assert checked["unrecognized"] == [] and checked["notes"] == []


def test_files_are_recognized_by_content_not_by_name():
    renamed = [(f"fichier_{n}.json", data) for n, (_, data) in enumerate(S6.stage5_files("ENT_A"))]
    renamed += [(name.removeprefix("ENT_B_"), data) for name, data in S6.stage5_files("ENT_B")]  # sans préfixe
    checked = corpus.check_corpus(renamed)
    assert checked["n_usable"] == 2 and checked["mode"] == corpus.MODE_EXPLORATORY


def test_unrecognized_and_invalid_json_files_are_listed_but_never_counted_as_interviews():
    episodes = json.dumps({"agent": "accountability_episode_builder", "interview_id": "ENT_A"}).encode()
    uploads = S6.uploads(S6.IDS_4) + [("notes.json", b"{not json"), ("autre.json", b'{"interview_id": "Z"}'),
                                      ("episodes.json", episodes)]
    checked = corpus.check_corpus(uploads)
    assert checked["n_total"] == 4 and checked["n_usable"] == 4
    assert [u["file"] for u in checked["unrecognized"]] == ["notes.json", "autre.json", "episodes.json"]
    assert checked["unrecognized"][0]["reason"] == "JSON invalide."


def test_missing_file_excludes_only_that_interview():
    uploads = [f for f in S6.uploads(S6.IDS_4) if f[0] != f"ENT_C_{MAN}"]
    checked = corpus.check_corpus(uploads)
    assert (checked["n_total"], checked["n_usable"]) == (4, 3)
    c = row(checked, "ENT_C")
    assert c["status"] == corpus.STATUS_EXCLUDED and c["reasons"] == [f"Fichier manquant : {MAN} (manifest)."]


def test_identical_duplicates_are_deduplicated_and_differing_duplicates_are_ambiguous():
    same = S6.uploads(["ENT_A", "ENT_B"]) + [S6.stage5_files("ENT_A")[0]]
    checked = corpus.check_corpus(same)
    assert checked["n_usable"] == 2 and "importé 2 fois à l'identique" in checked["notes"][0]

    other = tampered("ENT_A", TRAJ, lambda d: d.update(trajectory_summary="Autre version."))[0]
    checked = corpus.check_corpus(S6.uploads(["ENT_A", "ENT_B", "ENT_C"]) + [("copie.json", other[1])])
    a = row(checked, "ENT_A")
    assert a["status"] == corpus.STATUS_EXCLUDED and a["reasons"][0].startswith("Doublon ambigu : 2 versions")
    assert (checked["n_total"], checked["n_usable"]) == (3, 2)


def test_triplet_mixing_two_interviews_is_never_assembled():
    # le manifest de ENT_A porte l'identifiant de ENT_B : ENT_A n'a plus de manifest, ENT_B en a deux différents
    mixed = tampered("ENT_A", MAN, lambda d: d.update(interview_id="ENT_B"))
    checked = corpus.check_corpus(mixed + S6.uploads(["ENT_B", "ENT_C"]))
    assert row(checked, "ENT_A")["reasons"] == [f"Fichier manquant : {MAN} (manifest)."]
    assert row(checked, "ENT_B")["reasons"][0].startswith("Doublon ambigu")
    assert checked["n_usable"] == 1 and checked["mode"] == corpus.MODE_BLOCKED


def test_stage5_with_validation_errors_is_excluded_with_its_reason():
    checked = corpus.check_corpus(S6.uploads(S6.IDS_4 + ["ENT_X_INVALIDE"]))
    x = row(checked, "ENT_X_INVALIDE")
    assert x["status"] == corpus.STATUS_EXCLUDED
    assert x["reasons"] == ["validation_error_count = 2 (manifest 2, validation 2) : une étape 5 avec des erreurs de "
                            "validation n'est pas importée."]
    assert (checked["n_total"], checked["n_usable"], checked["mode"]) == (5, 4, corpus.MODE_COMPARATIVE)


def check_one(iid, kind, change):
    return row(corpus.check_corpus(tampered(iid, kind, change)), iid)


def test_incompatible_versions_and_incomplete_runs_are_refused():
    assert "Version incompatible : schema_version = 0.9" in check_one("ENT_A", TRAJ, lambda d: d.update(
        schema_version="0.9"))["reasons"][0]
    assert "Version incompatible : validator_version" in check_one("ENT_A", TRAJ, lambda d: d.update(
        validator_version="0.1"))["reasons"][0]
    partial = check_one("ENT_A", TRAJ, lambda d: d.update(status="PARTIAL", analysis_complete=False))
    assert any("Statut PARTIAL" in r for r in partial["reasons"])
    assert any("analysis_complete = false" in r for r in partial["reasons"])
    unavailable = check_one("ENT_A", VAL, lambda d: d.update(available=False))
    assert any("non disponible" in r for r in unavailable["reasons"])


def test_other_prompt_or_agent_version_is_only_a_warning():
    other = check_one("ENT_A", TRAJ, lambda d: d.update(prompt_sha256="0" * 64))
    assert other["status"] == corpus.STATUS_USABLE
    assert other["warnings"] == ["Configuration produite avec d'autres consignes que celles de l'étape 5 actuelle."]


def test_hashes_and_manifest_must_describe_the_document():
    hashes = check_one("ENT_A", MAN, lambda d: d["source_hashes"].update(practice_extractor_sha256="0" * 64))
    assert any("Empreintes (source_hashes)" in r for r in hashes["reasons"])
    config_changed = check_one("ENT_A", MAN, lambda d: d.update(configuration_type="mixed"))
    assert any("Le manifest ne décrit pas ce document" in r for r in config_changed["reasons"])
    counts = check_one("ENT_A", MAN, lambda d: d.update(kept_claim_count=99))
    assert counts["reasons"] == ["Comptes du manifest différents de ceux du document."]


def test_altered_documents_are_detected():
    # needs_review effacé sur un élément qui porte des review_reasons (comptes recalculés, puis objet)
    def clean_review(d):
        claim = next(c for c in d["trajectory_claims"] if c["needs_review"])
        claim["needs_review"] = False
    cleaned = check_one("ENT_O", TRAJ, clean_review)
    assert cleaned["status"] == corpus.STATUS_EXCLUDED and "Comptes incohérents" in cleaned["reasons"][0]

    def clean_review_and_counts(d):
        clean_review(d)
        d["claims_needing_review_count"] -= 1
    assert any("needs_review incohérent" in r for r in check_one("ENT_O", TRAJ, clean_review_and_counts)["reasons"])

    # raison de revue remplacée : incohérente avec les anomalies enregistrées
    def swap_reasons(d):
        claim = next(c for c in d["trajectory_claims"] if c["needs_review"])
        claim["review_reasons"] = ["CRITERION_NOT_GROUNDED"]
    assert any("review_reasons incohérentes" in r for r in check_one("ENT_O", TRAJ, swap_reasons)["reasons"])

    # affirmation retirée : comptes et listes recalculés
    def drop_claim(d):
        d["trajectory_claims"].pop()
    assert any("Comptes incohérents" in r for r in check_one("ENT_A", TRAJ, drop_claim)["reasons"])

    # liste par type modifiée
    assert any("Liste stable_boundaries" in r for r in check_one("ENT_A", TRAJ, lambda d: d.update(
        stable_boundaries=[]))["reasons"])

    # statut de validation « nettoyé »
    def validate_claim(d):
        claim = next(c for c in d["trajectory_claims"] if c["needs_review"])
        claim.update(validation_status="valid")
    assert any("statut de validation incohérent" in r for r in check_one("ENT_O", TRAJ, validate_claim)["reasons"])

    # identifiant d'un autre entretien (critère : absent des listes par type)
    def foreign(d):
        d["student_role_criteria"][0]["criterion_id"] = "ENT_B_RC001"
    assert any("identifiant étranger" in r for r in check_one("ENT_A", TRAJ, foreign)["reasons"])


def test_corpus_modes():
    assert corpus.corpus_mode(0) == corpus.corpus_mode(1) == corpus.MODE_BLOCKED
    assert corpus.corpus_mode(2) == corpus.MODE_EXPLORATORY
    assert corpus.corpus_mode(3) == corpus.corpus_mode(17) == corpus.MODE_COMPARATIVE
    assert corpus.check_corpus(S6.uploads(["ENT_A"]))["mode"] == corpus.MODE_BLOCKED
    assert corpus.check_corpus([])["n_total"] == 0


def test_corpus_id_depends_only_on_the_files():
    first = corpus.corpus_id(corpus.check_corpus(S6.uploads(S6.IDS_4)))
    assert first == corpus.corpus_id(corpus.check_corpus(list(reversed(S6.uploads(S6.IDS_4)))))
    assert first != corpus.corpus_id(corpus.check_corpus(S6.uploads(S6.IDS_8)))


def test_stage5_outputs_of_the_real_pipeline_are_importable(tmp_path):
    """Triplet produit par le pipeline complet (étapes 3, 4 et 5 simulées) et repris du run courant."""
    run, cache = S5.case_to_stage4(tmp_path, S5.EXCEPTION)
    run, _ = S5.run_stage5(run, cache, S5.EXCEPTION.mapper())
    uploads = corpus.run_stage5_uploads(run)
    assert sorted(n.removeprefix(f"{S5.EXCEPTION.interview_id}_") for n, _ in uploads) == sorted(
        [config.STUDENT_TRAJECTORY_FILENAME, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME,
         config.STUDENT_TRAJECTORY_MANIFEST_FILENAME])
    checked = corpus.check_corpus(uploads + S6.uploads(["ENT_A"]))
    assert row(checked, S5.EXCEPTION.interview_id)["status"] == corpus.STATUS_USABLE, checked["rows"]
    assert checked["n_usable"] == 2


def test_run_without_stage5_gives_no_upload(tmp_path):
    run, _ = S5.case_to_stage4(tmp_path, S5.EXCEPTION)
    assert corpus.run_stage5_uploads(run) == []
    assert corpus.run_stage5_uploads(None) == []
