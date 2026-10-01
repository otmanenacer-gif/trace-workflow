"""Corpus SYNTHÉTIQUE de l'étape 6 : sorties d'étape 5 authentiques et Comparator simulé (aucun appel réel).

Chaque entretien est composé de MODULES (échanges question / réponse + affirmations et critères attendus de
l'étape 5). Le triplet d'étape 5 (student_trajectory.json, student_trajectory_validation.json,
student_trajectory_manifest.json) est produit par le VRAI orchestrateur de l'étape 5
(`core.trajectory.run_stage5_interview` : préparation, validateur, document, manifest) avec un Trajectory
Mapper simulé : l'étape 6 importe donc de vraies sorties, jamais des objets fabriqués à la main.

Corpus de 17 entretiens (ENT_A … ENT_Q) :
- frontière « seulement quand je manque de temps » pour rédiger : A, B, C, E, I ;
- cas négatif D : « je l'utilise même quand j'ai le temps » ;
- critère d'effort : A, C, E, F, G, K, N ; jamais mentionné en B, D… (non observé) ; refus explicite en H
  (« L'effort n'est pas important pour moi ») ;
- maths / écriture : E, G, I, J, K ; cas inverse L ;
- changements temporels explicites : M, N seulement ;
- vérification appuyée uniquement sur des éléments à revoir : O, P ;
- Q : presque aucune accountability (zone ordinaire seulement) ;
- ENT_X_INVALIDE : sortie d'étape 5 réelle avec une erreur de validation (affirmation rejetée).

`scripted_comparator(plan)` résout un plan « par module » dans la représentation réellement envoyée au
Cross-Interview Comparator (identifiants abrégés I01, TC001…) ; les entretiens absents du corpus sont ignorés,
de sorte que le même plan sert aux corpus de 2, 4, 8 ou 17 entretiens.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from core import config
from tests.fake_llm import TRAJECTORY, FakeTransport, fake_settings, text_response

QUESTION = "Et pour {task}, comment ça se passe ?"


@dataclass(frozen=True)
class Ex:
    """Un échange : question de l'enquêteur, puis réponse de l'enquêté·e (citation exacte = toute la réponse)."""
    answer: str
    task: str
    use: str = "use"
    status: str = "accountability_episode"   # accountability_episode | ordinary_practice | unmarked
    moves: tuple = ()
    boundary: tuple = ()
    role: str | None = None
    external: str | None = None
    discipline: str | None = None
    domain: str = "academic"
    review: tuple = ()


@dataclass(frozen=True)
class Module:
    name: str
    exchanges: tuple
    claims: tuple = ()      # dicts : claim_type, description, items (indices locaux 1…), evidence, anchors…
    criteria: tuple = ()


def cl(claim_type, description, items, evidence=None, contexts=(), moves=(), anchors=(), confidence="medium",
       needs_review=False) -> dict:
    return {"claim_type": claim_type, "description": description, "items": list(items),
            "evidence": list(evidence or items), "contexts": list(contexts), "moves": list(moves),
            "anchors": list(anchors), "confidence": confidence, "needs_review": needs_review}


def cr(criterion, description, items, evidence=None, confidence="medium", needs_review=False) -> dict:
    return {"criterion": criterion, "description": description, "items": list(items),
            "evidence": list(evidence or items), "confidence": confidence, "needs_review": needs_review}


# --- Modules -----------------------------------------------------------------------------------------------

def time_boundary(first: str, second: str) -> Module:
    return Module("TIME", (
        Ex(f"Pour les {first}, je lui fais rédiger un paragraphe seulement quand je manque de temps, sinon j'écris "
           "moi-même.", first, moves=("restriction",), boundary=("seulement quand je manque de temps",)),
        Ex(f"Pour les {second} c'est pareil, je l'utilise pour rédiger seulement quand je manque de temps.", second,
           moves=("restriction",), boundary=("seulement quand je manque de temps",)),
    ), claims=(
        cl("stable_boundary", f"Pour les {first} et les {second}, l'étudiant·e borne la rédaction par l'outil de la "
                              "même manière : « seulement quand je manque de temps ».", [1, 2],
           contexts=(first, second), confidence="high"),
        cl("recurring_accounting_move", "Dans deux épisodes, l'étudiant·e restreint la rédaction par l'outil au "
                                        "manque de temps.", [1, 2], moves=("restriction",)),
    ), criteria=(
        cr("temps disponible", "Dans cet entretien, l'étudiant·e associe le recours à l'outil pour rédiger au manque "
                               "de temps (« seulement quand je manque de temps »).", [1, 2]),
    ))


ALWAYS = Module("ALWAYS", (
    Ex("Je lui fais rédiger mes introductions, je l'utilise même quand j'ai le temps, c'est plus simple.",
       "introductions", status="unmarked"),
    Ex("Pour les dissertations aussi je lui fais écrire des paragraphes, même quand j'ai le temps.", "dissertations",
       status="unmarked"),
), claims=(
    cl("ordinary_zone", "La rédaction d'introductions et de paragraphes de dissertation par l'outil est racontée sans "
                        "restriction : « je l'utilise même quand j'ai le temps ».", [1, 2],
       contexts=("introductions", "dissertations")),
))

EFFORT = Module("EFFORT", (
    Ex("Je ne lui demande pas de chercher les sources à ma place, je veux faire l'effort moi-même.",
       "recherche de sources", use="non_use", moves=("refusal", "appeal_to_effort"),
       role="étudiant·e qui cherche ses sources"),
    Ex("Pour les exercices, je les fais d'abord seul, l'effort c'est ce qui compte quand on est étudiant.",
       "exercices", use="non_use", moves=("appeal_to_effort", "general_rule"), role="étudiant·e qui fait l'effort"),
), claims=(
    cl("recurring_accounting_move", "Dans deux épisodes, l'étudiant·e rapporte le non-usage à l'effort fourni "
                                    "(« je veux faire l'effort moi-même »).", [1, 2], moves=("appeal_to_effort",)),
), criteria=(
    cr("effort", "Dans cet entretien, l'étudiant·e associe le fait d'être étudiant·e à l'effort fourni (« je veux "
                 "faire l'effort moi-même », « l'effort c'est ce qui compte »).", [1, 2]),
))

NO_EFFORT = Module("NO_EFFORT", (
    Ex("L'effort n'est pas important pour moi, ce qui compte c'est de rendre un bon devoir.", "devoirs maison",
       moves=("general_rule",), role="étudiant·e évalué·e sur le résultat"),
), criteria=(
    cr("résultat rendu", "Dans cet entretien, l'étudiant·e associe le travail étudiant au résultat rendu et dit : "
                         "« L'effort n'est pas important pour moi ».", [1]),
))

PLANS = Module("PLANS", (
    Ex("Je ne fais jamais faire mes plans par l'outil, je fais mes plans moi-même.", "plans de dissertation",
       use="non_use", moves=("general_rule", "refusal"), boundary=("jamais faire faire mes plans",),
       role="faire ses plans soi-même"),
    Ex("Une fois, faute de temps, je lui ai demandé un plan pour un commentaire.", "plan de commentaire",
       moves=("exception",)),
), claims=(
    cl("exception", "L'étudiant·e énonce une règle (« Je ne fais jamais faire mes plans ») et rapporte un cas présenté "
                    "comme unique (« Une fois, faute de temps »).", [1, 2], contexts=("plans",)),
), criteria=(
    cr("faire ses plans soi-même", "Dans cet entretien, l'étudiant·e associe le plan au fait de le faire soi-même "
                                   "(« je fais mes plans moi-même »).", [1]),
))

MATHS_WRITING = Module("MATHS", (
    Ex("En maths je lui demande directement la solution des exercices.", "exercices de maths", status="unmarked",
       discipline="mathématiques"),
    Ex("Pour les dissertations, je veux écrire moi-même, c'est mon travail.", "dissertations", use="non_use",
       moves=("appeal_to_authorship", "refusal"), boundary=("écrire moi-même",), role="auteur·e de ses dissertations",
       discipline="philosophie"),
), claims=(
    cl("contextual_variation", "En mathématiques, l'étudiant·e dit demander directement la solution des exercices ; "
                               "pour les dissertations, il ou elle dit vouloir « écrire moi-même ».", [1, 2],
       contexts=("exercices de maths", "dissertations")),
), criteria=(
    cr("être l'auteur·e de ses écrits", "Dans cet entretien, l'étudiant·e associe l'écriture des dissertations à "
                                        "« mon travail ».", [2]),
))

MATHS_REVERSED = Module("MATHS_REV", (
    Ex("En maths je fais tout moi-même, je ne lui demande jamais la solution.", "exercices de maths", use="non_use",
       moves=("refusal",), boundary=("jamais la solution",), discipline="mathématiques"),
    Ex("Pour les dissertations, je lui fais reformuler et même écrire des passages.", "dissertations",
       status="unmarked", discipline="philosophie"),
), claims=(
    cl("contextual_variation", "En mathématiques, l'étudiant·e dit tout faire soi-même (« je ne lui demande jamais "
                               "la solution ») ; pour les dissertations, il ou elle fait reformuler et écrire des "
                               "passages.",
       [1, 2], contexts=("exercices de maths", "dissertations")),
))

TEMPORAL = Module("TEMPORAL", (
    Ex("Au lycée je lui faisais rédiger mes commentaires de texte.", "commentaires de texte", use="past_use",
       status="unmarked"),
    Ex("Maintenant je ne m'en sers plus que pour relire mes dissertations.", "relecture", moves=("restriction",),
       boundary=("seulement relire",)),
), claims=(
    cl("explicit_temporal_change", "L'étudiant·e situe « Au lycée » la rédaction de commentaires de texte par l'outil "
                                   "et dit « Maintenant » ne s'en servir que pour relire.", [1, 2],
       anchors=((1, "Au lycée"), (2, "Maintenant")), contexts=("commentaires de texte", "relecture"),
       confidence="high"),
))

VERIFICATION = Module("VERIF", (
    Ex("Je vérifie toujours tout ce qu'il me donne avant de l'utiliser.", "réponses de l'outil",
       moves=("appeal_to_verification",), role="étudiant·e qui vérifie"),
), claims=(
    # un seul épisode : l'étape 5 signale SINGLE_SUPPORT_PATTERN, l'affirmation est à revoir
    cl("recurring_accounting_move", "L'étudiant·e rapporte l'usage à une vérification (« je vérifie toujours tout ce "
                                    "qu'il me donne »).", [1], moves=("appeal_to_verification",)),
), criteria=(
    cr("vérification", "Dans cet entretien, l'étudiant·e associe l'usage à la vérification de ce que l'outil donne.",
       [1], needs_review=True),
))

GRADED = Module("GRADED", (
    Ex("Pour les dossiers notés je ne l'utilise pas du tout.", "dossiers notés", use="non_use", moves=("refusal",),
       boundary=("pas pour les dossiers notés",), external="le professeur"),
    Ex("Pour les partiels à la maison non plus, je ne lui demande rien.", "partiels à la maison", use="non_use",
       moves=("refusal",), boundary=("pas pour les partiels",)),
), claims=(
    cl("stable_boundary", "Pour les dossiers notés et les partiels à la maison, l'étudiant·e dit ne pas utiliser "
                          "l'outil (« je ne l'utilise pas du tout »).", [1, 2],
       contexts=("dossiers notés", "partiels à la maison")),
    cl("recurring_accounting_move", "Dans deux épisodes, l'étudiant·e dit ne pas faire faire les travaux notés.",
       [1, 2], moves=("refusal",)),
), criteria=(
    cr("travaux notés faits sans l'outil", "Dans cet entretien, l'étudiant·e associe les travaux notés au fait de "
                                           "ne rien demander à l'outil.", [1, 2]),
))

TEACHER = Module("TEACHER", (
    Ex("Ma prof dirait que ce n'est pas mon travail si je lui faisais écrire mes devoirs.", "devoirs",
       use="non_use", moves=("appeal_to_external_judgment", "refusal"), external="la professeure",
       role="travail reconnu par la professeure"),
), criteria=(
    cr("jugement de l'enseignant·e", "Dans cet entretien, l'étudiant·e associe la limite de l'usage à ce que dirait "
                                     "sa professeure (« Ma prof dirait que ce n'est pas mon travail »).", [1]),
))

TENSION = Module("TENSION", (
    Ex("Je comprends tout ce qu'il me propose, sinon je ne le garde pas.", "explications de cours",
       moves=("appeal_to_learning", "general_rule"), role="comprendre ce qu'on rend"),
    Ex("Parfois je recopie sans tout comprendre, c'est vrai.", "explications de cours", moves=("self_evaluation",)),
), claims=(
    cl("unresolved_tension", "Deux formulations coexistent : « Je comprends tout ce qu'il me propose » et « Parfois je "
                             "recopie sans tout comprendre » ; l'entretien ne les articule pas.", [1, 2],
       contexts=("explications de cours",)),
), criteria=(
    cr("compréhension", "Dans cet entretien, l'étudiant·e associe l'usage à la compréhension (« Je comprends tout ce "
                        "qu'il me propose »).", [1]),
))

INFO = Module("INFO", (
    Ex("Je lui demande les horaires de la bibliothèque.", "recherche d'information", status="unmarked"),
    Ex("Je lui demande des définitions quand je lis un article.", "définitions", status="unmarked"),
), claims=(
    cl("ordinary_zone", "Des recherches d'information pratiques (horaires, définitions) sont racontées sans "
                        "restriction, justification ni évaluation explicite.", [1, 2],
       contexts=("recherche d'information",), confidence="high"),
))

TRANSLATION = Module("TRANSLATION", (
    Ex("Je lui fais traduire des articles en anglais.", "traduction d'articles", status="unmarked"),
    Ex("Je lui demande aussi de traduire des résumés.", "traduction de résumés", status="unmarked"),
), claims=(
    cl("ordinary_zone", "Des traductions d'articles et de résumés sont racontées sans restriction, justification ni "
                        "évaluation explicite.", [1, 2], contexts=("traduction",)),
))

PERSONAL = Module("PERSONAL", (
    Ex("Je lui demande des recettes de cuisine le soir.", "cuisine", status="unmarked", domain="personal"),
    Ex("Je lui demande des idées de sorties le week-end.", "sorties", status="unmarked", domain="personal"),
), claims=(
    cl("ordinary_zone", "Des demandes de la vie personnelle (recettes, sorties) sont racontées sans restriction, "
                        "justification ni évaluation explicite.", [1, 2], contexts=("vie personnelle",)),
))

BROKEN = Module("BROKEN", (
    Ex("Je lui demande de corriger l'orthographe de mes mails.", "mails", status="unmarked"),
    Ex("Je lui demande aussi de corriger mes lettres de motivation.", "lettres", status="unmarked"),
), claims=(
    cl("ordinary_zone", "Des corrections d'orthographe sont racontées sans justification.", [1, 2]),
    # épisode inexistant : erreur de validation de l'étape 5 → affirmation rejetée, validation_error_count = 1
    cl("stable_boundary", "Une frontière appuyée sur un épisode qui n'existe pas.", [("E", "E099")], evidence=[1]),
))


@dataclass
class Interview:
    interview_id: str
    configuration: str
    modules: tuple
    summary: str = ""
    index: dict = field(default_factory=dict)   # module → {"claims": [ids], "criteria": [ids]}


def _summary(interview: Interview) -> str:
    return interview.summary or ("Dans cet entretien, les usages se distribuent selon les tâches ; les passages sont "
                                 "décrits au discours rapporté, sans comparaison.")


INTERVIEWS = [
    Interview("ENT_A", "contextual_configuration",
              (time_boundary("dissertations", "fiches de lecture"), EFFORT, INFO, PLANS, TEACHER)),
    Interview("ENT_B", "contextual_configuration",
              (time_boundary("rapports de stage", "exposés"), PLANS, GRADED, INFO, TRANSLATION)),
    Interview("ENT_C", "contextual_configuration",
              (time_boundary("notes de synthèse", "dissertations"), EFFORT, TENSION, TRANSLATION)),
    Interview("ENT_D", "contextual_configuration",
              (ALWAYS, INFO, GRADED, TEACHER, PERSONAL)),
    Interview("ENT_E", "contextual_configuration",
              (time_boundary("exposés", "fiches de révision"), MATHS_WRITING, INFO, EFFORT)),
    Interview("ENT_F", "contextual_configuration",
              (EFFORT, GRADED, TRANSLATION, TENSION, PERSONAL)),
    Interview("ENT_G", "contextual_configuration",
              (MATHS_WRITING, EFFORT, PLANS, INFO)),
    Interview("ENT_H", "contextual_configuration",
              (NO_EFFORT, GRADED, INFO, PERSONAL)),
    Interview("ENT_I", "contextual_configuration",
              (MATHS_WRITING, time_boundary("dissertations", "commentaires"), TRANSLATION, PERSONAL)),
    Interview("ENT_J", "contextual_configuration",
              (MATHS_WRITING, TENSION, INFO, TEACHER)),
    Interview("ENT_K", "contextual_configuration",
              (MATHS_WRITING, GRADED, EFFORT, PERSONAL)),
    Interview("ENT_L", "contextual_configuration",
              (MATHS_REVERSED, INFO, PLANS, TRANSLATION)),
    Interview("ENT_M", "mixed",
              (TEMPORAL, TEACHER, TRANSLATION, GRADED)),
    Interview("ENT_N", "mixed",
              (TEMPORAL, EFFORT, INFO, TENSION)),
    Interview("ENT_O", "contextual_configuration",
              (VERIFICATION, INFO, PLANS, PERSONAL)),
    Interview("ENT_P", "contextual_configuration",
              (VERIFICATION, TRANSLATION, GRADED)),
    Interview("ENT_Q", "no_clear_pattern",
              (INFO,)),
]
INVALID = Interview("ENT_X_INVALIDE", "contextual_configuration", (BROKEN, INFO))
BY_ID = {i.interview_id: i for i in [*INTERVIEWS, INVALID]}
IDS = [i.interview_id for i in INTERVIEWS]
IDS_2 = IDS[:2]
IDS_4 = IDS[:4]
IDS_8 = IDS[:8]
IDS_17 = IDS


# --- Production des sorties de l'étape 5 (vrai orchestrateur) -------------------------------------------------

def _layout(interview: Interview):
    """Échanges numérotés (1…) avec leur module ; références locales → globales."""
    rows, offsets = [], {}
    for module in interview.modules:
        offsets[module.name] = len(rows)
        rows.extend((module, ex) for ex in module.exchanges)
    return rows, offsets


def stage5_inputs(interview: Interview, analysis_dir: Path):
    """Transcription, sorties d'étape 4 (forme réelle) et matériau de l'étape 5 préparé par TRACE."""
    from core import trajectory
    from core import trajectory_candidates as tc

    iid = interview.interview_id
    rows, _ = _layout(interview)
    turns, episodes, unmarked, practices = [], [], [], []
    for number, (_, ex) in enumerate(rows, start=1):
        q, a = f"{iid}_T{2 * number - 1:04d}", f"{iid}_T{2 * number:04d}"
        turns += [{"turn_id": q, "speaker": "enqueteur", "text": QUESTION.format(task=ex.task)},
                  {"turn_id": a, "speaker": "enquete", "text": ex.answer}]
        pid, evidence = f"{iid}_P{number:03d}", [{"turn_id": a, "quote": ex.answer}]
        practices.append({"practice_id": pid, "evidence": evidence, "use_status": ex.use, "academic_task": ex.task,
                          "practice_domain": ex.domain, "discipline": ex.discipline, "stated_frequency": None,
                          "stated_reason": [], "assessment_context": None})
        if ex.status == "unmarked":
            unmarked.append({"practice_id": pid, "summary": f"Usage raconté : {ex.task}.", "turn_start": q,
                             "turn_end": a})
            continue
        episodes.append({
            "episode_id": f"{iid}_E{number:03d}", "episode_status": ex.status, "practice_ids": [pid],
            "signal_ids": [], "turn_start": q, "turn_end": a,
            "accounting_moves": [{"type": m, "description": f"Opération {m}.", "evidence_turn_ids": [a]}
                                 for m in ex.moves],
            "boundary_objects": list(ex.boundary), "accountability_problem": f"Usage pour {ex.task}.",
            "episode_summary": f"L'étudiant·e parle de {ex.task}.", "student_role_reference": ex.role,
            "external_reference": ex.external, "confidence": "medium", "needs_review": bool(ex.review),
            "review_reasons": list(ex.review), "speaker_warnings": [], "evidence": evidence,
            "validation_status": "needs_review" if ex.review else "valid", "usable_for_next_stages": True})
    transcript = {"interview_id": iid, "source": {"sha256": hashlib.sha256(iid.encode()).hexdigest()},
                  "turns": turns}
    material = tc.build_material(transcript, {"episodes": episodes, "unmarked_practices": unmarked}, practices, [], {})
    request = trajectory.build_request(transcript, material, {}) if material["summary"]["item_count"] >= 2 else None
    fake = lambda name: hashlib.sha256(f"{iid}/{name}".encode()).hexdigest()  # noqa: E731
    source_hashes = {"source_sha256": transcript["source"]["sha256"], "structured_transcript_sha256": fake("t"),
                     "practice_extractor_sha256": fake("p"), "interaction_signals_sha256": fake("s"),
                     "speaker_audit_sha256": fake("a"), "accountability_episodes_sha256": fake("e"),
                     "accountability_validation_sha256": fake("v")}
    prepared = trajectory.Stage5Input(
        interview_id=iid, analysis_dir=analysis_dir, transcript=transcript,
        source_sha256=transcript["source"]["sha256"],
        stage4={"status": trajectory.STAGE4_COMPLETE, "reasons": [], "restored": False}, source_hashes=source_hashes,
        speaker_warnings={}, material=material, request=request)
    return prepared, rows


def mapper_output(interview: Interview) -> dict:
    """Réponse du Trajectory Mapper simulé (identifiants abrégés, comme le vrai modèle)."""
    rows, offsets = _layout(interview)

    def ref(module: Module, local):
        if isinstance(local, tuple):  # ("E", "E099") : identifiant passé tel quel (lecteur adverse)
            return "episode", local[1]
        number = offsets[module.name] + local
        ex = rows[number - 1][1]
        return ("practice", f"P{number:03d}") if ex.status == "unmarked" else ("episode", f"E{number:03d}")

    def turn(module: Module, local: int) -> str:
        return f"T{2 * (offsets[module.name] + local):04d}"

    claims, criteria = [], []
    for module in interview.modules:
        for c in module.claims:
            refs = [ref(module, i) for i in c["items"]]
            claims.append({"claim_type": c["claim_type"], "description": c["description"],
                           "episode_ids": [i for k, i in refs if k == "episode"],
                           "practice_ids": [i for k, i in refs if k == "practice"], "contexts": c["contexts"],
                           "accounting_move_types": c["moves"],
                           "temporal_anchors": [{"text": t, "turn_id": turn(module, n)} for n, t in c["anchors"]],
                           "evidence_turn_ids": [turn(module, n) for n in c["evidence"]],
                           "confidence": c["confidence"], "needs_review": c["needs_review"]})
        for c in module.criteria:
            criteria.append({"criterion": c["criterion"], "description": c["description"],
                             "episode_ids": [ref(module, i)[1] for i in c["items"]],
                             "evidence_turn_ids": [turn(module, n) for n in c["evidence"]],
                             "confidence": c["confidence"], "needs_review": c["needs_review"]})
    return {"configuration_type": interview.configuration, "claims": claims, "student_role_criteria": criteria,
            "trajectory_summary": _summary(interview), "confidence": "medium", "needs_review": False,
            "mapper_notes": None}


def _module_index(interview: Interview) -> dict:
    """module → identifiants des affirmations et critères produits par l'étape 5 (dans l'ordre de sortie)."""
    index, claim_n, criterion_n = {}, 0, 0
    for module in interview.modules:
        entry = index.setdefault(module.name, {"claims": [], "criteria": []})
        for _ in module.claims:
            claim_n += 1
            entry["claims"].append(f"{interview.interview_id}_TC{claim_n:03d}")
        for _ in module.criteria:
            criterion_n += 1
            entry["criteria"].append(f"{interview.interview_id}_RC{criterion_n:03d}")
    return index


STAGE5_NAMES = (config.STUDENT_TRAJECTORY_FILENAME, config.STUDENT_TRAJECTORY_VALIDATION_FILENAME,
                config.STUDENT_TRAJECTORY_MANIFEST_FILENAME)


@lru_cache(maxsize=None)
def stage5_files(interview_id: str) -> tuple[tuple[str, bytes], ...]:
    """Les 3 JSON de l'étape 5 d'un entretien synthétique, produits par le vrai orchestrateur (FakeTransport)."""
    from core import trajectory
    from core.analysis_cache import AnalysisCache
    from core.llm_client import LLMClient

    interview = BY_ID[interview_id]
    with tempfile.TemporaryDirectory(prefix="trace_stage6_fixture_") as tmp:
        analysis_dir = Path(tmp) / interview_id / config.ANALYSIS_SUBDIR
        prepared, _ = stage5_inputs(interview, analysis_dir)
        output = mapper_output(interview)
        transport = FakeTransport({TRAJECTORY: lambda params: text_response(output, input_tokens=4000,
                                                                            output_tokens=2500)})
        settings = fake_settings()

        async def run():
            client = LLMClient(settings, transport=transport)
            try:
                return await trajectory.run_stage5_interview(prepared, client, AnalysisCache(Path(tmp) / "cache"),
                                                             settings)
            finally:
                await client.aclose()

        manifest = asyncio.run(run())
        assert manifest["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS"), (interview_id, manifest)
        return tuple((f"{interview_id}_{name}", (analysis_dir / name).read_bytes()) for name in STAGE5_NAMES)


def module_index(interview_id: str) -> dict:
    return _module_index(BY_ID[interview_id])


def uploads(ids) -> list[tuple[str, bytes]]:
    return [f for iid in ids for f in stage5_files(iid)]


def document(interview_id: str, name: str = config.STUDENT_TRAJECTORY_FILENAME) -> dict:
    return json.loads(dict(stage5_files(interview_id))[f"{interview_id}_{name}"])


def write_corpus(directory: Path, ids) -> list[Path]:
    """Écrit les triplets sur disque (test navigateur). Renvoie les chemins."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in uploads(ids):
        path = directory / name
        path.write_bytes(data)
        paths.append(path)
    return paths


# --- Cross-Interview Comparator simulé ---------------------------------------------------------------------

_MATERIAL_RE = re.compile(r"<corpus>\n(.*)\n</corpus>", re.DOTALL)


def sent_material(params: dict) -> dict:
    """Représentation réellement transmise au Comparator."""
    return json.loads(_MATERIAL_RE.search(params["messages"][0]["content"]).group(1).replace("<\\/", "</"))


def cc(claim_type: str, description: str, support=(), counter=(), contexts=(), confidence="medium",
       needs_review=False, criterion_label=None, related=()) -> dict:
    """support : (entretien, module, sélection) — sélection « c0 » (1re affirmation du module), « r0 » (1er critère).
    counter : (entretien, module, sélection, relation, description)."""
    return {"claim_type": claim_type, "description": description, "support": list(support), "counter": list(counter),
            "contexts": list(contexts), "confidence": confidence, "needs_review": needs_review,
            "criterion_label": criterion_label, "related": list(related)}


def comparator_output(params: dict, plan: dict) -> dict:
    material = sent_material(params)
    alias = {entry["interview_id"]: entry["id"] for entry in material["interviews"]}
    by_alias = {entry["id"]: entry for entry in material["interviews"]}

    def resolve(iid: str, module: str, selection: str) -> dict | None:
        if iid not in alias:
            return None
        ids = module_index(iid)[module]["claims" if selection[0] == "c" else "criteria"]
        full = ids[int(selection[1:])]
        local = full.rsplit("_", 1)[-1]
        entry = by_alias[alias[iid]]
        item = (entry.get("claims") or {}).get(local) or (entry.get("criteria") or {}).get(local)
        if item is None:  # affirmation rejetée par l'étape 5 : absente de la représentation
            return None
        kind = "claim_ids" if selection[0] == "c" else "criterion_ids"
        return {"interview_id": alias[iid], kind: [local], "turns": item.get("turns", [])}

    claims = []
    for c in plan["claims"]:
        support: dict[str, dict] = {}
        for iid, module, selection in c["support"]:
            found = resolve(iid, module, selection)
            if found is None:
                continue
            entry = support.setdefault(found["interview_id"], {"interview_id": found["interview_id"], "claim_ids": [],
                                                               "criterion_ids": [], "evidence_turn_ids": []})
            for key in ("claim_ids", "criterion_ids"):
                entry[key] += found.get(key, [])
            entry["evidence_turn_ids"] += [t for t in found["turns"] if t not in entry["evidence_turn_ids"]]
        if not support:
            continue
        counters = []
        for iid, module, selection, relation, description in c["counter"]:
            found = resolve(iid, module, selection)
            if found is None:
                continue
            counters.append({"interview_id": found["interview_id"], "relation": relation, "description": description,
                             "claim_ids": found.get("claim_ids", []), "criterion_ids": found.get("criterion_ids", []),
                             "evidence_turn_ids": found["turns"]})
        claims.append({"claim_type": c["claim_type"], "description": c["description"],
                       "criterion_label": c["criterion_label"], "support": list(support.values()),
                       "counterexamples": counters, "contexts": c["contexts"],
                       "n_supporting_interviews": len(support), "related_claim_numbers": c["related"],
                       "confidence": c["confidence"], "needs_review": c["needs_review"]})
    return {"cross_case_claims": claims, "cross_case_summary": plan["summary"], "confidence": plan["confidence"],
            "needs_review": plan.get("needs_review", False), "comparator_notes": None}


def scripted_comparator(plan: dict, *, mutate=None):
    """Comparator simulé ; `mutate(output, material)` permet une version adverse."""
    def respond(params: dict):
        output = comparator_output(params, plan)
        if mutate is not None:
            output = mutate(output, sent_material(params))
        return text_response(output, input_tokens=len(params["messages"][0]["content"]) // 3,
                             output_tokens=150 + 220 * len(output["cross_case_claims"]))
    return respond


TIME_SUPPORT = [(i, "TIME", "c0") for i in ("ENT_A", "ENT_B", "ENT_C", "ENT_E", "ENT_I")]
EFFORT_SUPPORT = [(i, "EFFORT", "r0") for i in ("ENT_A", "ENT_C", "ENT_E", "ENT_F", "ENT_G", "ENT_K", "ENT_N")]
MATHS_SUPPORT = [(i, "MATHS", "c0") for i in ("ENT_E", "ENT_G", "ENT_I", "ENT_J", "ENT_K")]
INFO_SUPPORT = [(i, "INFO", "c0") for i in ("ENT_A", "ENT_B", "ENT_D", "ENT_E", "ENT_G", "ENT_H", "ENT_J", "ENT_L",
                                            "ENT_N", "ENT_O", "ENT_Q")]
GRADED_SUPPORT = [(i, "GRADED", "c0") for i in ("ENT_B", "ENT_D", "ENT_F", "ENT_H", "ENT_K", "ENT_M", "ENT_P")]

GOOD_PLAN = {
    "claims": [
        cc("recurring_boundary", "Une même frontière borne la rédaction par l'outil au manque de temps (« seulement "
                                 "quand je manque de temps »), reprise pour différents écrits.", TIME_SUPPORT,
           counter=[("ENT_D", "ALWAYS", "c0", "contrary_case",
                     "L'entretien ENT_D rapporte une rédaction par l'outil « même quand j'ai le temps ».")],
           contexts=["rédaction"], confidence="high"),
        cc("recurring_student_role_criterion", "Le critère de l'effort fourni est formulé pour rester reconnaissable "
                                               "comme étudiant·e (« je veux faire l'effort moi-même »).",
           EFFORT_SUPPORT, counter=[("ENT_H", "NO_EFFORT", "r0", "explicit_refusal",
                                     "L'entretien ENT_H dit : « L'effort n'est pas important pour moi ».")],
           criterion_label="effort", confidence="high"),
        cc("contextual_association", "Dans plusieurs entretiens, les mathématiques sont associées à l'obtention d'une "
                                     "réponse et l'écriture des dissertations au fait d'en être l'auteur·e ; il s'agit "
                                     "d'une association observée.", MATHS_SUPPORT,
           counter=[("ENT_L", "MATHS_REV", "c0", "contrary_case",
                     "L'entretien ENT_L associe au contraire les mathématiques au fait de tout faire soi-même.")],
           contexts=["mathématiques", "dissertations"]),
        cc("recurring_boundary", "Les travaux notés sont tenus à distance de l'outil (« je ne l'utilise pas du "
                                 "tout »).", GRADED_SUPPORT, contexts=["travaux notés"]),
        cc("ordinary_zone_pattern", "Des recherches d'information pratiques sont racontées sans justification.",
           INFO_SUPPORT, contexts=["recherche d'information"], confidence="high"),
        cc("exception_pattern", "Une règle sur les plans coexiste avec un cas présenté comme unique (« Une fois, "
                                "faute de temps »).", [(i, "PLANS", "c0") for i in ("ENT_A", "ENT_B", "ENT_G", "ENT_L",
                                                                                    "ENT_O")], contexts=["plans"]),
        cc("temporal_pattern", "Un changement explicitement daté par l'enquêté·e (« Au lycée » / « Maintenant ») "
                               "concerne la rédaction de commentaires de texte.", [(i, "TEMPORAL", "c0")
                                                                                   for i in ("ENT_M", "ENT_N")],
           contexts=["commentaires de texte"]),
        cc("recurring_accounting_move", "Le rapport de l'usage à une vérification est formulé.",
           [(i, "VERIF", "c0") for i in ("ENT_O", "ENT_P")], confidence="high"),
        cc("negative_case", "L'entretien ENT_D rapporte une rédaction par l'outil « même quand j'ai le temps », ce "
                            "qui complique la frontière du manque de temps.", [("ENT_D", "ALWAYS", "c0")],
           related=[1]),
    ],
    "summary": "Plusieurs frontières, critères et zones ordinaires sont décrits avec leurs entretiens d'appui et leurs "
               "cas négatifs.",
    "confidence": "medium",
}
