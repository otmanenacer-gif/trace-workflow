# TRACE

**Analyse ethnométhodologique des usages étudiants des IAG**

Outil expérimental d'analyse qualitative assistée par IA, destiné à analyser
des entretiens semi-directifs. **TRACE s'exécute entièrement en local** : le Python de TRACE fait tout le travail
déterministe (ingestion, préparation, validation, sorties) et appelle lui-même un **modèle local servi par
Ollama** pour jouer les agents, à partir des prompts du dépôt ([`docs/local_runtime.md`](docs/local_runtime.md)).
**Aucune API externe, aucune clé, aucune donnée envoyée hors de l'ordinateur ; Internet n'est pas nécessaire une
fois Ollama et le modèle installés.**

```
Streamlit → TRACE → agents TRACE → modèle local (Ollama) → validateurs TRACE → étape suivante
```

**Version actuelle :**

1. ingestion **déterministe** des entretiens (sans IA) ;
2. **étape 3** — **exécutée localement (Ollama), sans aucun appel API** — deux agents **indépendants**,
   uniquement sur action explicite : *Practice Extractor* (ce que l'étudiant·e
   fait, ou ne fait pas, avec ou sans IAG) et *Interaction Signal Reader*
   (comment il ou elle le raconte). Aucune synthèse, aucune analyse théorique à ce stade ;
3. **étape 3.5** — *Speaker Attribution Auditor* : avant les deux agents,
   signale les tours dont le locuteur semble mal attribué (règles
   déterministes, puis au plus un appel au modèle local par entretien, aucun s'il n'y a
   pas de tour suspect). La transcription n'est **jamais** modifiée ;
4. **étape 4** — *épisodes d'accountability* : à partir des sorties de l'étape 3
   (jamais de l'entretien entier), un programme déterministe propose des
   candidats, l'*Accountability Episode Builder* — **exécuté localement
   (Ollama), sans appel API** — les classe en épisodes d'accountability, pratiques ordinaires ou
   cas incertains, puis un validateur déterministe vérifie chaque citation.
   Lancée uniquement sur action explicite, après l'étape 3 ;
5. **étape 5** — *configuration et trajectoire intra-entretien* : à partir des
   épisodes de l'étape 4 (jamais de l'entretien entier), une préparation
   déterministe (ancrages temporels explicites, régularités) puis le *Trajectory
   Mapper* — **exécuté localement (Ollama)**, un appel par entretien — décrivent ce qui se répète, reste
   stable, varie selon les tâches, fait exception, reste en tension, change
   explicitement dans le temps, ou est raconté sans justification ; un validateur
   déterministe requalifie toute « évolution » fondée sur le seul ordre de
   l'entretien. Un entretien à la fois, aucune comparaison entre entretiens ;
6. **étape 6** — *comparaison inter-entretiens* : à partir des seules sorties
   validées de l'étape 5 de plusieurs entretiens (importées depuis des fichiers
   ou reprises du run, jamais les transcriptions), un contrôle du corpus, une
   préparation déterministe puis le *Cross-Interview Comparator* — **exécuté
   localement (Ollama)**, un appel pour tout le corpus — décrivent régularités, variantes, contrastes, cas
   négatifs et configurations minoritaires des frontières, critères du métier
   d'étudiant et manières de rendre compte. Comptes en entretiens, non-observation
   distinguée de l'absence, aucune typologie de personnes ; un validateur
   déterministe vérifie chaque appui.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # facultatif : modèle, réglages d'Ollama, tailles des blocs
```

`requirements.txt` suffit (Streamlit, pypdf, python-docx, Pydantic, pytest) : aucun SDK de modèle, aucune clé ni
variable obligatoire. Puis, une fois : installer [Ollama](https://ollama.com/download) et le modèle recommandé :

```bash
ollama pull qwen2.5:7b
```

## Étapes 3 à 6 en local (Ollama)

Détails : [`docs/local_runtime.md`](docs/local_runtime.md). Dans Streamlit, chaque section (« Analyse IA — Étape
3 », « Étape 4 », « Étape 5 », « Étape 6 ») a un bouton « Exécuter l'étape N (local, Ollama) » : TRACE prépare
le matériau, appelle le modèle local pour chaque agent (prompt `prompts/*.md`, JSON Schema strict), valide la
réponse (Pydantic puis validateur de l'étape ; corrections locales si besoin), enregistre les sorties habituelles
et les affiche. L'étape s'exécute **en arrière-plan** (processus indépendant de la page) : la page affiche l'agent en
cours, le temps écoulé et la liste des appels ; elle peut être rafraîchie ou fermée, et une exécution interrompue
reprend au premier appel non terminé (chaque réponse validée est enregistrée immédiatement, jamais recalculée).
Mesures et benchmark : `python scripts/trace_benchmark.py plan|run`. L'en-tête rappelle « Exécution : locale · Runtime : Ollama · Modèle : … · Données externes :
aucune » et l'état d'Ollama. Sans interface :

```bash
python scripts/trace_local.py runtime                          # Ollama actif ? modèle installé ?
python scripts/trace_local.py ingest chemin/entretien.docx     # run + ingestion (déterministe)
python scripts/trace_local.py run <run> --until 5              # étapes 3 → 5 (étape déjà complète : non rejouée)
python scripts/trace_local.py stage6 <run_A> <run_B> …          # étape 6 (aussi dossiers ou fichiers JSON)
python scripts/trace_local.py status <run>
```

Si Ollama est arrêté ou si le modèle manque, TRACE affiche une erreur claire (`ollama serve`, `ollama pull
<modèle>`) et n'écrit rien ; aucun autre fournisseur n'est jamais utilisé.

## Lancer l'application

```bash
ollama serve           # si Ollama n'est pas déjà lancé (l'application Ollama le fait aussi)
streamlit run app.py
```

## Lancer les tests

```bash
python -m pytest
```

Tests ciblés :

```bash
python -m pytest tests/test_document_extractor.py     # extraction TXT / DOCX / PDF
python -m pytest tests/test_transcript_structurer.py  # tours de parole
python -m pytest tests/test_ingestion.py              # couverture, robustesse, runs
python -m pytest tests/test_llm_client.py             # réponses d'agent : JSON, schéma, erreurs ; schémas stricts ; aucun SDK
python -m pytest tests/test_local_agent_runner.py     # runner local : Ollama, sortie structurée, corrections, erreurs
python -m pytest tests/test_local_pipeline.py         # étapes 3 à 6 en local : gardes, restaurations, mêmes sorties
python -m pytest tests/test_local_cli.py              # CLI locale
python -m pytest tests/test_local_performance.py      # fenêtre de contexte par appel, mesures, corrections, ordre
python -m pytest tests/test_local_jobs.py             # arrière-plan : processus détaché, refresh, reprise sans recalcul
python -m pytest tests/test_stage3_selectivity.py     # étape 3 : pratiques liées à une IAG, contradictions démontrées
python -m pytest tests/test_stage3_repair.py          # étape 3 : citations corrigées sans modèle, 1 réparation max
python -m pytest tests/test_evidence_validator.py     # validation des citations
python -m pytest tests/test_analysis_cache.py         # cache des analyses
python -m pytest tests/test_analysis_pipeline.py      # indépendance, agents en parallèle, échecs isolés
python -m pytest tests/test_app_stage3.py             # interface de l'étape 3 (AppTest)
python -m pytest tests/test_speaker_attribution_auditor.py  # étape 3.5 : audit des locuteurs
python -m pytest tests/test_interpretation_guard.py   # étape 3.5 : « réparation » en contexte
python -m pytest tests/test_stage3_5_synthetic.py     # étape 3.5 : scénario global synthétique
python -m pytest tests/test_interaction_chunking.py   # étape 3.6 : Interaction Reader par blocs
python -m pytest tests/test_practice_chunking.py      # étape 3.7 : Practice Extractor par blocs, fusion
python -m pytest tests/test_signal_selectivity.py     # étape 3.7 : pertinence, micro-marqueurs
python -m pytest tests/test_stage3_7_warnings.py      # étape 3.7 : les 5 avertissements du vrai run
python -m pytest tests/test_stage3_7_synthetic.py     # étape 3.7 : entretien long synthétique (330 tours)
python -m pytest tests/test_accountability_candidates.py  # étape 4 : candidats déterministes
python -m pytest tests/test_accountability_validator.py   # étape 4 : validateur des épisodes
python -m pytest tests/test_stage4_pipeline.py        # étape 4 : référence, cache, FAILED / PARTIAL
python -m pytest tests/test_stage4_long.py            # étape 4 : entretien long synthétique (320 tours)
python -m pytest tests/test_app_stage4.py             # étape 4 : interface (AppTest)
python -m pytest tests/test_stage3_restore.py         # restauration de sorties de l'étape 3 téléchargées
python -m pytest tests/test_stage3_refilter_restore.py # refilter canonique : réouverture et restauration → étape 4
python -m pytest tests/test_stage4_1.py               # étape 4.1 : proximité, fusions, requête normalisée
python -m pytest tests/test_stage4_blocks.py          # étape 4.2 : blocs de 4 candidats, fusion, reprise, troncature
python -m pytest tests/test_stage4_repair.py          # étape 4.3 : réparations ciblées (candidat, épisode, invariants)
python -m pytest tests/test_trajectory_candidates.py  # étape 5 : préparation déterministe, ancrages temporels
python -m pytest tests/test_trajectory_validator.py   # étape 5 : validateur (requalifications, vocabulaire)
python -m pytest tests/test_stage5_sociological.py    # étape 5 : cas A à H (temporalité, contexte, exception…)
python -m pytest tests/test_stage5_pipeline.py        # étape 5 : blocages, cache, un appel par entretien, échecs
python -m pytest tests/test_stage5_long.py            # étape 5 : entretien long « OTMANE-like »
python -m pytest tests/test_stage4_restore.py         # restauration de sorties de l'étape 4 téléchargées
python -m pytest tests/test_app_stage5.py             # étape 5 : interface (AppTest)
python -m pytest tests/test_cross_interview_corpus.py    # étape 6 : import du corpus, contrôles d'intégrité
python -m pytest tests/test_cross_interview_material.py  # étape 6 : préparation déterministe, représentation, tailles
python -m pytest tests/test_cross_interview_validator.py # étape 6 : validateur (appuis, comptes, revues, vocabulaire)
python -m pytest tests/test_stage6_sociological.py    # étape 6 : cas A à J (régularité, cas négatif, non-observation…)
python -m pytest tests/test_stage6_pipeline.py        # étape 6 : cache, invalidation, échecs, 17 entretiens
python -m pytest tests/test_app_stage6.py             # étape 6 : interface (AppTest)
python scripts/stage6_payload_report.py              # étape 6 : tailles de la représentation (2 / 8 / 17), 0 appel
```

Les tests n'utilisent que des documents **synthétiques** générés à la volée
(`tests/synthetic_docs.py`, `tests/synthetic_interviews.py`, `tests/synthetic_stage4.py`…) : aucun vrai
entretien n'est versionné. **Aucun test n'utilise un vrai modèle, une API ni le réseau** : les agents sont simulés
(faux Ollama derrière le vrai runner local, `tests/fake_llm.FakeLocalAgentRunner` ; `FakeAgents` pour
l'orchestration) et `tests/conftest.py` fait échouer immédiatement toute connexion réseau, même vers un vrai Ollama.

Test navigateur (Playwright + Chromium, faux Ollama derrière le vrai runner local, lancé à la main ;
nécessite `pip install playwright` et un Chromium, hors `requirements.txt`) :

```bash
python tests/e2e/browser_check.py               # scénarios A à P
python tests/e2e/browser_check.py --only-stage6  # étape 6 seulement (N à P)
```

Démonstration du parcours complet des étapes 3 à 6 **sans modèle réel** (faux Ollama, agents simulés) : `streamlit run tests/e2e/fake_llm_app.py` (pour l'étape 4, importer l'entretien
synthétique `Entretien_etape_4.txt`, écrit par `tests/synthetic_stage4.py`, ou laisser le test navigateur le
faire).

## Fonctionnement

1. Saisir (facultatif) la problématique de recherche et la sauvegarder.
2. Importer un ou plusieurs entretiens (PDF, DOCX, TXT).
3. Cliquer sur **Lancer l'analyse** :
   création du run → copie sûre des fichiers → extraction → structuration → contrôle qualité.
4. Pour chaque fichier, l'interface affiche le statut d'ingestion, le nombre de
   tours et d'avertissements, un aperçu des premiers tours de parole, et permet
   de télécharger `structured_transcript.json` et `ingestion_report.json`.

5. **Étape 3 (facultative, sur action explicite)** : dans la section
   « Analyse IA — Étape 3 », choisir « Test — un entretien » ou « Corpus
   complet », puis cliquer sur « Exécuter l'étape 3 (local, Ollama) ». TRACE appelle le modèle local (un appel
   d'audit des locuteurs seulement si des tours suspects sont détectés, puis un appel par agent ou par bloc),
   valide chaque réponse et enregistre les sorties. L'interface affiche alors les statuts
   de l'audit et des deux agents, le nombre de tours suspects et à vérifier, de
   pratiques, de signaux et de citations invalides, un
   aperçu, les avertissements de locuteur (« Ces suggestions ne modifient pas
   la transcription originale. ») et les téléchargements JSON. Une étape déjà terminée et à jour n'est pas rejouée.

6. **Étape 4 (facultative, sur action explicite, après l'étape 3)** : dans la
   section « Étape 4 — Épisodes d'accountability », choisir un entretien (ou
   tous), puis cliquer sur « Exécuter l'étape 4 (local, Ollama) » (candidats déterministes, un appel au modèle
   local par bloc de candidats, entretiens bloqués signalés). L'interface affiche le statut
   de l'étape 3, le nombre de candidats, d'épisodes d'accountability, de
   pratiques ordinaires (examinées et sans marqueur), d'incertains et
   d'avertissements, un aperçu des 3 premiers épisodes et les téléchargements
   `accountability_episodes.json` et `accountability_episode_validation.json`.

7. **Étapes 5 et 6** : mêmes boutons « Exécuter l'étape N (local, Ollama) » dans les sections « Étape 5 » (par entretien) et
   « Étape 6 » (sur un corpus de sorties de l'étape 5 importées ou reprises du run).

L'ingestion ne déclenche jamais d'agent. Aucune étape n'appelle une API ; seul le modèle local (Ollama) est utilisé.

Chaque run produit :

```
data/inputs/<run_id>/                       copies des fichiers importés (jamais modifiées)
data/outputs/<run_id>/metadata.json         run, fichiers, état du pipeline, résumé d'ingestion
data/outputs/<run_id>/problematique.txt
data/outputs/<run_id>/interviews/<interview_id>/
    raw_text.txt                            texte brut extrait
    structured_transcript.json              tours de parole
    ingestion_report.json                   rapport de qualité
    analysis/                               étape 3 (si lancée)
        speaker_attribution_audit.json      audit des locuteurs (étape 3.5) : tours suspects, suggestions
        speaker_audit_manifest.json         version, empreintes, moteur, champs legacy (0 appel API)
        practice_extractor.json             pratiques, citations annotées
        interaction_signals.json            signaux interactionnels, citations annotées
        practice_manifest.json              agent, versions, empreintes, moteur, champs legacy (0 appel API)
        interaction_manifest.json
        evidence_validation.json            validation déterministe des citations
        accountability_episodes.json        étape 4 : candidats, épisodes, pratiques sans marqueur
        accountability_episode_manifest.json  étape 4 : version, empreintes des sources, moteur, champs legacy
        accountability_episode_validation.json  étape 4 : validation déterministe des épisodes
data/outputs/<run_id>/local_runs/stage<N>/  bilan de l'étape (status.json) et journal de chaque appel au modèle local
data/cache/analysis/<agent>/<clé>.json      cache des réponses validées (même requête, même modèle : réutilisée)
```

## Couche d'ingestion

**Rôle.** Transformer chaque fichier d'entretien en une représentation
structurée, traçable et vérifiable, sur laquelle les futurs agents pourront
s'appuyer en revenant toujours au matériau brut. Tout est déterministe : règles
explicites, aucune IA, aucune interprétation.

**Formats supportés.**

| Format | Bibliothèque | Détail |
|---|---|---|
| TXT | Python | UTF-8 (avec/sans BOM), UTF-16 avec BOM ; repli cp1252 puis latin-1 avec avertissement |
| DOCX | python-docx | paragraphes dans l'ordre du document, tableaux ligne par ligne |
| PDF | pypdf | texte page par page, numéro de page conservé ; **pas d'OCR** |

Un PDF sans texte (scanné) n'est jamais « complété » : il est marqué `FAIL`
avec `NO_TEXT_EXTRACTED` et nécessite une vérification manuelle.

**Conservation du matériau brut.** Trois niveaux sont conservés séparément :
le fichier source (copie + SHA-256 revérifié), le texte brut extrait
(`raw_text.txt`) et les tours de parole. Rien n'est corrigé : fautes,
hésitations (« euh »), répétitions, oralité et grossièretés sont conservées.
Les seules transformations sont techniques et documentées : fins de ligne
normalisées en `\n`, BOM retiré, marqueur de locuteur (`Enquêteur :`) déplacé
du texte vers le champ `marker`, blancs retirés en début et fin de tour.

**Détection des locuteurs.** Par règles explicites uniquement (libellés
reconnus en début de ligne suivis de `:`) ; en cas de doute le locuteur est
`unknown`, jamais deviné. Détail des règles : [`docs/data_model.md`](docs/data_model.md).

**Contrôle de couverture.** Après structuration, le texte brut et les tours
(marqueurs inclus) sont comparés hors blancs : ils doivent être identiques.
Toute perte (> 1 % des mots) ou tout ajout rend l'entretien `FAIL`.

**Rapport de qualité.** `ingestion_report.json` donne le nombre de caractères,
de tours (enquêteur / enquêté / unknown), de pages et de pages vides, la part
approximative du texte attribuée à un locuteur, la couverture, les
avertissements et le statut `PASS`, `PASS_WITH_WARNINGS` ou `FAIL`. Aucune
« précision » n'est revendiquée faute de vérité terrain.

**Avertissements.** Chaque avertissement a un code, une gravité (`info`,
`warning`, `error`) et un message. `error` → `FAIL` ; `warning` →
`PASS_WITH_WARNINGS` (vérification manuelle recommandée). Liste complète :
[`docs/data_model.md`](docs/data_model.md#codes-davertissement).

**Format `structured_transcript.json`** (extrait) :

```json
{
  "schema_version": "1.0",
  "interview_id": "ELOISE",
  "source": {"filename": "Eloise.pdf", "sha256": "…", "format": "pdf", "page_count": 12},
  "turn_count": 214,
  "turns": [
    {
      "turn_id": "ELOISE_T0001",
      "index": 1,
      "speaker": "enqueteur",
      "speaker_raw": "Enquêteur",
      "marker": "Enquêteur :",
      "text": "Est-ce que tu peux te présenter ?",
      "source": {"file": "Eloise.pdf", "page": 1, "page_end": 1, "line_start": 3, "line_end": 3}
    }
  ],
  "warnings": []
}
```

## Étape 3 — analyse IA

Documentation complète : [`docs/agents_stage3.md`](docs/agents_stage3.md) et,
pour l'audit des locuteurs, [`docs/speaker_attribution_audit.md`](docs/speaker_attribution_audit.md).

- **Indépendance** : chaque agent reçoit ses propres consignes, son propre
  schéma et le même entretien compact ; jamais la sortie de l'autre.
- **Speaker Attribution Auditor** (étape 3.5) : règles déterministes puis, s'il
  y a des tours suspects, UN appel sur un extrait (candidats + voisins) ; ne
  modifie jamais la transcription ; les tours douteux sont signalés aux deux
  agents par un `speaker_warning`, le locuteur officiel restant inchangé.
- **Practice Extractor** : descriptif et « aveugle » à la théorie ; reprend
  les raisons données par l'enquêté·e, n'en invente pas. Cherche les usages ET
  les non-usages / refus (`non_use_reason`), y compris contradictoires sur un
  même thème, sans les fusionner ; décrit aussi les usages personnels et
  professionnels (`practice_domain`). Un entretien long est lu en blocs qui se
  chevauchent, puis les pratiques sont fusionnées sans LLM, sans jamais réunir
  deux conduites différentes (étape 3.7, [`docs/stage3_7.md`](docs/stage3_7.md)).
- **Interaction Signal Reader** : relève des marques observables (hésitation,
  autocorrection, minimisation, affect explicitement nommé, référence au
  jugement d'un·e enseignant·e, contradiction entre tours, préférence,
  évaluation métadiscursive de sa propre formulation…) sans inférer
  d'état psychologique. Un entretien long est lu en blocs qui se chevauchent,
  plus une lecture légère des passages éloignés, puis fusionné sans LLM
  (étape 3.6, [`docs/interaction_chunking.md`](docs/interaction_chunking.md)) ;
  un bloc en échec (réponse non conforme) rend l'analyse `PARTIAL`, jamais présentée comme complète.
  Étape 3.7 : seuls les phénomènes **pertinents pour le récit des pratiques**
  sont relevés (pas d'inventaire de « euh », « juste », « un peu »…) ; une
  couche déterministe complète les `turn_ids` et écarte, en les conservant à
  part (`set_aside_signals`), les micro-marqueurs isolés ou redondants et les
  signaux appuyés sur la seule question de l'enquêteur.
- **Preuves** : chaque pratique et chaque signal cite l'entretien ; un
  validateur déterministe vérifie que chaque citation est une sous-chaîne
  exacte du tour cité, que les `turn_id` existent et appartiennent à cet
  entretien, que les intervalles sont ordonnés. Rien n'est corrigé : les
  objets douteux sont marqués `needs_review`.
- **Sécurité** : la transcription est une donnée, jamais une instruction ;
  elle est encadrée et échappée ; la réponse de l'agent doit respecter un schéma JSON
  strict, puis elle est revalidée (Pydantic, validateurs déterministes).
- **Garde-fou** : vocabulaire interprétatif signalé dans les champs rédigés ;
  « réparation » est examinée en contexte (la réparation d'un lave-vaisselle
  n'est pas signalée, une « réparation discursive » l'est).
- **Garde contre les réexécutions** : une étape terminée et calculée sur les sorties actuelles de l'étape
  précédente n'est pas rejouée ; le cache (clé : prompt, message, schéma, modèle, paramètres) fait qu'après un
  échec partiel seuls les appels manquants sont refaits, et qu'un agent modifié est seul relancé.
- **Coût** : aucun — exécution locale, 0 appel API. Les champs `api_calls` et `billed_this_run` des manifests
  sont conservés pour la compatibilité des formats et valent toujours 0 / `false` ; `usage` donne les tokens
  traités par le modèle local.
- **Configuration** : aucune variable obligatoire ; `TRACE_LOCAL_MODEL` (défaut `qwen2.5:7b`), `TRACE_OLLAMA_URL`
  (défaut `http://localhost:11434`, boucle locale uniquement), voir `docs/local_runtime.md` ; `TRACE_INTERACTION_CHUNK_TOKENS` (défaut 5 000) et
  `TRACE_PRACTICE_CHUNK_TOKENS` (défaut 4 000) fixent la taille d'un bloc de chaque agent.

## Étape 4 — épisodes d'accountability

Documentation complète : [`docs/stage4_accountability_episodes.md`](docs/stage4_accountability_episodes.md).

- **Sens du terme** : accountability au sens ethnométhodologique — rendre une
  conduite descriptible, intelligible, reconnaissable. Aucune motivation cachée,
  aucune « stratégie » : on décrit ce que le texte fait (« l'étudiant limite
  explicitement l'usage à la reformulation »).
- **Candidats déterministes** (`core/accountability_candidates.py`, sans LLM) :
  pratique + signal proche (même tour, ≤ 2 positions, même échange), contradiction
  entre tours rattachée aux seuls tours cités, usage / non-usage d'une même tâche,
  frontière explicite (« mais pas », « à ma place », « moi-même », « sauf »…).
  Jamais à partir d'un « euh », d'une intensification isolée ou de la seule question
  de l'enquêteur.
- **Accountability Episode Builder** (prompt versionné 1.0) : pour chaque candidat,
  `accountability_episode`, `ordinary_practice` (racontée comme allant de soi) ou
  `uncertain` ; opérations (`restriction`, `exception`, `general_rule`, `distinction`,
  `refusal`, `appeal_to_external_judgment`, `self_evaluation`…) et frontières
  explicites (« faire / faire faire »). Une pratique sans aucun marqueur n'est pas
  envoyée au modèle (pratique ordinaire « sans marqueur »).
- **Validateur déterministe** : identifiants, citations mot pour mot, voix de
  l'enquêté·e, opérations, pratiques ordinaires « étoffées », fusions abusives,
  avertissements de locuteur propagés, vocabulaire psychologisant interdit.
- **Étape 3 incomplète** : FAILED → étape 4 `BLOCKED` (aucun appel) ; PARTIAL →
  étape 4 `PARTIAL`, `analysis_complete: false`.
- **Appels au modèle local** : aucun sans candidat, sinon un par bloc d'au plus 4 candidats (composantes
  entières ; la réponse, qui recopie les citations, reste sous la réserve de sortie) ; 0 appel API ; modifier
  l'étape 4 ne relance qu'elle. Chaque bloc validé est mis en cache : une troncature ou un arrêt ne fait rejouer
  que les blocs manquants. Requête normalisée (chaque pratique, signal, citation et tour une seule fois,
  identifiants abrégés). Coût estimé ou mesuré : `python scripts/stage4_blocks_report.py plan|measure RUN`.
  Anomalie d'un épisode ou candidat sans disposition : réparation ciblée de ce seul candidat, jamais du bloc
  (`python scripts/stage4_blocks_report.py repairs RUN` les prévoit sans appel).
- **Étape 4.1** : proximité mesurée dans le tour (≤ 200 caractères entre citations) ;
  fusion seulement entre candidats reliés explicitement (sinon `DISCONNECTED_MERGE`,
  épisode rejeté et non utilisable) ; affects et intentions jamais employés comme
  catégorie sauf s'ils sont dits par l'enquêté·e, mot de l'enquêteur jamais attribué
  à l'étudiant·e.

## Étape 5 — configuration et trajectoire intra-entretien

Documentation complète : [`docs/stage5_trajectory.md`](docs/stage5_trajectory.md).

- **Un entretien à la fois** : quelles frontières, règles et manières de rendre compte de ses usages se
  répètent, restent stables, varient selon les tâches ou contextes, comportent des exceptions, entrent en
  tension, ou changent explicitement dans le temps ; quelles zones sont racontées sans justification ; quels
  critères du « métier d'étudiant » sont explicitement mobilisés. Aucune comparaison entre entretiens,
  aucune typologie.
- **Ordre de l'entretien ≠ ordre biographique** : un `explicit_temporal_change` exige deux états
  comparables, une différence documentée et des ancrages explicites de l'enquêté·e qui les ordonnent (« au
  lycée » / « maintenant », « je ne … plus »…) ; sinon le validateur le **requalifie** en variation
  contextuelle, et une configuration `temporal_trajectory` / `mixed` sans changement validé est requalifiée.
- **Préparation déterministe** (`core/trajectory_candidates.py`) : épisodes `usable_for_next_stages` et
  pratiques sans marqueur, ancrages temporels repérés dans les tours de l'enquêté·e, régularités (opérations
  et frontières répétées, mêmes tâches, règle + cas, tensions signalées, pratiques ordinaires) ; représentation
  normalisée.
- **Validateur** (`core/trajectory_validator.py`) : appuis utilisables, tours de l'enquêté·e, ancrages exacts,
  exception = règle + cas, propagation des épisodes à revoir (une affirmation qui ne repose que sur eux est à
  revoir), vocabulaire psychologisant ou de récit de conversion, mot de l'enquêteur jamais attribué.
- **Appels au modèle local** : aucun sans matériau suffisant, sinon 1 par entretien (0 appel API) ; une étape 5 terminée et
  à jour n'est pas rejouée.
- **Sorties** : `student_trajectory.json`, `student_trajectory_validation.json`, `student_trajectory_manifest.json`.

## Étape 6 — comparaison inter-entretiens

Documentation complète : [`docs/stage6_cross_interview.md`](docs/stage6_cross_interview.md).

- **Constituer le corpus Stage 6** (section « Étape 6 », disponible même sans run) : importer les 3 JSON de
  l'étape 5 de chaque entretien et/ou reprendre ceux du run courant. Reconnaissance par le contenu, regroupement
  par `interview_id`, contrôles (versions, analyse complète, `validation_error_count = 0`, empreintes, comptes et
  revues recalculés, doublons ambigus) ; un entretien invalide est exclu avec sa raison. Tableau
  `interview_id | statut | claims | needs_review | configuration | importé`, N importés / N exploitables.
  Moins de 2 entretiens exploitables : bloqué ; 2 : exploratoire ; 3 et plus : comparatif. Aucun appel.
- **Préparation déterministe** (`core/cross_interview_material.py`) : affirmations et critères utilisables,
  `review_reasons`, index (familles présentes / non observées, opérations, contextes, frontières, exceptions,
  tensions, zones ordinaires), regroupements à formulation identique seulement ; représentation normalisée
  (≈ 15 % des JSON de l'étape 5 ; ≈ 9 400 tokens pour 17 entretiens synthétiques).
- **Validateur** (`core/cross_interview_validator.py`) : aucun appui inventé, comptes recalculés en entretiens,
  positions `explicit_presence` / `explicit_refusal` / `contrary_case` / `not_observed`, éléments à revoir
  secondaires (jamais une régularité forte), cas négatifs toujours conservés, régularité sur un seul entretien
  requalifiée, pas de typologie, de causalité ni de généralisation au-delà du N observé.
- **Appels au modèle local** : 1 par corpus (pas de MAP → REDUCE : non justifié par la mesure), 0 appel API ; un corpus déjà
  analysé à l'identique n'est pas rejoué ; ajouter ou modifier un entretien n'invalide que l'étape 6.
- **Sorties** (`data/outputs/cross_interview/<corpus_id>/analysis/`) : `cross_interview_comparison.json`,
  `cross_interview_validation.json`, `cross_interview_manifest.json`.

## Restaurer une étape 4 déjà calculée

Après avoir restauré l'étape 3 (section suivante), la section « Étape 5 » propose « Restaurer des résultats
Stage 4 existants » : importer `accountability_episodes.json` et `accountability_episode_validation.json`.
`core/stage4_restore.py` vérifie le même entretien, les versions, une analyse complète sans erreur de
validation, les empreintes de l'étape 3 installée, puis **recalcule** candidats et validation des épisodes
pour refuser tout fichier altéré ; les fichiers sont recopiés octet pour octet et l'interface affiche « Stage 4
restauré depuis fichiers — 0 appel API ». L'étape 5 peut alors être lancée directement, sans relancer l'étape 4.

## Restaurer une étape 3 déjà calculée

**Le stockage local de l'application n'est pas durable.** Les runs (`data/outputs/`) vivent sur le disque de la
machine qui exécute Streamlit ; sur Streamlit Cloud, un redéploiement (fusion d'une PR, redémarrage) les
efface. Un entretien réimporté crée alors un nouveau run sans sortie de l'étape 3 : il faudrait refaire tous
les appels de l'étape 3, et l'étape 4 demande de la relancer. **Téléchargez donc toujours les JSON de
l'étape 3.**

Pour les réutiliser sans aucun appel API : réimporter l'entretien et lancer l'ingestion, puis, dans la
section « Étape 4 », ouvrir « Restaurer des résultats Stage 3 existants », choisir l'entretien et importer
les 4 fichiers téléchargés (`practice_extractor.json`, `interaction_signals.json`,
`evidence_validation.json`, `speaker_attribution_audit.json`, préfixés ou non par l'identifiant de
l'entretien). Le module `core/stage3_restore.py` :

- reconnaît chaque fichier par son contenu ; refuse un fichier manquant, en double, illisible ou inconnu ;
- exige le même `interview_id` dans les quatre fichiers, égal à celui de l'entretien choisi (aucun mélange
  d'entretiens) ;
- exige pour Practice Extractor et Interaction Reader la version de schéma actuelle, un statut exploitable
  (SUCCESS, SUCCESS_WITH_WARNINGS, CACHED) et une analyse complète (jamais PARTIAL / FAILED /
  `analysis_complete: false`) ; une autre version d'agent est acceptée avec un avertissement ;
- compare l'empreinte de `structured_transcript.json` enregistrée par l'audit des locuteurs à celle de
  l'entretien réimporté : des sorties calculées sur un autre fichier ou une autre segmentation des tours
  (par exemple avant le correctif de segmentation) sont refusées ;
- revérifie chaque citation sur la transcription actuelle (tour inexistant, citation déclarée valide qui
  ne l'est plus → refus) et refuse un `evidence_validation.json` avec des anomalies critiques, des totaux
  incohérents ou plus de 10 % de citations invalides.

Une fois validés, les fichiers sont installés dans `analysis/` dans leur **version canonique** : la sélectivité
déterministe actuelle de l'étape 3 (celle que réapplique `trace_local.py refilter`) et la validation des preuves
sont réappliquées aux objets importés. Des fichiers téléchargés AVANT un `refilter` donnent donc exactement les
objets du run refiltré : aucune pratique ni aucun signal écarté n'est réintroduit, et l'étape 4 reçoit la même
étape 3. Des fichiers déjà à jour sont recopiés **octet pour octet**. Les manifests de restauration
(`stage3_restore_manifest.json` : nom importé, SHA-256 importé et installé, taille, `reselected`, bilan de la
resélection) en gardent la trace ; les sorties périmées de l'étape 4 sont retirées. Aucune étape 3, aucun audit des locuteurs, aucun appel :
l'interface affiche « Stage 3 restauré depuis fichiers — 0 appel API » et l'étape 4 lit ces fichiers comme
des sorties normales. Le cache de l'étape 3, lui, n'est pas reconstitué : relancer l'étape 3 sur cet
entretien (`--force`) referait ses appels au modèle local.

## Architecture

```
app.py                 interface Streamlit
core/config.py                 chemins, formats acceptés, étapes du pipeline
core/run_manager.py            création des runs, copie des fichiers, métadonnées
core/schemas.py                version du schéma, locuteurs, statuts, codes d'avertissement
core/document_extractor.py     extraction du texte brut (TXT, DOCX, PDF)
core/transcript_structurer.py  découpage en tours de parole (règles explicites)
core/ingestion_validator.py    contrôle de couverture, statistiques, rapport de qualité
core/ingestion.py              orchestration par fichier et par run
core/llm_client.py             réponse d'un agent : JSON validé par Pydantic, erreurs (AWAITING_AGENT…), paramètres
core/local_pipeline.py         étapes 3 à 6 en local : gardes, enchaînement 3 → 5, corpus de l'étape 6, bilans
core/local_agent_runner.py     LocalAgentRunner : Ollama local, JSON Schema, Pydantic, corrections locales, journal,
                               fenêtre de contexte par appel, mesures de chaque tentative
core/local_jobs.py             exécution en arrière-plan (processus détaché, job.json, battement), reprise
core/stage3_repair.py          étape 3 : objets fautifs seuls (déterministe d'abord, 1 réparation max), état persistant
core/citation_resolver.py      étape 3 : correction déterministe et conservatrice des citations (passage littéral)
core/agent_checks.py           contrôle méthodologique d'une réponse (validateurs existants) avant correction
core/analysis.py               étape 3 : deux agents en parallèle, sorties, manifests, statuts
core/analysis_cache.py         cache déterministe des analyses
core/evidence_validator.py     validation déterministe des citations
core/interpretation_guard.py   détection du vocabulaire interprétatif dans les sorties
core/speaker_attribution_auditor.py  étape 3.5 : audit des locuteurs (règles, extrait, revalidation)
core/interaction_chunking.py   étape 3.6 : blocs, sélection à longue distance, fusion (Interaction Reader)
core/practice_chunking.py      étape 3.7 : blocs et fusion déterministe des pratiques (Practice Extractor)
core/signal_selectivity.py     étape 3.7 : turn_ids complétés, micro-marqueurs et appuis « enquêteur » écartés
core/accountability_candidates.py  étape 4 : candidats d'épisodes déterministes, représentation compacte
core/accountability_episode_validator.py  étape 4 : validation déterministe des épisodes
core/accountability.py         étape 4 : orchestration, états de l'étape 3, cache, sorties
core/stage3_restore.py         restauration validée de sorties de l'étape 3 téléchargées (0 appel)
core/trajectory_candidates.py  étape 5 : préparation déterministe (ancrages temporels, régularités, représentation)
core/trajectory_validator.py   étape 5 : validation déterministe (requalifications, propagation, vocabulaire)
core/trajectory.py             étape 5 : orchestration, état de l'étape 4, cache, sorties
core/stage4_restore.py         restauration validée de sorties de l'étape 4 téléchargées (0 appel)
core/cross_interview_corpus.py    étape 6 : import et contrôle des triplets de l'étape 5 (0 appel)
core/cross_interview_material.py  étape 6 : préparation déterministe (index, représentation normalisée)
core/cross_interview_validator.py étape 6 : validation déterministe (appuis, comptes, revues, cas négatifs)
core/cross_interview.py        étape 6 : orchestration, seuil, cache, sorties par corpus
agents/base.py                 citation, identité versionnée d'un agent, entretien compact
agents/practice_extractor.py   agent 1 : version, schéma de sortie
agents/interaction_signal_reader.py  agent 2 : version, schéma de sortie
agents/accountability_episode_builder.py  étape 4 : version, schéma d'un épisode, taxonomie des opérations
agents/trajectory_mapper.py    étape 5 : version, schéma des affirmations et des critères
agents/cross_interview_comparator.py  étape 6 : version, schéma des affirmations inter-entretiens
prompts/practice_extractor.md  consignes de l'agent 1
prompts/interaction_signal_reader.md  consignes de l'agent 2
prompts/speaker_attribution_auditor.md  consignes de l'auditeur des locuteurs
prompts/interaction_long_distance_reader.md  consignes de la lecture à longue distance (entretien long)
prompts/accountability_episode_builder.md  consignes de l'Accountability Episode Builder (étape 4, v1.0)
prompts/trajectory_mapper.md   consignes du Trajectory Mapper (étape 5, v1.0)
prompts/cross_interview_comparator.md  consignes du Cross-Interview Comparator (étape 6, v1.0)
scripts/stage6_payload_report.py  mesure de la représentation de l'étape 6 (aucun appel)
scripts/trace_local.py         CLI locale (runtime, ingestion, étapes 3 à 6, état), sans interface
CLAUDE.md                      notes pour le développement du dépôt (inutile pour utiliser TRACE)
docs/local_runtime.md          exécution locale : Ollama, modèle, réglages, corrections, erreurs, fichiers
docs/data_model.md             format des données transmis aux agents
docs/agents_stage3.md          étape 3 : méthode, schémas, validation, cache, configuration
docs/speaker_attribution_audit.md  étape 3.5 : audit de l'attribution des locuteurs
docs/interaction_chunking.md   étape 3.6 : Interaction Reader sur les entretiens longs
docs/stage3_7.md               étape 3.7 : Practice Extractor par blocs, sélectivité, avertissements
docs/stage4_accountability_episodes.md  étape 4 : épisodes d'accountability
docs/stage5_trajectory.md      étape 5 : configuration et trajectoire intra-entretien
docs/stage6_cross_interview.md étape 6 : comparaison inter-entretiens
data/inputs|outputs/   données des runs (non versionnées)
logs/                  journal trace.log (non versionné)
tests/                 tests automatiques (pytest), documents synthétiques uniquement
```

## Sécurité

- Aucun secret : TRACE n'utilise ni clé ni API ; `.env` (réglages facultatifs) est ignoré par git.
- Aucune donnée ne quitte l'ordinateur : TRACE n'appelle qu'Ollama sur la boucle locale (toute autre adresse est
  refusée, les proxys ne sont jamais utilisés) ; Internet n'est pas nécessaire après l'installation.
- Les données d'entretien (`data/inputs/**`, `data/outputs/**`), le cache
  d'analyse (`data/cache/**`) et les journaux (`logs/**`) ne sont pas versionnés.
- Les journaux ne contiennent ni transcript ni citation : identifiants et
  statuts seulement.

## Limites connues de l'ingestion

- Pas d'OCR : un PDF scanné est signalé, pas lu.
- PDF : l'ordre de lecture dépend de pypdf (mises en page en colonnes, en-têtes et
  numéros de page peuvent se retrouver dans les tours) ; les lignes vides n'existent
  pas dans un PDF, ce qui influence le découpage des segments sans marqueur.
- DOCX : en-têtes / pieds de page, notes, commentaires et zones de texte ne sont pas extraits.
- Seuls les libellés listés sont reconnus. Un marqueur explicite (Enquêteur, Enquêté,
  Interviewer, Interviewé…) au milieu d'une ligne ouvre un nouveau tour ; un libellé
  ambigu (Q, R, Question, Réponse, Participant) en milieu de ligne est seulement signalé.
  Garde-fou : un tour de plus de 4 000 caractères contenant encore au moins 2 marqueurs
  internes rend l'entretien `FAIL` (jamais envoyé aux agents IA).
- Un texte non attribué situé après le premier marqueur (ex. « Fin de l'entretien »)
  est rattaché au tour précédent, faute de règle permettant de le distinguer.
