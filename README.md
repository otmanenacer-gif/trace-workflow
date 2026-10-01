# TRACE

**Analyse ethnométhodologique des usages étudiants des IAG**

Outil expérimental d'analyse qualitative assistée par IA, destiné à analyser
des entretiens semi-directifs. **Version actuelle :**

1. ingestion **déterministe** des entretiens (sans IA) ;
2. **étape 3** — deux agents IA **indépendants**, lancés **en parallèle** et
   uniquement sur action explicite : *Practice Extractor* (ce que l'étudiant·e
   fait, ou ne fait pas, avec ou sans IAG) et *Interaction Signal Reader*
   (comment il ou elle le raconte). Aucune synthèse, aucune analyse théorique à ce stade ;
3. **étape 3.5** — *Speaker Attribution Auditor* : avant les deux agents,
   signale les tours dont le locuteur semble mal attribué (règles
   déterministes, puis au plus un appel LLM par entretien, aucun s'il n'y a
   pas de tour suspect). La transcription n'est **jamais** modifiée.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # facultatif : clé API et modèle pour l'étape 3
```

Sans `ANTHROPIC_API_KEY` ni `ANTHROPIC_MODEL`, l'application fonctionne
normalement pour l'ingestion ; l'analyse IA est désactivée avec un message explicite.

## Lancer l'application

```bash
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
python -m pytest tests/test_llm_client.py             # client LLM : config, réessais, réponses invalides
python -m pytest tests/test_evidence_validator.py     # validation des citations
python -m pytest tests/test_analysis_cache.py         # cache des analyses
python -m pytest tests/test_analysis_pipeline.py      # indépendance, parallélisme, échecs isolés
python -m pytest tests/test_app_stage3.py             # interface de l'étape 3 (AppTest)
python -m pytest tests/test_speaker_attribution_auditor.py  # étape 3.5 : audit des locuteurs
python -m pytest tests/test_interpretation_guard.py   # étape 3.5 : « réparation » en contexte
python -m pytest tests/test_stage3_5_synthetic.py     # étape 3.5 : scénario global synthétique
python -m pytest tests/test_interaction_chunking.py   # étape 3.6 : Interaction Reader par blocs
python -m pytest tests/test_practice_chunking.py      # étape 3.7 : Practice Extractor par blocs, fusion
python -m pytest tests/test_signal_selectivity.py     # étape 3.7 : pertinence, micro-marqueurs
python -m pytest tests/test_stage3_7_warnings.py      # étape 3.7 : les 5 avertissements du vrai run
python -m pytest tests/test_stage3_7_synthetic.py     # étape 3.7 : entretien long synthétique (330 tours)
```

Les tests n'utilisent que des documents **synthétiques** générés à la volée
(`tests/synthetic_docs.py`, `tests/synthetic_interviews.py`) : aucun vrai
entretien n'est versionné. **Aucun test n'appelle l'API Anthropic** : le LLM
est simulé et `tests/conftest.py` interdit le transport réel et toute
connexion réseau.

Test navigateur (Playwright + Chromium, LLM simulé, lancé à la main ;
nécessite `pip install playwright` et un Chromium, hors `requirements.txt`) :

```bash
python tests/e2e/browser_check.py
```

Démonstration locale du parcours complet de l'étape 3 **sans clé ni appel
réel** (LLM simulé) : `streamlit run tests/e2e/fake_llm_app.py`.

Test **réel** de l'étape 3 sur l'entretien synthétique (2 appels API,
**consomme des tokens**, exige une option explicite) ; `--stage35` utilise
l'entretien synthétique de l'étape 3.5 (3 appels : audit des locuteurs + 2 agents) :

```bash
python scripts/smoke_test_stage3.py --confirm-api-cost
python scripts/smoke_test_stage3.py --confirm-api-cost --stage35
```

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
   complet » (confirmation supplémentaire), puis cliquer sur « Lancer les deux
   analyses IA ». Le bouton indique le nombre d'appels API payants prévus
   (2 par entretien absent du cache, plus 1 d'audit des locuteurs seulement si
   des tours suspects sont détectés). L'interface affiche ensuite les statuts
   de l'audit et des deux agents, le nombre de tours suspects et à vérifier, de
   pratiques, de signaux et de citations invalides, les tokens consommés, un
   aperçu, les avertissements de locuteur (« Ces suggestions ne modifient pas
   la transcription originale. ») et les téléchargements JSON.

L'ingestion ne déclenche jamais d'appel IA. Les étapes du pipeline au-delà
de l'extraction des pratiques et de l'analyse interactionnelle restent inactives.

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
        speaker_audit_manifest.json         version, empreintes, modèle, cache, tokens de l'audit
        practice_extractor.json             pratiques, citations annotées
        interaction_signals.json            signaux interactionnels, citations annotées
        practice_manifest.json              agent, versions, empreintes, modèle, cache, tokens
        interaction_manifest.json
        evidence_validation.json            validation déterministe des citations
data/cache/analysis/<agent>/<clé>.json      cache global des réponses validées
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
  un bloc tronqué rend l'analyse `PARTIAL`, jamais présentée comme complète.
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
  elle est encadrée et échappée ; la sortie est contrainte par un schéma JSON
  puis revalidée.
- **Garde-fou** : vocabulaire interprétatif signalé dans les champs rédigés ;
  « réparation » est examinée en contexte (la réparation d'un lave-vaisselle
  n'est pas signalée, une « réparation discursive » l'est).
- **Cache** : une analyse n'est repayée que si le fichier, le transcript, le
  prompt, la version de l'agent, le schéma, le modèle ou les paramètres
  changent ; modifier un agent ou l'auditeur ne relance que lui.
- **Coût** : tokens d'entrée / sortie et nombre d'appels affichés ; aucun
  montant inventé.
- **Configuration** : `ANTHROPIC_API_KEY` et `ANTHROPIC_MODEL` (environnement
  ou `.env`) ; `TRACE_MAX_CONCURRENCY` (défaut 2) limite les appels simultanés ;
  `TRACE_INTERACTION_CHUNK_TOKENS` (défaut 5 000) et `TRACE_PRACTICE_CHUNK_TOKENS`
  (défaut 4 000) fixent la taille d'un bloc de chaque agent.

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
core/llm_client.py             client LLM (SDK Anthropic) : config, réessais, sortie JSON validée
core/analysis.py               étape 3 : deux agents en parallèle, sorties, manifests, statuts
core/analysis_cache.py         cache déterministe des analyses
core/evidence_validator.py     validation déterministe des citations
core/interpretation_guard.py   détection du vocabulaire interprétatif dans les sorties
core/speaker_attribution_auditor.py  étape 3.5 : audit des locuteurs (règles, extrait, revalidation)
core/interaction_chunking.py   étape 3.6 : blocs, sélection à longue distance, fusion (Interaction Reader)
core/practice_chunking.py      étape 3.7 : blocs et fusion déterministe des pratiques (Practice Extractor)
core/signal_selectivity.py     étape 3.7 : turn_ids complétés, micro-marqueurs et appuis « enquêteur » écartés
agents/base.py                 citation, identité versionnée d'un agent, entretien compact
agents/practice_extractor.py   agent 1 : version, schéma de sortie
agents/interaction_signal_reader.py  agent 2 : version, schéma de sortie
prompts/practice_extractor.md  consignes de l'agent 1
prompts/interaction_signal_reader.md  consignes de l'agent 2
prompts/speaker_attribution_auditor.md  consignes de l'auditeur des locuteurs
prompts/interaction_long_distance_reader.md  consignes de la lecture à longue distance (entretien long)
scripts/smoke_test_stage3.py   test réel (payant, sur confirmation) sur l'entretien synthétique
docs/data_model.md             format des données transmis aux agents
docs/agents_stage3.md          étape 3 : méthode, schémas, validation, cache, configuration
docs/speaker_attribution_audit.md  étape 3.5 : audit de l'attribution des locuteurs
docs/interaction_chunking.md   étape 3.6 : Interaction Reader sur les entretiens longs
docs/stage3_7.md               étape 3.7 : Practice Extractor par blocs, sélectivité, avertissements
data/inputs|outputs/   données des runs (non versionnées)
logs/                  journal trace.log (non versionné)
tests/                 tests automatiques (pytest), documents synthétiques uniquement
```

## Sécurité

- Aucun secret dans le dépôt : `.env` est ignoré par git. La clé est lue
  uniquement dans `ANTHROPIC_API_KEY` et n'est jamais affichée ni journalisée.
- Les données d'entretien (`data/inputs/**`, `data/outputs/**`), le cache
  d'analyse (`data/cache/**`) et les journaux (`logs/**`) ne sont pas versionnés.
- Les journaux ne contiennent ni transcript ni citation : identifiants,
  statuts, tokens et durées seulement.

## Limites connues de l'ingestion

- Pas d'OCR : un PDF scanné est signalé, pas lu.
- PDF : l'ordre de lecture dépend de pypdf (mises en page en colonnes, en-têtes et
  numéros de page peuvent se retrouver dans les tours) ; les lignes vides n'existent
  pas dans un PDF, ce qui influence le découpage des segments sans marqueur.
- DOCX : en-têtes / pieds de page, notes, commentaires et zones de texte ne sont pas extraits.
- Seuls les libellés listés sont reconnus ; un marqueur collé au milieu d'une ligne
  est signalé mais pas découpé.
- Un texte non attribué situé après le premier marqueur (ex. « Fin de l'entretien »)
  est rattaché au tour précédent, faute de règle permettant de le distinguer.
