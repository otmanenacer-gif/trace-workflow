# TRACE

**Analyse ethnométhodologique des usages étudiants des IAG**

Outil expérimental d'analyse qualitative assistée par IA, destiné à analyser
des entretiens semi-directifs. **Version actuelle : ingestion déterministe
des entretiens** — aucun agent IA, aucun appel à l'API Anthropic, aucune analyse.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # clé non utilisée pour l'instant
```

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
```

Les tests n'utilisent que des documents **synthétiques** générés à la volée
(`tests/synthetic_docs.py`) : aucun vrai entretien n'est versionné.

## Fonctionnement

1. Saisir (facultatif) la problématique de recherche et la sauvegarder.
2. Importer un ou plusieurs entretiens (PDF, DOCX, TXT).
3. Cliquer sur **Lancer l'analyse** :
   création du run → copie sûre des fichiers → extraction → structuration → contrôle qualité.
4. Pour chaque fichier, l'interface affiche le statut d'ingestion, le nombre de
   tours et d'avertissements, un aperçu des premiers tours de parole, et permet
   de télécharger `structured_transcript.json` et `ingestion_report.json`.

Seule l'étape « Structuration des entretiens » est active ; les autres restent inactives.

Chaque run produit :

```
data/inputs/<run_id>/                       copies des fichiers importés (jamais modifiées)
data/outputs/<run_id>/metadata.json         run, fichiers, état du pipeline, résumé d'ingestion
data/outputs/<run_id>/problematique.txt
data/outputs/<run_id>/interviews/<interview_id>/
    raw_text.txt                            texte brut extrait
    structured_transcript.json              tours de parole
    ingestion_report.json                   rapport de qualité
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
docs/data_model.md             format des données transmis aux futurs agents
agents/                        futurs agents (un module par étape)
prompts/               futurs prompts
data/inputs|outputs/   données des runs (non versionnées)
logs/                  journal trace.log (non versionné)
tests/                 tests automatiques (pytest), documents synthétiques uniquement
```

## Sécurité

- Aucun secret dans le dépôt : `.env` est ignoré par git.
- Les données d'entretien (`data/inputs/**`, `data/outputs/**`) et les journaux (`logs/**`) ne sont pas versionnés.

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
