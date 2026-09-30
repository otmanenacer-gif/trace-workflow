# TRACE

**Analyse ethnométhodologique des usages étudiants des IAG**

Outil expérimental d'analyse qualitative assistée par IA, destiné à analyser
des entretiens semi-directifs. **Version actuelle : squelette technique
uniquement** — aucun agent IA, aucun appel à l'API Anthropic, aucune analyse.

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

## Fonctionnement

1. Saisir (facultatif) la problématique de recherche et la sauvegarder.
2. Importer un ou plusieurs entretiens (PDF, DOCX, TXT).
3. Cliquer sur **Lancer l'analyse** : un run est créé avec un identifiant unique
   (`run_AAAAMMJJ_HHMMSS_xxxxxxxx`).

Chaque run produit :

```
data/inputs/<run_id>/     copies des fichiers importés (les originaux ne sont jamais modifiés)
data/outputs/<run_id>/    problematique.txt + metadata.json (futures sorties des agents)
```

`metadata.json` contient l'identifiant, la date, la problématique, la liste des
fichiers (nom, format, taille, empreinte SHA-256) et l'état des étapes du pipeline.

## Architecture

```
app.py                 interface Streamlit
core/config.py         chemins, formats acceptés, étapes du pipeline
core/run_manager.py    création des runs, copie des fichiers, métadonnées
agents/                futurs agents (un module par étape)
prompts/               futurs prompts
data/inputs|outputs/   données des runs (non versionnées)
logs/                  journal trace.log (non versionné)
tests/                 tests automatiques (pytest)
```

## Sécurité

- Aucun secret dans le dépôt : `.env` est ignoré par git.
- Les données d'entretien (`data/`) et les journaux (`logs/`) ne sont pas versionnés.
