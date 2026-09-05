# Assistant Documentaire — Comores

Pipeline RAG (Retrieval-Augmented Generation) découpé en étapes claires,
sur le modèle de l'atelier "Objectif-IA", pour répondre à des questions sur
les Comores à partir d'un corpus de ~700 documents (PDF/DOCX/PPTX/XLSX).

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Ouvrir `config.py` et modifier `DOSSIER_DOCUMENTS` pour pointer vers ton
dossier de documents.

## Pipeline, étape par étape

```
┌──────────┐     ┌───────────────┐     ┌────────┐     ┌─────┐
│   ETL    │ ──▶ │ ENTRAÎNEMENT  │ ──▶ │ TESTS  │ ──▶ │ APP │
└──────────┘     └───────────────┘     └────────┘     └─────┘
 extraction        embeddings          évaluation      Gradio
 + chunking        (index sémantique)  qualité
```

| Étape | Fichier | Commande | Ce que ça fait | Sortie (cache) |
|---|---|---|---|---|
| 1. ETL | `etl.py` | `python etl.py` | Extrait le texte de chaque fichier, découpe en passages | `data/01_extraction.pkl`, `data/02_passages.pkl` |
| 2. Entraînement | `entrainement.py` | `python entrainement.py` | Encode chaque passage en vecteur (embedding) | `data/03_embeddings.npz` |
| 3. Tests | `test_assistant.py` | `pytest test_assistant.py -v` (unitaires)<br>`python test_assistant.py` (évaluation qualité) | Vérifie la logique + mesure si le moteur retrouve les bons documents | `data/04_rapport_tests.json` |
| 4. App | `app.py` | `python app.py` | Charge le LLM, lance l'interface Gradio (lien public temporaire) | — |

Ou tout en une fois :

```bash
python main.py              # pipeline complet
python main.py --sans-app   # s'arrête après les tests (utile en CI/CD)
```

## Pourquoi séparer en étapes avec du cache ?

Avec ~700 documents, ré-extraire et ré-encoder tout à chaque essai serait
beaucoup trop lent. Chaque étape lit le résultat mis en cache par l'étape
précédente et ne recalcule que ce qui a changé (fichier nouveau/modifié,
corpus modifié). Tu peux donc relancer `python app.py` autant de fois que tu
veux sans jamais retraverser 700 PDF.

## Tests

- **Tests unitaires** (`pytest test_assistant.py -v`) : vérifient le
  chunking, la détection de catégorie, la gestion des fichiers corrompus —
  sans toucher aux vrais documents, donc rapides et reproductibles partout.
- **Évaluation qualité** (`python test_assistant.py`) : vérifie, sur une
  liste de questions dont on connaît la catégorie attendue
  (`QUESTIONS_EVALUATION` dans `test_assistant.py`), si le moteur de
  recherche retrouve bien la bonne source. À compléter avec tes propres
  questions/réponses connues au fur et à mesure que tu explores le corpus.

## "Entraînement" : à ne pas confondre avec du fine-tuning

Aucun modèle de langage n'est ré-entraîné (fine-tuné) ici. Ce que
`entrainement.py` "entraîne", c'est le moteur de RECHERCHE (les embeddings),
pas le modèle de RÉDACTION. C'est un choix : le RAG permet de citer les
sources précisément et de mettre à jour le corpus sans jamais retoucher au
LLM — voir la note dans `entrainement.py` pour le détail du raisonnement.

## Limite connue : qualité du LLM local

Le modèle local (Qwen 0.5B-1.5B) est suffisant pour un corpus restreint,
mais peut avoir du mal à bien synthétiser sur un corpus aussi large et
multi-thématique. Une option utilisant l'API Claude est fournie en
commentaire dans `app.py` (`demander_au_modele_claude`).

## Scans sans OCR

Beaucoup de documents anciens sont des PDF scannés : l'ETL les détecte
(texte extrait < 50 caractères) et les signale, mais ne fait pas d'OCR
automatiquement. À ajouter si besoin (Tesseract / pytesseract).
