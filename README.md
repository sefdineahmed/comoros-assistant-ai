# 🇰🇲 comoros-assistant-ai

**Assistant documentaire intelligent sur les Comores**, basé sur une architecture RAG (Retrieval-Augmented Generation) : il répond à des questions sur l'archipel des Comores (histoire, anthropologie, santé, démographie, écologie, religion, linguistique, tourisme, etc.) en citant précisément ses sources, à partir d'un corpus de plus de 700 documents académiques, administratifs et institutionnels.

> ⚠️ Projet réalisé dans le cadre de l'atelier **Objectif-IA**. Ceci n'est pas un produit fini : c'est un pipeline pédagogique, documenté étape par étape, pensé pour être compris et modifié facilement.

## Sommaire

- [À propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du pipeline](#architecture-du-pipeline)
- [Structure du dépôt](#structure-du-dépôt)
- [Corpus documentaire](#corpus-documentaire)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Tests](#tests)
- [Limites connues](#limites-connues)
- [Pistes d'amélioration](#pistes-damélioration)
- [Licence](#licence)


## À propos

Ce projet transforme un fonds documentaire hétérogène (PDF, Word, Excel, PowerPoint) portant sur les Comores en un assistant conversationnel capable de répondre à des questions précises, **en citant le document source pour chaque affirmation** — plutôt que de laisser un modèle de langage halluciner des réponses génériques.

Le corpus couvre notamment :

| Domaine | Exemples de thématiques |
|---|---|
| Histoire & Archéologie | Peuplement de l'archipel, sultanats, colonisation, esclavage |
| Anthropologie & Sociologie | Système foncier, matrilocalité, polygamie, valeurs socioculturelles |
| Démographie & Diaspora | Recensements, migrations vers la France/Madagascar/Zanzibar |
| Écologie & Environnement | Volcanisme (Karthala), mangroves, changement climatique, biodiversité |
| Santé | Paludisme, système de santé, épidémiologie |
| Éducation | Programmes scolaires, enseignement coranique, université des Comores |
| Genre | Statut de la femme, polygamie, égalité hommes-femmes |
| Religion | Islam aux Comores, pratiques religieuses, réformisme |
| Linguistique & Littérature | Shikomori, poésie comorienne, sociolinguistique |
| Économie & Entreprise | Investissement, douane, secteur privé |
| Système des Nations Unies | Rapports UNICEF, PNUD, réponse aux catastrophes |
| Tourisme & Patrimoine | Écotourisme, urbanisme, patrimoine bâti |

## Fonctionnalités

- 📂 **Extraction multi-formats** : PDF, DOCX, PPTX, XLSX, parcourus récursivement dans l'arborescence de dossiers
- 🏷️ **Catégorisation automatique** : chaque passage garde la trace de son dossier d'origine (ex. `Santé`, `Religion`, `Pêche`) comme métadonnée
- ✂️ **Découpage intelligent** (chunking) avec chevauchement, qui coupe de préférence en fin de phrase
- 🧠 **Recherche sémantique multilingue** (français / anglais / arabe) via des embeddings `sentence-transformers`
- 📎 **Citations systématiques** : chaque réponse indique le(s) document(s) source(s)
- 🚫 **Anti-hallucination** : si l'information n'est dans aucun document, l'assistant le dit explicitement plutôt que d'inventer
- ⚡ **Mise en cache à chaque étape** : extraction, découpage et embeddings ne sont recalculés que si les fichiers sources changent
- 🧪 **Tests automatisés** : tests unitaires (pytest) + évaluation qualité du moteur de recherche
- 🌐 **Interface web** (Gradio) avec filtre par catégorie
- 🔍 **Filtrage par catégorie** pour affiner la recherche à un domaine précis

## Architecture du pipeline

Le projet suit une architecture en 4 étapes séquentielles, chacune persistant son résultat sur disque pour éviter tout recalcul inutile :

```
┌──────────┐     ┌───────────────┐     ┌────────┐     ┌─────┐
│   ETL    │ ──▶ │ ENTRAÎNEMENT  │ ──▶ │ TESTS  │ ──▶ │ APP │
└──────────┘     └───────────────┘     └────────┘     └─────┘
```

| Étape | Rôle | Entrée | Sortie (cache) |
|---|---|---|---|
| **1. ETL** | Extract (lecture des fichiers) → Transform (nettoyage + découpage) → Load (sauvegarde) | Corpus brut (PDF/DOCX/PPTX/XLSX) | `data/01_extraction.pkl`, `data/02_passages.pkl` |
| **2. Entraînement** | Encode chaque passage en vecteur sémantique (embedding) | Passages découpés | `data/03_embeddings.npz` |
| **3. Tests** | Vérifie la logique (unitaires) + mesure si le moteur retrouve les bonnes sources | Index entraîné | `data/04_rapport_tests.json` |
| **4. App** | Charge le LLM, répond aux questions avec citations, expose une interface web | Index + modèle de langage | — |

> ℹ️ Le mot « entraînement » désigne ici la construction de l'**index de recherche sémantique** (embeddings), pas un fine-tuning de modèle de langage. Voir le détail du raisonnement dans `entrainement.py`.

## Structure du dépôt

```
comoros-assistant-ai/
├── config.py              # Tous les réglages centralisés (chemins, tailles, modèles)
├── etl.py                  # Extract → Transform → Load
├── entrainement.py         # Construction de l'index sémantique (embeddings)
├── test_assistant.py       # Tests unitaires (pytest) + évaluation qualité
├── app.py                  # Génération de réponses + interface Gradio
├── main.py                 # Orchestrateur : lance tout le pipeline en une commande
├── requirements.txt        # Dépendances Python
├── README.md                # Ce fichier
├── .gitignore
└── data/                    # Généré automatiquement (caches)
    ├── 01_extraction.pkl
    ├── 02_passages.pkl
    ├── 03_embeddings.npz
    └── 04_rapport_tests.json
```

## Corpus documentaire

Le corpus (plus de 700 documents, plusieurs Go) **n'est pas versionné dans ce dépôt** — Git n'est pas adapté au stockage de fichiers binaires volumineux, et beaucoup de ces documents ont un statut de diffusion propre à respecter.

**Les documents sont disponibles ici :**
[Google Drive - Documents sur les Comores](https://drive.google.com/drive/folders/1RxAWS7RQasmIqolefXz2F_7_NEXsqCJt?usp=drive_link)

Pour utiliser ce projet :
1. Télécharger (ou synchroniser) le dossier Drive ci-dessus sur ta machine
2. Renseigner son chemin local dans `config.py` (voir [Configuration](#configuration))

L'arborescence attendue est celle du dossier Drive, avec une catégorie par sous-dossier de premier niveau (`Santé/`, `Religion/`, `Pêche/`, etc.) — c'est ce sous-dossier qui sert de métadonnée de citation.

## Prérequis

- Python 3.10 ou plus récent
- ~10 Go d'espace disque libre (corpus + caches + modèles)
- Une carte GPU est un plus pour l'étape de génération (sinon un modèle plus léger est utilisé automatiquement), mais n'est pas indispensable

## Installation

```bash
git clone https://github.com/sefdineahmed/comoros-assistant-ai.git
comoros-assistant-ai
pip install -r requirements.txt
```

## Configuration

Ouvrir `config.py` et adapter au minimum :

```python
# Chemin vers le dossier de documents téléchargé depuis le Drive
DOSSIER_DOCUMENTS = "/chemin/vers/Quelques documents sur les Comores"
```

Autres réglages disponibles dans ce fichier : taille des passages (`TAILLE_CHUNK`), modèle d'embedding (`MODELE_EMBEDDING`), modèle de génération (`MODELE_LLM`), nombre de passages récupérés par question (`K_PASSAGES_DEFAUT`).

## Utilisation

### Pipeline complet, en une commande

```bash
python main.py                # ETL → Entraînement → Tests → App (interface web)
python main.py --sans-app     # s'arrête après les tests (utile en CI/CD)
```

### Étape par étape

```bash
python etl.py             # 1. Extraction + découpage des documents
python entrainement.py    # 2. Construction de l'index sémantique
python test_assistant.py  # 3. Évaluation qualité du moteur de recherche
python app.py              # 4. Lance l'interface web (lien Gradio public temporaire)
```

Chaque script peut être relancé indépendamment : grâce au cache, seules les données modifiées sont retraitées.

## Tests

```bash
# Tests unitaires (chunking, extracteurs, détection de catégorie...)
pytest test_assistant.py -v

# Évaluation qualité du moteur de recherche (nécessite l'ETL + l'entraînement déjà faits)
python test_assistant.py
```

Les questions d'évaluation (`QUESTIONS_EVALUATION` dans `test_assistant.py`) vérifient que le moteur retrouve bien la bonne catégorie de document pour une question donnée. Cette liste est à enrichir progressivement.

## Limites connues

- **PDF scannés sans OCR** : une partie du corpus (rapports coloniaux, thèses anciennes, journaux officiels) est composée de scans images. L'extraction de texte échoue silencieusement pour ces fichiers (signalés en warning à l'exécution de l'ETL). Une intégration OCR (Tesseract) n'est pas encore implémentée.
- **Qualité du LLM local** : par défaut, le projet utilise un petit modèle local (Qwen 0.5B/1.5B) pour rester utilisable sans GPU ni clé API. Sur un corpus aussi large et multi-thématique, la qualité de synthèse reste limitée par rapport à un modèle plus puissant (voir l'option API Claude commentée dans `app.py`).
- **Temps d'entraînement** : le corpus complet produit plus de 100 000 passages à encoder ; cette étape peut prendre de 30 minutes à plusieurs heures sur CPU (elle n'est réalisée qu'une fois, grâce au cache).
- **Fichiers `.xls` anciens** : le format Excel 97-2003 (`.xls`) n'est pas géré par `openpyxl` ; ces fichiers sont ignorés (signalés en warning).

## Pistes d'amélioration

- [ ] Intégration OCR pour les PDF scannés (Tesseract / pytesseract)
- [ ] Bascule vers l'API Claude pour la génération (meilleure qualité de synthèse et de citation)
- [ ] Support du format `.xls` via `xlrd`
- [ ] Base vectorielle dédiée (FAISS, Qdrant, Chroma) pour une recherche plus rapide à grande échelle
- [ ] Interface de correction/validation des réponses par un expert du domaine
- [ ] Extraction et indexation des métadonnées bibliographiques (auteur, année, type de document)

## Licence

À définir selon les droits de diffusion des documents sources (voir le dossier Drive). Le code du pipeline (hors corpus documentaire) peut être publié sous licence de ton choix (ex. MIT).
