# -*- coding: utf-8 -*-
"""
config.py — Tous les réglages du projet au même endroit.
Modifie ce fichier, tout le reste s'adapte automatiquement.
"""

import os
import torch

# =============================================================================
# CHEMINS
# =============================================================================

# ⬇️ À adapter : chemin vers ton dossier de documents.
# Par défaut, on suppose que "comores_ia/" et "Quelques documents sur les
# Comores/" sont deux dossiers VOISINS (même dossier parent) :
#   Objectif-IA/
#   ├── Quelques documents sur les Comores/
#   └── comores_ia/            ← tu lances les scripts depuis ici
# D'où le "../" pour remonter d'un niveau. Si ton arborescence est
# différente, remplace directement par le chemin absolu, ex (Linux/Mac) :
#   DOSSIER_DOCUMENTS = "/home/sefdine/github/portfolio-bi-telecom/Objectif-IA/Quelques documents sur les Comores"
DOSSIER_DOCUMENTS = "../Quelques documents sur les Comores"

# Dossier où on stocke tous les fichiers intermédiaires (cache ETL, index,
# résultats de tests). Rien de tout ça n'est à modifier à la main : chaque
# script lit/écrit dedans automatiquement.
DOSSIER_DATA = "./data"
os.makedirs(DOSSIER_DATA, exist_ok=True)

CACHE_EXTRACTION = os.path.join(DOSSIER_DATA, "01_extraction.pkl")   # sortie de l'ETL (Extract+Transform)
CACHE_PASSAGES = os.path.join(DOSSIER_DATA, "02_passages.pkl")       # documents découpés en chunks
CACHE_EMBEDDINGS = os.path.join(DOSSIER_DATA, "03_embeddings.npz")   # sortie de l'entraînement
RAPPORT_TESTS = os.path.join(DOSSIER_DATA, "04_rapport_tests.json")  # sortie des tests

# =============================================================================
# EXTRACTION (ETL)
# =============================================================================

EXTENSIONS_SUPPORTEES = {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}
LONGUEUR_MIN_DOCUMENT = 50   # en dessous, on considère le doc vide/illisible (scan sans OCR)

# =============================================================================
# CHUNKING (Transform)
# =============================================================================

TAILLE_CHUNK = 900
CHEVAUCHEMENT = 150

# =============================================================================
# ENTRAÎNEMENT (moteur sémantique)
# =============================================================================

MODELE_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE_EMBEDDING = 64

# =============================================================================
# GÉNÉRATION (LLM)
# =============================================================================

if torch.cuda.is_available():
    MODELE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
    DEVICE = 0
else:
    MODELE_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
    DEVICE = -1

MAX_NEW_TOKENS = 400
K_PASSAGES_DEFAUT = 5

ROLE = (
    "Tu es un assistant documentaire expert sur les Comores (archipel : "
    "Grande Comore/Ngazidja, Anjouan/Ndzuwani, Mohéli/Mwali, et Mayotte). "
    "Tu couvres l'histoire, l'archéologie, l'anthropologie, la démographie, "
    "la diaspora, l'écologie, l'éducation, l'économie, le genre, la génétique, "
    "la linguistique, le patrimoine, la religion, la santé, la sociologie, "
    "le monde swahili, les projets de l'ONU et le tourisme aux Comores. "
    "Tu réponds de façon précise, nuancée et académique."
)

REGLE_CITATION = (
    " Tu réponds UNIQUEMENT à partir des documents fournis ci-dessous. "
    "Pour chaque affirmation, indique le document source entre parenthèses "
    "(nom du fichier). Si plusieurs documents se contredisent, signale-le. "
    "Si la réponse ne figure dans AUCUN document fourni, réponds exactement : "
    "« Je ne trouve pas cette information dans les documents disponibles. »"
)
