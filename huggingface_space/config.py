# -*- coding: utf-8 -*-
"""
config.py — Version Space : pas d'ETL ici, on télécharge un index déjà
calculé depuis un Hugging Face Dataset (voir hf_utils.py).
"""

import os
import torch

# =============================================================================
# ⬇️ À RENSEIGNER : le repo_id du dataset créé avec upload_dataset_to_hf.py
# =============================================================================
DATASET_REPO_ID = "TON_PSEUDO/comores-ia-index"  # <-- change ceci !
DATASET_PRIVE = False  # True si tu as uploadé le dataset avec --prive

# =============================================================================
# CHEMINS LOCAUX (au sein du Space, reconstruits au démarrage)
# =============================================================================
DOSSIER_DATA = "./data"
os.makedirs(DOSSIER_DATA, exist_ok=True)

NOM_FICHIER_PASSAGES = "02_passages.pkl"
NOM_FICHIER_EMBEDDINGS = "03_embeddings.npz"
CACHE_PASSAGES = os.path.join(DOSSIER_DATA, NOM_FICHIER_PASSAGES)
CACHE_EMBEDDINGS = os.path.join(DOSSIER_DATA, NOM_FICHIER_EMBEDDINGS)

# =============================================================================
# MOTEUR SÉMANTIQUE (doit être IDENTIQUE au modèle utilisé lors de
# l'entraînement local, sinon les vecteurs ne seront plus comparables)
# =============================================================================
MODELE_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =============================================================================
# GÉNÉRATION — Space gratuit = CPU uniquement, donc modèle léger
# =============================================================================
MODELE_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = -1  # CPU (les Spaces gratuits n'ont pas de GPU)

MAX_NEW_TOKENS = 350  # un peu réduit par rapport au local pour des réponses plus rapides
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
