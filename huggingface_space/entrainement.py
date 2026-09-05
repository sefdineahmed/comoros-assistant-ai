# -*- coding: utf-8 -*-
"""
entrainement.py — Version Space.

Contrairement à la version développement (dépôt principal), ce module ne
sait PAS reconstruire l'index à partir des documents bruts : il se contente
de charger l'index déjà calculé (passages + embeddings), téléchargé par
hf_utils.py. Toute la logique de recherche sémantique reste identique.
"""

import pickle

import numpy as np

import config


def charger_index():
    """Charge les passages et les embeddings déjà téléchargés en local."""
    with open(config.CACHE_PASSAGES, "rb") as f:
        DOCUMENTS = pickle.load(f)

    cache = np.load(config.CACHE_EMBEDDINGS, allow_pickle=True)
    vecteurs = cache["vecteurs"]

    from sentence_transformers import SentenceTransformer
    encodeur = SentenceTransformer(config.MODELE_EMBEDDING)

    print(f"✅ Index chargé : {len(DOCUMENTS)} passages, "
          f"vecteurs de dimension {vecteurs.shape[1]}.")
    return DOCUMENTS, vecteurs, encodeur


def chercher(question, DOCUMENTS, vecteurs, encodeur, k=None, categorie_filtre=None):
    """Renvoie les k passages les plus proches sémantiquement de la question."""
    k = k or config.K_PASSAGES_DEFAUT
    v_question = encodeur.encode(question, normalize_embeddings=True)
    similarites = vecteurs @ v_question

    if categorie_filtre:
        masque = np.array([d["categorie"] == categorie_filtre for d in DOCUMENTS])
        similarites = np.where(masque, similarites, -1)

    indices = np.argsort(-similarites)[:k]
    return [
        {**DOCUMENTS[i], "score": float(similarites[i])}
        for i in indices if similarites[i] > -1
    ]
