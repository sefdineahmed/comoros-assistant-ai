# -*- coding: utf-8 -*-
"""
compresser_embeddings.py — À lancer une fois, en local.

Réduit la taille de data/03_embeddings.npz d'environ moitié en stockant les
vecteurs en float16 au lieu de float32. Aucune perte notable de qualité pour
la recherche sémantique (la précision float16 est largement suffisante pour
comparer des similarités cosinus). Utile pour faciliter l'upload sur
Hugging Face si ta connexion est lente/instable.

USAGE :
    python compresser_embeddings.py
"""

import os

import numpy as np

import config


def compresser():
    if not os.path.exists(config.CACHE_EMBEDDINGS):
        raise FileNotFoundError(
            f"'{config.CACHE_EMBEDDINGS}' introuvable. Lance d'abord entrainement.py."
        )

    taille_avant = os.path.getsize(config.CACHE_EMBEDDINGS) / 1e6
    print(f"📦 Fichier actuel : {taille_avant:.1f} Mo")

    cache = np.load(config.CACHE_EMBEDDINGS, allow_pickle=True)
    vecteurs = cache["vecteurs"]
    empreinte = cache["empreinte"]

    if vecteurs.dtype == np.float16:
        print("✅ Déjà en float16, rien à faire.")
        return

    vecteurs_compresses = vecteurs.astype(np.float16)
    np.savez(config.CACHE_EMBEDDINGS, vecteurs=vecteurs_compresses, empreinte=empreinte)

    taille_apres = os.path.getsize(config.CACHE_EMBEDDINGS) / 1e6
    print(f"✅ Fichier compressé : {taille_apres:.1f} Mo "
          f"(gain de {100 * (1 - taille_apres / taille_avant):.0f}%)")


if __name__ == "__main__":
    compresser()
