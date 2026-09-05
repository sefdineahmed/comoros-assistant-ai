# -*- coding: utf-8 -*-
"""
hf_utils.py — Télécharge l'index précalculé (passages + embeddings) depuis
le Hugging Face Dataset au démarrage du Space, si besoin.

Le Space n'a pas de stockage persistant : à chaque redémarrage, ce module
retélécharge les fichiers (rapide : le cache de téléchargement Hugging Face
évite de retélécharger si rien n'a changé côté dataset).
"""

import os

import config


def assurer_donnees_locales():
    """Vérifie que les fichiers d'index existent en local ; sinon les
    télécharge depuis config.DATASET_REPO_ID."""
    if os.path.exists(config.CACHE_PASSAGES) and os.path.exists(config.CACHE_EMBEDDINGS):
        print("📦 Index déjà présent en local, pas de téléchargement.")
        return

    if not config.DATASET_REPO_ID or "TON_PSEUDO" in config.DATASET_REPO_ID:
        raise RuntimeError(
            "config.DATASET_REPO_ID n'est pas configuré. "
            "Renseigne-le après avoir lancé upload_dataset_to_hf.py."
        )

    from huggingface_hub import hf_hub_download

    print(f"⬇️  Téléchargement de l'index depuis le dataset '{config.DATASET_REPO_ID}'...")
    for nom_fichier, destination in [
        (config.NOM_FICHIER_PASSAGES, config.CACHE_PASSAGES),
        (config.NOM_FICHIER_EMBEDDINGS, config.CACHE_EMBEDDINGS),
    ]:
        chemin_telecharge = hf_hub_download(
            repo_id=config.DATASET_REPO_ID,
            repo_type="dataset",
            filename=nom_fichier,
        )
        # hf_hub_download renvoie un chemin dans le cache HF ; on s'assure
        # qu'il est aussi accessible à l'endroit attendu par le reste du code.
        if os.path.abspath(chemin_telecharge) != os.path.abspath(destination):
            import shutil
            shutil.copy(chemin_telecharge, destination)
        print(f"   ✅ {nom_fichier}")

    print("✅ Index prêt.")
