# -*- coding: utf-8 -*-
"""
upload_dataset_to_hf.py — À LANCER UNE SEULE FOIS, CHEZ TOI (pas sur le Space).

Publie ton index déjà calculé (data/02_passages.pkl + data/03_embeddings.npz)
sur un Hugging Face Dataset. Le Space ira le télécharger tout seul au
démarrage, au lieu de refaire l'ETL + l'entraînement (impossible sur un
Space gratuit : pas de stockage persistant, pas de temps de build illimité).

PRÉREQUIS :
    pip install huggingface_hub
    huggingface-cli login          # (ou export HF_TOKEN=hf_xxx...)
    Avoir déjà lancé etl.py + entrainement.py au moins une fois en local.

USAGE :
    python upload_dataset_to_hf.py --repo-id TON_PSEUDO/comores-ia-index
"""

import argparse
import os

import config


def uploader(repo_id, prive=False):
    from huggingface_hub import HfApi

    for chemin in (config.CACHE_PASSAGES, config.CACHE_EMBEDDINGS):
        if not os.path.exists(chemin):
            raise FileNotFoundError(
                f"'{chemin}' introuvable. Lance d'abord :\n"
                "  python etl.py\n  python entrainement.py"
            )

    api = HfApi()
    print(f"📤 Création/mise à jour du dataset '{repo_id}' (privé={prive})...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=prive, exist_ok=True)

    for chemin in (config.CACHE_PASSAGES, config.CACHE_EMBEDDINGS):
        nom_fichier = os.path.basename(chemin)
        print(f"   ⬆️  {nom_fichier} ({os.path.getsize(chemin) / 1e6:.1f} Mo)...")
        api.upload_file(
            path_or_fileobj=chemin,
            path_in_repo=nom_fichier,
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"\n✅ Terminé. Dataset disponible sur : https://huggingface.co/datasets/{repo_id}")
    print(f"   ➡️  Renseigne ce repo_id dans huggingface_space/config.py "
          f"(DATASET_REPO_ID = \"{repo_id}\")")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True,
                         help="Ex: mon-pseudo/comores-ia-index")
    parser.add_argument("--prive", action="store_true",
                         help="Rend le dataset privé (nécessite un token HF avec accès dans le Space)")
    args = parser.parse_args()
    uploader(args.repo_id, prive=args.prive)
