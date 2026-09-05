# -*- coding: utf-8 -*-
"""
entrainement.py — Jalon 2 de l'atelier, version production.

On ne "fine-tune" pas de modèle ici (voir note en bas). On "entraîne" au
sens où l'atelier l'entend : on construit l'INDEX SÉMANTIQUE, c'est-à-dire
qu'on transforme chaque passage en vecteur (embedding), une seule fois,
pour pouvoir ensuite chercher instantanément les passages les plus proches
de n'importe quelle question.

Exécuter directement ce fichier lance l'entraînement complet :
    python entrainement.py
(nécessite d'avoir lancé etl.py avant)
"""

import hashlib

import numpy as np

import config
import etl


def _empreinte_corpus(textes):
    """Empreinte du corpus entier : si elle ne change pas d'une exécution à
    l'autre, on sait qu'on peut réutiliser l'index déjà calculé.
    encode(errors='ignore') : protection défensive contre d'éventuels
    caractères Unicode invalides qui auraient échappé au nettoyage de l'ETL."""
    return hashlib.md5("".join(textes).encode("utf-8", errors="ignore")).hexdigest()


def entrainer(DOCUMENTS=None, forcer=False):
    """Construit (ou recharge depuis le cache) les embeddings de tous les
    passages. Renvoie (vecteurs, encodeur)."""
    from sentence_transformers import SentenceTransformer

    DOCUMENTS = DOCUMENTS or etl.charger_passages()

    # On inclut catégorie + titre dans le texte encodé : le moteur sait ainsi
    # que "espèces de tortues" vient de Biodiversité et pas de Pêche.
    textes = [f"[{d['categorie']}] {d['titre']} - {d['texte']}" for d in DOCUMENTS]
    empreinte = _empreinte_corpus(textes)

    encodeur = SentenceTransformer(config.MODELE_EMBEDDING)

    if not forcer:
        import os
        if os.path.exists(config.CACHE_EMBEDDINGS):
            cache = np.load(config.CACHE_EMBEDDINGS, allow_pickle=True)
            if str(cache["empreinte"]) == empreinte:
                print("📦 Embeddings chargés depuis le cache (corpus inchangé, pas de recalcul).")
                # .astype(float32) : fonctionne que le cache soit en float16 ou float32
                return cache["vecteurs"].astype(np.float32), encodeur

    print(f"🧠 Entraînement du moteur sémantique sur {len(textes)} passages "
          f"(modèle : {config.MODELE_EMBEDDING})...")
    vecteurs = encodeur.encode(
        textes,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=config.BATCH_SIZE_EMBEDDING,
    )
    # Sauvegarde en float16 : fichier ~2x plus léger, sans perte notable pour
    # la recherche sémantique (utile pour l'upload vers Hugging Face).
    np.savez(config.CACHE_EMBEDDINGS, vecteurs=vecteurs.astype(np.float16), empreinte=empreinte)
    print(f"✅ Entraînement terminé : {len(vecteurs)} vecteurs de dimension "
          f"{vecteurs.shape[1]}, sauvegardés dans {config.CACHE_EMBEDDINGS}")
    return vecteurs, encodeur


def charger_index():
    """Utilisé par app.py et test_assistant.py : recharge l'index déjà
    entraîné sans jamais recalculer si rien n'a changé."""
    DOCUMENTS = etl.charger_passages()
    vecteurs, encodeur = entrainer(DOCUMENTS)
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


# -----------------------------------------------------------------------------
# NOTE — pourquoi on ne fine-tune pas de LLM ici
# -----------------------------------------------------------------------------
# "Entraîner" un modèle de langage (au sens fine-tuning, ajuster ses poids)
# demanderait des milliers d'exemples de questions/réponses annotées sur les
# Comores, une carte GPU sérieuse, et du temps — hors de portée d'un projet
# comme celui-ci, et rarement nécessaire : le RAG (Retrieval-Augmented
# Generation) donne de meilleurs résultats pour ce cas d'usage, car il permet
# de citer précisément la source et de mettre à jour le corpus sans jamais
# ré-entraîner quoi que ce soit. C'est ce que fait ce module : on "entraîne"
# uniquement le moteur de RECHERCHE (les embeddings), pas le modèle de
# RÉDACTION (le LLM), exactement comme dans l'atelier.
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    DOCUMENTS = etl.charger_passages()
    vecteurs, encodeur = entrainer(DOCUMENTS)

    # Petite démonstration en console
    for q in ["Quelle est l'origine du peuplement des Comores ?",
              "Quels sont les enjeux de la pêche artisanale ?"]:
        print(f"\n❓ {q}")
        for p in chercher(q, DOCUMENTS, vecteurs, encodeur, k=3):
            print(f"   [{p['score']:.2f}] [{p['categorie']}] {p['fichier']}")
