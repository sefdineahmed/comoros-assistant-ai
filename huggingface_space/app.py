# -*- coding: utf-8 -*-
"""
app.py — Point d'entrée du Hugging Face Space.
Hugging Face détecte et exécute automatiquement ce fichier (voir app_file
dans l'en-tête YAML de README.md).
"""

import spaces  # DOIT être le tout premier import (exigence ZeroGPU)
import torch

import config
import hf_utils
import entrainement


# =============================================================================
# GÉNÉRATION
# =============================================================================

def charger_modele_local():
    from transformers import pipeline
    import transformers
    transformers.logging.set_verbosity_error()

    print("Chargement du modèle local...")
    # device=-1 : le modèle reste sur CPU tant qu'on n'est pas dans un appel
    # décoré @spaces.GPU (ZeroGPU n'attache un GPU que le temps de cet appel).
    generateur = pipeline("text-generation", model=config.MODELE_LLM, device=-1,
                           torch_dtype=torch.float16)
    generateur.tokenizer.clean_up_tokenization_spaces = False
    generateur.model.generation_config.max_new_tokens = config.MAX_NEW_TOKENS
    generateur.model.generation_config.do_sample = False
    generateur.model.generation_config.temperature = None
    generateur.model.generation_config.top_p = None
    generateur.model.generation_config.top_k = None
    print("✅ Modèle local chargé !")
    return generateur


@spaces.GPU(duration=60)
def demander_au_modele_local(generateur, messages):
    """Décoré @spaces.GPU : Hugging Face attache un GPU réel juste pour la
    durée de cet appel (jusqu'à 60s ici), puis le libère. On déplace donc le
    modèle sur le GPU au début, et on le rend au CPU à la fin pour ne pas
    garder de mémoire GPU réservée entre deux questions."""
    generateur.model.to("cuda")
    try:
        sortie = generateur(messages)
        return sortie[0]["generated_text"][-1]["content"]
    finally:
        generateur.model.to("cpu")


def repondre(question, DOCUMENTS, vecteurs, encodeur, generateur, k=None, categorie_filtre=None):
    passages = entrainement.chercher(question, DOCUMENTS, vecteurs, encodeur,
                                      k=k, categorie_filtre=categorie_filtre)

    if not passages:
        return "Je ne trouve pas cette information dans les documents disponibles.", []

    contexte = "\n\n".join(
        f"### [{p['categorie']}] {p['fichier']}\n{p['texte']}" for p in passages
    )

    messages = [
        {"role": "system", "content": config.ROLE + config.REGLE_CITATION},
        {"role": "user", "content": f"Documents :\n{contexte}\n\nQuestion : {question}"},
    ]

    reponse = demander_au_modele_local(generateur, messages)
    sources = [{"fichier": p["fichier"], "categorie": p["categorie"], "score": round(p["score"], 2)}
               for p in passages]
    return reponse, sources


# =============================================================================
# CHARGEMENT AU DÉMARRAGE (une seule fois, au lancement du Space)
# =============================================================================

print("=" * 60)
print("Démarrage de l'Assistant Documentaire — Comores")
print("=" * 60)

hf_utils.assurer_donnees_locales()
DOCUMENTS, vecteurs, encodeur = entrainement.charger_index()
generateur = charger_modele_local()

CATEGORIES = sorted(set(d["categorie"] for d in DOCUMENTS))
NB_DOCUMENTS_UNIQUES = len(set(d["chemin"] for d in DOCUMENTS))


# =============================================================================
# INTERFACE GRADIO
# =============================================================================

import gradio as gr


def assistant_web(question, categorie):
    if not question.strip():
        return "Posez une question sur les Comores !"
    filtre = None if categorie == "Toutes les catégories" else categorie
    reponse, sources = repondre(question, DOCUMENTS, vecteurs, encodeur, generateur,
                                 categorie_filtre=filtre)
    sources_txt = "\n".join(
        f"📎 {s['fichier']} ({s['categorie']}, score {s['score']})" for s in sources
    )
    return f"{reponse}\n\n---\n{sources_txt}"


demo = gr.Interface(
    fn=assistant_web,
    inputs=[
        gr.Textbox(label="Votre question", placeholder="Ex : Quelle est l'histoire du sultanat de Ngazidja ?"),
        gr.Dropdown(choices=["Toutes les catégories"] + CATEGORIES,
                    value="Toutes les catégories", label="Filtrer par catégorie (optionnel)"),
    ],
    outputs=gr.Textbox(label="Réponse de l'assistant", lines=15),
    title="🇰🇲 Assistant Documentaire — Comores",
    description=(
        f"Répond à partir de {NB_DOCUMENTS_UNIQUES} documents "
        f"({len(DOCUMENTS)} passages indexés) couvrant l'histoire, la culture, "
        f"la société et l'environnement des Comores.\n\n"
        "⚠️ Réponses générées par un petit modèle local (Qwen 0.5B, gratuit) : "
        "les réponses peuvent prendre 10-30 secondes et rester perfectibles. "
        "Chaque réponse cite ses sources."
    ),
    examples=[
        ["Quelle est l'origine du peuplement des Comores ?", "Toutes les catégories"],
        ["Quels sont les enjeux de la pêche artisanale ?", "Toutes les catégories"],
        ["Comment s'organise le système éducatif aux Comores ?", "Toutes les catégories"],
    ],
)

if __name__ == "__main__":
    demo.launch()
