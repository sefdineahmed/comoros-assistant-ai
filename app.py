# -*- coding: utf-8 -*-
"""
app.py — Jalon 3 (génération) + Jalon 4 (mise en ligne) de l'atelier.

Charge l'index déjà entraîné (etl.py + entrainement.py doivent avoir tourné
au moins une fois), charge le LLM, et lance l'interface Gradio.

    python app.py
"""

import config
import entrainement


# =============================================================================
# GÉNÉRATION — le LLM qui rédige sous contrainte
# =============================================================================

def charger_modele_local():
    from transformers import pipeline
    import transformers
    transformers.logging.set_verbosity_error()

    print("Chargement du modèle local (1 à 3 minutes la première fois)...")
    generateur = pipeline("text-generation", model=config.MODELE_LLM, device=config.DEVICE)
    generateur.tokenizer.clean_up_tokenization_spaces = False
    generateur.model.generation_config.max_new_tokens = config.MAX_NEW_TOKENS
    generateur.model.generation_config.do_sample = False
    generateur.model.generation_config.temperature = None
    generateur.model.generation_config.top_p = None
    generateur.model.generation_config.top_k = None
    print("✅ Modèle local chargé !")
    return generateur


def demander_au_modele_local(generateur, messages):
    sortie = generateur(messages)
    return sortie[0]["generated_text"][-1]["content"]


# --- OPTION B (recommandée pour ce corpus) : appel à l'API Claude -----------
# import anthropic
# client = anthropic.Anthropic(api_key="TA_CLE_API")
#
# def demander_au_modele_claude(messages):
#     system = next(m["content"] for m in messages if m["role"] == "system")
#     user = next(m["content"] for m in messages if m["role"] == "user")
#     reponse = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=800,
#         system=system,
#         messages=[{"role": "user", "content": user}],
#     )
#     return reponse.content[0].text
# ------------------------------------------------------------------------------


def repondre(question, DOCUMENTS, vecteurs, encodeur, generateur, k=None, categorie_filtre=None):
    """L'assistant complet : recherche + rédaction sous contrainte + citations."""
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
# MISE EN LIGNE — interface Gradio
# =============================================================================

def lancer_interface(DOCUMENTS, vecteurs, encodeur, generateur):
    import gradio as gr

    categories = sorted(set(d["categorie"] for d in DOCUMENTS))

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
            gr.Dropdown(choices=["Toutes les catégories"] + categories,
                        value="Toutes les catégories", label="Filtrer par catégorie (optionnel)"),
        ],
        outputs=gr.Textbox(label="Réponse de l'assistant", lines=15),
        title="Assistant Documentaire — Comores",
        description=f"Répond à partir de {len(set(d['chemin'] for d in DOCUMENTS))} documents "
                     f"({len(DOCUMENTS)} passages indexés) couvrant l'histoire, la culture, "
                     f"la société et l'environnement des Comores.",
    )
    demo.launch(share=True)


if __name__ == "__main__":
    DOCUMENTS, vecteurs, encodeur = entrainement.charger_index()
    generateur = charger_modele_local()

    question_test = "Quelle est l'origine du peuplement des Comores ?"
    reponse, sources = repondre(question_test, DOCUMENTS, vecteurs, encodeur, generateur)
    print("\n❓", question_test)
    print("💬", reponse)
    print("📎 Sources :", sources)

    lancer_interface(DOCUMENTS, vecteurs, encodeur, generateur)
