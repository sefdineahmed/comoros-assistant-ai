# -*- coding: utf-8 -*-
"""
test_assistant.py — Jalon 3 de l'atelier, version production.

Deux niveaux de tests :

1. TESTS UNITAIRES (pytest) — rapides, ne touchent pas aux vrais documents,
   vérifient que les briques de base (chunking, extracteurs) se comportent
   correctement, y compris sur des cas limites (texte vide, fichier
   corrompu...).
       pytest test_assistant.py -v

2. ÉVALUATION QUALITÉ (à lancer manuellement) — reproduit l'expérience de
   l'atelier : poser les mêmes questions AVEC et SANS documents, comparer,
   et vérifier que le moteur de recherche retrouve bien le bon document pour
   des questions dont on connaît la réponse à l'avance.
       python test_assistant.py
"""

import os
import json

import pytest

import config
import etl


# =============================================================================
# 1. TESTS UNITAIRES — logique pure, sans dépendre des vrais documents
# =============================================================================

def test_decouper_respecte_la_taille_approximative():
    texte = "Phrase un. " * 200  # texte long et répétitif
    passages = etl.decouper(texte, taille=100, chevauchement=20)
    assert len(passages) > 1
    # Chaque passage ne doit pas dépasser largement la taille demandée
    assert all(len(p) <= 130 for p in passages)


def test_decouper_texte_vide():
    assert etl.decouper("") == []


def test_decouper_texte_plus_court_que_la_taille():
    texte = "Un texte court."
    passages = etl.decouper(texte, taille=500)
    assert passages == [texte]


def test_decouper_coupe_en_fin_de_phrase_si_possible():
    texte = "Ceci est une phrase complète. " + "Suite du texte qui continue encore un peu. " * 5
    passages = etl.decouper(texte, taille=40, chevauchement=5)
    # Le premier passage doit se terminer par un point si un point était proche
    assert passages[0].rstrip().endswith(".")


def test_extraire_pdf_fichier_inexistant_ne_plante_pas():
    resultat = etl.extraire_pdf("/chemin/qui/nexiste/pas.pdf")
    assert resultat == ""


def test_extraire_docx_fichier_inexistant_ne_plante_pas():
    resultat = etl.extraire_docx("/chemin/qui/nexiste/pas.docx")
    assert resultat == ""


def test_lister_fichiers_ignore_les_extensions_non_supportees(tmp_path):
    (tmp_path / "Catégorie A").mkdir()
    (tmp_path / "Catégorie A" / "doc.pdf").write_text("contenu")
    (tmp_path / "Catégorie A" / "image.jpg").write_bytes(b"\xff\xd8\xff")  # pas supporté
    (tmp_path / "Catégorie A" / "notes.txt").write_text("pas supporté non plus")

    fichiers = etl.lister_fichiers(str(tmp_path))
    noms = {nom for _, _, nom in fichiers}
    assert noms == {"doc.pdf"}


def test_lister_fichiers_detecte_la_categorie_depuis_le_sous_dossier(tmp_path):
    (tmp_path / "Santé").mkdir()
    (tmp_path / "Santé" / "rapport.pdf").write_text("contenu")

    fichiers = etl.lister_fichiers(str(tmp_path))
    chemin, categorie, nom = fichiers[0]
    assert categorie == "Santé"


def test_transform_ignore_les_documents_trop_courts():
    docs_bruts = {
        "a.pdf": {"texte": "x", "categorie": "Test", "nom": "a.pdf", "chemin": "a.pdf"},  # trop court
        "b.pdf": {"texte": "Un texte suffisamment long pour être conservé par le pipeline. " * 3,
                  "categorie": "Test", "nom": "b.pdf", "chemin": "b.pdf"},
    }
    documents = etl.transform(docs_bruts)
    fichiers = {d["fichier"] for d in documents}
    assert "a.pdf" not in fichiers
    assert "b.pdf" in fichiers


def test_transform_conserve_les_metadonnees():
    docs_bruts = {
        "x.pdf": {"texte": "Un texte suffisamment long pour être conservé par le pipeline. " * 3,
                  "categorie": "Religion", "nom": "x.pdf", "chemin": "/a/b/x.pdf"},
    }
    documents = etl.transform(docs_bruts)
    assert documents[0]["categorie"] == "Religion"
    assert documents[0]["fichier"] == "x.pdf"
    assert documents[0]["chemin"] == "/a/b/x.pdf"


# =============================================================================
# 2. ÉVALUATION QUALITÉ — nécessite l'ETL + l'entraînement déjà faits
# =============================================================================

# ⬇️ À compléter avec de vraies paires question → document attendu, une fois
# que tu connais ton corpus. C'est ce qui te permet de mesurer objectivement
# si le moteur de recherche retrouve les bonnes sources (et pas seulement
# "ça a l'air de marcher").
QUESTIONS_EVALUATION = [
    {
        "question": "Quelle est l'origine du peuplement de la Grande Comore ?",
        "categorie_attendue": "Archéologie",
    },
    {
        "question": "Quels sont les principaux enjeux de la pêche artisanale aux Comores ?",
        "categorie_attendue": "Agriculture - Pêche",
    },
    {
        "question": "Comment s'organise le système éducatif aux Comores ?",
        "categorie_attendue": "Education",
    },
    {
        "question": "Quelle est la situation démographique des Comores ?",
        "categorie_attendue": "Démographie",
    },
    {
        "question": "Quel est le rôle de l'islam dans la société comorienne ?",
        "categorie_attendue": "Religion",
    },
]


def evaluer_recherche(DOCUMENTS, vecteurs, encodeur, k=5):
    """Pour chaque question de test, vérifie si au moins un des k passages
    retrouvés appartient bien à la catégorie attendue. Renvoie un rapport
    détaillé (taux de succès + détail par question)."""
    import entrainement

    resultats = []
    for cas in QUESTIONS_EVALUATION:
        passages = entrainement.chercher(cas["question"], DOCUMENTS, vecteurs, encodeur, k=k)
        categories_trouvees = [p["categorie"] for p in passages]
        succes = cas["categorie_attendue"] in categories_trouvees
        resultats.append({
            "question": cas["question"],
            "categorie_attendue": cas["categorie_attendue"],
            "categories_trouvees": categories_trouvees,
            "succes": succes,
            "meilleur_score": round(passages[0]["score"], 3) if passages else None,
        })

    taux_succes = sum(r["succes"] for r in resultats) / len(resultats)
    rapport = {"taux_succes": taux_succes, "details": resultats}

    with open(config.RAPPORT_TESTS, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Taux de succès du moteur de recherche : {taux_succes:.0%}")
    for r in resultats:
        icone = "✅" if r["succes"] else "❌"
        print(f"{icone} {r['question']}")
        print(f"    attendu : {r['categorie_attendue']} | trouvé : {r['categories_trouvees']}")
    print(f"\n📄 Rapport détaillé sauvegardé dans {config.RAPPORT_TESTS}")
    return rapport


def comparer_avec_sans_documents(question):
    """Reproduit l'expérience de l'atelier (Jalon 3) : la même question,
    posée AVEC puis SANS les documents, pour visualiser concrètement l'effet
    du RAG contre l'hallucination."""
    import entrainement
    import app  # réutilise repondre() et demander_au_modele_local()

    DOCUMENTS, vecteurs, encodeur = entrainement.charger_index()
    generateur = app.charger_modele_local()

    sans_docs = app.demander_au_modele_local(generateur, [
        {"role": "system", "content": config.ROLE},
        {"role": "user", "content": question},
    ])
    avec_docs, sources = app.repondre(question, DOCUMENTS, vecteurs, encodeur, generateur)

    print(f"\n❓ {question}")
    print(f"\n❌ SANS documents :\n   {' '.join(sans_docs.split())[:400]}")
    print(f"\n✅ AVEC documents :\n   {' '.join(avec_docs.split())}")
    print(f"   📎 Sources : {sources}")


if __name__ == "__main__":
    import entrainement

    if not os.path.exists(config.CACHE_PASSAGES):
        print("⚠️ Aucun index trouvé. Lance d'abord : python etl.py puis python entrainement.py")
    else:
        DOCUMENTS, vecteurs, encodeur = entrainement.charger_index()
        evaluer_recherche(DOCUMENTS, vecteurs, encodeur)
        comparer_avec_sans_documents("Quelle est l'origine du peuplement des Comores ?")
