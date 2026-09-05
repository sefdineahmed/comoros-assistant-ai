# -*- coding: utf-8 -*-
"""
main.py — Lance tout le pipeline dans l'ordre, comme l'atelier mais en un
seul appel :

    ETL  →  Entraînement  →  Tests  →  App (mise en ligne)

Usage :
    python main.py              # tout, jusqu'à l'ouverture de l'app
    python main.py --sans-app   # s'arrête après les tests (utile en CI)
"""

import argparse

import etl
import entrainement
import test_assistant
import app


def main(lancer_app=True):
    print("=" * 70)
    print("JALON 1 · ETL (Extract → Transform → Load)")
    print("=" * 70)
    DOCUMENTS = etl.run_etl()

    print("\n" + "=" * 70)
    print("JALON 2 · ENTRAÎNEMENT (construction de l'index sémantique)")
    print("=" * 70)
    vecteurs, encodeur = entrainement.entrainer(DOCUMENTS)

    print("\n" + "=" * 70)
    print("JALON 3 · TESTS (évaluation qualité de la recherche)")
    print("=" * 70)
    test_assistant.evaluer_recherche(DOCUMENTS, vecteurs, encodeur)

    if not lancer_app:
        print("\n✅ Pipeline terminé (app non lancée, --sans-app).")
        return

    print("\n" + "=" * 70)
    print("JALON 4 · MISE EN LIGNE")
    print("=" * 70)
    generateur = app.charger_modele_local()
    app.lancer_interface(DOCUMENTS, vecteurs, encodeur, generateur)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sans-app", action="store_true",
                         help="S'arrête après les tests, sans lancer l'interface web.")
    args = parser.parse_args()
    main(lancer_app=not args.sans_app)
