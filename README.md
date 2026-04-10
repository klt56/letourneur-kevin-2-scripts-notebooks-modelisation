# letourneur-kevin-2-scripts-notebooks-modelisation
Notebooks, model comparisons and MLflow tracking for the Air Paradis sentiment analysis project.


# Air Paradis – Scripts, notebooks et expérimentations

Ce dépôt contient la phase d’expérimentation du projet **Air Paradis** de classification de sentiment sur tweets.

## Contenu du dépôt
- notebooks de modélisation
- essais comparatifs entre plusieurs approches :
  - modèle simple (TF-IDF + régression logistique)
  - modèles avancés sur mesure
  - approche BERT
- suivi des expérimentations avec **MLflow**
- scripts utiles à la phase de modélisation

## Objectif
Comparer plusieurs approches de classification de sentiment, suivre les essais avec MLflow, puis identifier le modèle le plus pertinent avant la mise en ligne.

## Jeu de données
Les expérimentations sont basées sur le dataset **Sentiment140**.

**Remarque :**
le fichier CSV complet d’origine n’est pas inclus dans ce dépôt s’il dépasse la limite de taille GitHub.
Les notebooks peuvent néanmoins être consultés avec leurs sorties enregistrées.

## Résumé de la démarche
- préparation et échantillonnage des données
- baseline classique avec TF-IDF + régression logistique
- test de modèles avancés sur mesure
- comparaison de prétraitements et d’embeddings
- expérimentation avec BERT
- comparaison des résultats via MLflow

## Modèle retenu dans le projet global
Le modèle retenu pour la mise en ligne dans le projet complet est le **BiLSTM stemming** (`bilstm_stemming_v1`).

## Installation
Créer un environnement virtuel puis installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
