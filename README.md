NBA Win Predictor
Description
Ce projet a pour objectif de prédire le vainqueur d’un match NBA en se basant sur les performances récentes et les statistiques contextuelles des équipes.
Le projet inclut :
-Préparation et nettoyage d’un dataset Kaggle.
-Création de features pour chaque match : rolling stats sur les derniers matchs, ELO score dynamique, jours de repos, série de victoires, etc.
-Modélisation avec plusieurs algorithmes : Logistic Regression, Random Forest, Gradient Boosting.
-Analyse de l’importance des variables pour comprendre ce qui influence le résultat d’un match.
-Simulation de saison complète pour comparer les prédictions aux classements réels.
-Application Streamlit pour tester des matchups et visualiser les probabilités de victoire.

Voici sa structure
nba-win-predictor/

 data/
  raw/          qui sont les données brutes Kaggle
  test_data.py   qui est la préparation et transformation des données
   processed/    qui sont les features prêtes pour le modèle

reports/          qui sont les résultats EDA, métriques, coefficients, simulations
 eda.py           qui est l'analyse exploratoire
interpretability.py   qui modélise l'importance des features et leur impact
 model.py             Entraînement et évaluation des modèles
season_simulation.py    Simulation d’une saison complète
 streamlitapp.py         Application streamlit

 README.md

 Fonctionnalités principales:
-Prédiction probabiliste de la victoire pour l’équipe à domicile.
-Visualisation des variables qui influencent le plus la prédiction.
-Simulation Monte Carlo d’une saison complète pour évaluer la précision des modèles sur plusieurs matchs.
-Application interactive pour tester des matchups et afficher les statistiques récentes des équipes.
