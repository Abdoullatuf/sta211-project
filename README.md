# STA211 - Projet de Classification

## Description
Projet de machine learning pour la classification de données publicitaires dans le cadre du cours STA211.

## Structure du Projet

```
sta211-project/
├── data/                      # Données du projet
│   ├── raw/                   # Données brutes
│   └── processed/             # Données prétraitées
├── modules/                   # Modules Python personnalisés
├── outputs/                   # Résultats et sorties
│   ├── figures/              # Graphiques et visualisations
│   ├── modeling/             # Modèles sauvegardés
│   ├── predictions/          # Prédictions finales
│   └── evaluation_test/      # Évaluations sur données test
├── models/                    # Modèles entraînés
├── config/                    # Fichiers de configuration
├── notebooks/                 # Notebooks Jupyter
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Modelisation_et_Optimisation.ipynb
│   └── 03_Stacking_et_predictions_finales.ipynb
└── requirements.txt           # Dépendances Python
```

## Installation

1. Cloner le projet :
```bash
git clone <repository-url>
cd sta211-project
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. **Exploration des données** : Ouvrir `01_EDA_Preprocessing.ipynb`
2. **Modélisation** : Exécuter `02_Modelisation_et_Optimisation.ipynb`
3. **Prédictions finales** : Utiliser `03_Stacking_et_predictions_finales.ipynb`

## Auteur
Maoulida Abdoullatuf