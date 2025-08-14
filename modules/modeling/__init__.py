"""
Module de modélisation du projet STA211.
"""

from .ensembles import (
    create_voting_ensemble, create_bagging_ensemble, create_stacking_ensemble,
    optimize_ensemble, train_all_ensembles, compare_models_vs_ensembles,
    load_ensemble, get_ensemble_feature_importance
)

__all__ = [
    'create_voting_ensemble', 'create_bagging_ensemble', 'create_stacking_ensemble',
    'optimize_ensemble', 'train_all_ensembles', 'compare_models_vs_ensembles',
    'load_ensemble', 'get_ensemble_feature_importance'
]