"""
Module d'évaluation du projet STA211.
"""

from .metrics import (
    calculate_basic_metrics, calculate_detailed_metrics,
    optimize_threshold, analyze_threshold_sensitivity,
    plot_evaluation_dashboard, compare_models_visualization,
    generate_evaluation_report, export_metrics_to_csv
)

__all__ = [
    'calculate_basic_metrics', 'calculate_detailed_metrics',
    'optimize_threshold', 'analyze_threshold_sensitivity',
    'plot_evaluation_dashboard', 'compare_models_visualization',
    'generate_evaluation_report', 'export_metrics_to_csv'
]