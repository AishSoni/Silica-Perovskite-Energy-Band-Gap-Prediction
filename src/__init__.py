"""
Perovskite Bandgap Prediction Package

A complete machine learning pipeline for predicting bandgap energies
of double perovskite materials using Materials Project data.

Modules:
    data_io: Data loading and preparation
    featurize: Feature engineering (~300 descriptors)
    preprocess: Data preprocessing and splitting
    models: Model training and hyperparameter tuning
    eval: Evaluation metrics and visualization
    utils: Utility functions for reproducibility
"""

__version__ = "1.0.0"
__author__ = "Perovskite ML Project"

from . import utils
from . import data_io
from . import featurize
from . import preprocess
from . import models
from . import eval

__all__ = [
    'utils',
    'data_io',
    'featurize',
    'preprocess',
    'models',
    'eval'
]
