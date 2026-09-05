"""Training: dataset construction, walk-forward validation, trainer, acceptance."""

from .dataset import TrainingDataset, TrainingDatasetBuilder
from .walkforward import Fold, walk_forward_folds
from .validation import ValidationMetrics, simulate_validation, ModelValidator, AcceptanceResult
from .trainer import ModelTrainer, TrainingReport

__all__ = ["TrainingDataset", "TrainingDatasetBuilder", "Fold", "walk_forward_folds", "ValidationMetrics",
           "simulate_validation", "ModelValidator", "AcceptanceResult", "ModelTrainer", "TrainingReport"]
