"""
Module Facade giữ khả năng tương thích ngược (Backward Compatibility).
Các class đã được phân tách dạng hướng đối tượng vào các package chuyên biệt:
- models: ArcMarginProduct, MyModel
- data: AnimeFaceCropper, OregairuDualDataset, eval_transform
- training: ModelTrainer
- inference: AnimePredictor
"""

from models import ArcMarginProduct, MyModel
from data import AnimeFaceCropper, eval_transform
from inference import AnimePredictor

__all__ = [
    "ArcMarginProduct",
    "MyModel",
    "AnimeFaceCropper",
    "eval_transform",
    "AnimePredictor",
]
