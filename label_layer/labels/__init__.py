"""
label_layer/labels — Label Module Registry.
"""

from label_layer.labels import regression, classification, ranking, horizons, custom

__all__ = [
    "regression",
    "classification",
    "ranking",
    "horizons",
    "custom",
]
