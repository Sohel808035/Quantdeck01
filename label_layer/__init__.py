"""
label_layer — QuantSphereX Label Factory Package.

Provides institutional-grade label generation for classification,
regression, and cross-sectional ranking across multiple forecast horizons.
"""

from label_layer.config import LabelConfig, STANDARD_HORIZONS
from label_layer.factory import (
    build_label_panel,
    build_ticker_labels,
    build_target_panel,
    get_label_registry,
    TARGET_COL,
    VERSION,
)
from label_layer.labels.custom import (
    CustomLabelBuilder,
    make_threshold_label,
    make_drawdown_label,
    make_composite_label,
)

__all__ = [
    "LabelConfig",
    "STANDARD_HORIZONS",
    "build_label_panel",
    "build_ticker_labels",
    "build_target_panel",
    "get_label_registry",
    "TARGET_COL",
    "VERSION",
    "CustomLabelBuilder",
    "make_threshold_label",
    "make_drawdown_label",
    "make_composite_label",
]
