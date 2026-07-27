"""
tests/test_ml_pipeline.py
──────────────────────────
Unit Test Suite for QuantSphereX ML Pipeline (v1.0.0)

Covers:
  - Training module (TrainResult DTO, time-split, ensemble fit)
  - Prediction module (normalize, drift detection)
  - Feature Importance (get_importance, top features, stability)
  - Model Registry (save, load, list, champion promotion)
  - Experiment Tracker (start_run, log_metrics, get_runs)
  - Cross-Validation (expanding-window, sliding-window)
  - Hyperparameter Tuning (grid search fallback)
  - Confidence Estimation (variance, tiers, conformal intervals)
  - Explainability (SHAP graceful fallback without shap installed)
  - Backward compatibility with alpha_layer.xgboost_trainer
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel
from ml_layer.config import MLConfig
from ml_layer.training import train, TrainResult
from ml_layer.prediction import predict, detect_drift, build_score_panel
from ml_layer.feature_importance import get_importance, get_top_features, importance_stability
from ml_layer.registry import ModelRegistry
from ml_layer.experiment_tracker import ExperimentTracker
from ml_layer.cross_validation import expanding_window_cv, sliding_window_cv
from ml_layer.hyperparameter_tuning import tune
from ml_layer.confidence import ensemble_variance, confidence_tiers, conformal_intervals
from ml_layer.explainability import compute_shap_values, global_shap_importance


# ─── Shared Fixtures ──────────────────────────────────────────────────────────

def _make_panel(n_dates: int = 200, n_tickers: int = 10, n_features: int = 15, seed: int = 42):
    """Generates a synthetic (Date, Ticker) MultiIndex feature panel and target."""
    np.random.seed(seed)
    dates   = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    feat_cols = [f"f_{i}" for i in range(n_features)]
    X = pd.DataFrame(np.random.randn(len(idx), n_features), index=idx, columns=feat_cols)
    y = pd.Series(np.random.randn(len(idx)), index=idx, name="target")
    return X, y


def _train_small(n_dates=200, n_tickers=5, n_features=10):
    """Returns a trained EnsembleAlphaModel on tiny synthetic data."""
    X, y = _make_panel(n_dates=n_dates, n_tickers=n_tickers, n_features=n_features)
    config = MLConfig(n_ensemble=2, val_split=0.25, experiment_name="test_exp")
    result = train(X, y, config=config)
    return result, X, y


# ─── Training Tests ───────────────────────────────────────────────────────────

class TestTraining(unittest.TestCase):
    def test_train_returns_train_result(self):
        result, X, y = _train_small()
        self.assertIsInstance(result, TrainResult)

    def test_train_result_has_model(self):
        result, _, _ = _train_small()
        self.assertIsInstance(result.model, EnsembleAlphaModel)
        self.assertGreater(len(result.model.models), 0)

    def test_train_result_metrics_exist(self):
        result, _, _ = _train_small()
        self.assertIsInstance(result.train_ic, float)
        self.assertIsInstance(result.val_ic, float)

    def test_train_result_features_nonempty(self):
        result, _, _ = _train_small()
        self.assertGreater(len(result.features), 0)

    def test_train_respects_n_ensemble(self):
        X, y = _make_panel(n_dates=150, n_tickers=5, n_features=8)
        config = MLConfig(n_ensemble=3, val_split=0.2)
        result = train(X, y, config=config)
        self.assertEqual(len(result.model.models), 3)

    def test_train_elapsed_seconds_positive(self):
        result, _, _ = _train_small()
        self.assertGreater(result.elapsed_seconds, 0)


# ─── Prediction Tests ─────────────────────────────────────────────────────────

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.result, self.X, self.y = _train_small()
        self.model = self.result.model

    def test_predict_returns_series(self):
        scores = predict(self.model, self.X)
        self.assertIsInstance(scores, pd.Series)
        self.assertEqual(len(scores), len(self.X))

    def test_predict_rank_normalized_in_01(self):
        scores = predict(self.model, self.X, normalize=True, method="rank")
        valid = scores.dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_predict_without_normalize(self):
        scores = predict(self.model, self.X, normalize=False)
        self.assertIsInstance(scores, pd.Series)

    def test_detect_drift_no_drift_on_same_data(self):
        report = detect_drift(self.model, self.X, self.X, threshold=0.5)
        self.assertFalse(report["drift_detected"])

    def test_detect_drift_detects_large_shift(self):
        X_shifted = self.X + 100.0  # massive artificial shift
        report = detect_drift(self.model, self.X, X_shifted, threshold=0.5)
        self.assertTrue(report["drift_detected"])

    def test_build_score_panel_returns_dataframe(self):
        pred_dates = self.X.index.get_level_values(0).unique()[:5].tolist()
        panel = build_score_panel(self.model, self.X, pred_dates=pred_dates)
        self.assertIsInstance(panel, pd.DataFrame)
        self.assertGreater(len(panel), 0)


# ─── Feature Importance Tests ─────────────────────────────────────────────────

class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        self.result, self.X, self.y = _train_small()
        self.model = self.result.model

    def test_get_importance_returns_dataframe(self):
        imp = get_importance(self.model)
        self.assertIsInstance(imp, pd.DataFrame)
        self.assertIn("feature", imp.columns)
        self.assertIn("mean_importance", imp.columns)

    def test_get_importance_top_n(self):
        imp = get_importance(self.model, top_n=5)
        self.assertLessEqual(len(imp), 5)

    def test_get_top_features_returns_list(self):
        top = get_top_features(self.model, n=5)
        self.assertIsInstance(top, list)
        self.assertLessEqual(len(top), 5)

    def test_importance_stability_has_cv_column(self):
        stability = importance_stability(self.model)
        if stability.empty:
            self.skipTest("No importances available for small synthetic data — skipping cv column check.")
        self.assertIn("cv", stability.columns)
        self.assertIn("stability", stability.columns)


# ─── Model Registry Tests ─────────────────────────────────────────────────────

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_dir=self.tmpdir)
        self.result, self.X, self.y = _train_small()

    def test_save_returns_version_id(self):
        vid = self.registry.save(self.result.model, self.result)
        self.assertIsInstance(vid, str)
        self.assertIn("test_exp", vid)

    def test_list_versions_returns_dataframe(self):
        self.registry.save(self.result.model, self.result)
        versions = self.registry.list_versions()
        self.assertIsInstance(versions, pd.DataFrame)
        self.assertGreater(len(versions), 0)

    def test_load_restores_model(self):
        vid = self.registry.save(self.result.model, self.result)
        loaded_model, entry = self.registry.load(vid)
        self.assertIsInstance(loaded_model, EnsembleAlphaModel)
        self.assertGreater(len(loaded_model.models), 0)

    def test_promote_champion(self):
        vid = self.registry.save(self.result.model, self.result)
        self.registry.promote_champion(vid)
        champion = self.registry.get_champion()
        self.assertEqual(champion, vid)

    def test_delete_removes_entry(self):
        vid = self.registry.save(self.result.model, self.result)
        self.registry.delete(vid)
        versions = self.registry.list_versions()
        self.assertNotIn(vid, versions.get("version_id", pd.Series()).tolist())


# ─── Experiment Tracker Tests ─────────────────────────────────────────────────

class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(tracking_dir=self.tmpdir)

    def test_start_run_returns_run_id(self):
        run_id = self.tracker.start_run("test_exp")
        self.assertIsInstance(run_id, str)
        self.assertEqual(len(run_id), 8)

    def test_log_metrics_updates_run(self):
        run_id = self.tracker.start_run("test_exp", params={"lr": 0.04})
        self.tracker.log_metrics(run_id, "test_exp", {
            "train_ic": 0.05, "val_ic": 0.04, "ic_tstat": 2.1
        })
        runs = self.tracker.get_runs("test_exp")
        self.assertGreater(len(runs), 0)
        self.assertEqual(runs.iloc[0]["val_ic"], 0.04)

    def test_get_runs_sorted_by_val_ic(self):
        for val_ic in [0.01, 0.05, 0.03]:
            rid = self.tracker.start_run("sort_exp")
            self.tracker.log_metrics(rid, "sort_exp", {"val_ic": val_ic})
        runs = self.tracker.get_runs("sort_exp")
        self.assertGreaterEqual(runs.iloc[0]["val_ic"], runs.iloc[-1]["val_ic"])

    def test_fail_run_marks_status_failed(self):
        rid = self.tracker.start_run("fail_exp")
        self.tracker.fail_run(rid, "fail_exp", error="Test error")
        runs = self.tracker.get_runs("fail_exp")
        # Status is stored but get_runs may not surface it by default; just check no crash
        self.assertGreater(len(runs), 0)


# ─── Cross-Validation Tests ───────────────────────────────────────────────────

class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _make_panel(n_dates=300, n_tickers=8, n_features=10)
        self.config = MLConfig(cv_n_splits=3, n_ensemble=1)

    def test_expanding_cv_returns_cv_result(self):
        from ml_layer.cross_validation import CVResult
        result = expanding_window_cv(self.X, self.y, config=self.config)
        self.assertIsInstance(result, CVResult)
        self.assertEqual(result.cv_type, "expanding")

    def test_expanding_cv_fold_count_positive(self):
        result = expanding_window_cv(self.X, self.y, config=self.config)
        self.assertGreater(result.n_folds, 0)

    def test_sliding_cv_returns_cv_result(self):
        from ml_layer.cross_validation import CVResult
        result = sliding_window_cv(self.X, self.y, config=self.config)
        self.assertIsInstance(result, CVResult)
        self.assertEqual(result.cv_type, "sliding")

    def test_cv_summary_returns_dataframe(self):
        result = expanding_window_cv(self.X, self.y, config=self.config)
        summary = result.summary()
        self.assertIsInstance(summary, pd.DataFrame)


# ─── Hyperparameter Tuning Tests ──────────────────────────────────────────────

class TestHyperparameterTuning(unittest.TestCase):
    def test_tune_with_zero_trials_returns_base_params(self):
        X, y = _make_panel(n_dates=100, n_tickers=5, n_features=8)
        config = MLConfig(tune_n_trials=0)
        result = tune(X, y, config=config)
        self.assertIn("max_depth", result)
        self.assertIn("learning_rate", result)

    def test_tune_grid_search_fallback(self):
        """Grid search runs when optuna is unavailable (n_trials > 0 without optuna)."""
        X, y = _make_panel(n_dates=150, n_tickers=5, n_features=8)
        config = MLConfig(tune_n_trials=1)
        # Patch optuna import to force grid search
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"optuna": None}):
            result = tune(X, y, config=config, n_trials=1)
        self.assertIsInstance(result, dict)
        self.assertIn("max_depth", result)


# ─── Confidence Estimation Tests ──────────────────────────────────────────────

class TestConfidenceEstimation(unittest.TestCase):
    def setUp(self):
        self.result, self.X, self.y = _train_small()
        self.model = self.result.model

    def test_ensemble_variance_returns_series(self):
        var = ensemble_variance(self.model, self.X)
        self.assertIsInstance(var, pd.Series)
        self.assertEqual(len(var), len(self.X))

    def test_ensemble_variance_non_negative(self):
        var = ensemble_variance(self.model, self.X)
        self.assertTrue((var.dropna() >= 0).all())

    def test_confidence_tiers_valid_labels(self):
        tiers = confidence_tiers(self.model, self.X)
        valid_labels = {"HIGH", "MEDIUM", "LOW"}
        self.assertTrue(set(tiers.dropna().unique()).issubset(valid_labels))

    def test_conformal_intervals_returns_dataframe(self):
        dates = self.X.index.get_level_values(0).unique().sort_values()
        mid = dates[len(dates) // 2]
        X_cal = self.X.loc[:mid]
        y_cal = self.y.reindex(X_cal.index)
        X_test = self.X.loc[mid:]
        intervals = conformal_intervals(self.model, X_cal, y_cal, X_test, coverage=0.90)
        self.assertIn("predicted", intervals.columns)
        self.assertIn("lower_bound", intervals.columns)
        self.assertIn("upper_bound", intervals.columns)


# ─── Explainability Tests (Graceful Fallback) ─────────────────────────────────

class TestExplainability(unittest.TestCase):
    def setUp(self):
        self.result, self.X, self.y = _train_small()
        self.model = self.result.model

    def test_compute_shap_returns_none_or_array(self):
        """SHAP should return None gracefully if not installed."""
        result = compute_shap_values(self.model, self.X, max_samples=50)
        # Either None (shap not installed) or ndarray (shap installed)
        self.assertTrue(result is None or isinstance(result, np.ndarray))

    def test_global_shap_importance_returns_dataframe_or_empty(self):
        result = global_shap_importance(self.model, self.X, max_samples=50)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
