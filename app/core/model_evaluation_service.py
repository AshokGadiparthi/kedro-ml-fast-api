"""
Model Evaluation Service
========================
Loads Kedro pipeline artifacts (Phase 3-5) and computes all evaluation data
needed by the frontend Model Evaluation Dashboard (all 5 tabs).

Data sources:
  Phase 2: feature_importance.csv, X_test_selected.csv, y_test.pkl, selected_features.pkl
  Phase 3: best_model.pkl, model_evaluation.pkl, phase3_predictions.csv, scalers.pkl
  Phase 4: phase4_report.csv, phase4_summary.json, trained models
  Phase 5: phase5_metrics.json (if available)

Computes at API time (not stored on disk):
  - ROC curve arrays (fpr, tpr, thresholds)
  - Precision-Recall curve arrays
  - Confusion matrix at any threshold
  - Business impact from confusion matrix
  - Production readiness criteria
  - Optimal threshold search
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import math

logger = logging.getLogger(__name__)

# Kedro project path (same pattern as jobs.py)
KEDRO_PROJECT_PATH = Path(os.getenv(
    'KEDRO_PROJECT_PATH',
    '/home/ashok/work/latest/full/kedro-ml-engine-integrated'
))

def _sanitize_for_json(obj):
    """Replace inf, -inf, nan with None so JSON serialization works."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj

def _resolve(relative_path: str) -> str:
    """Resolve a relative path against KEDRO_PROJECT_PATH."""
    return str(KEDRO_PROJECT_PATH / relative_path)


# ============================================================================
# ARTIFACT LOADERS
# ============================================================================

def _load_pickle(path: str):
    """Load a pickle file, return None on failure."""
    full = _resolve(path)
    try:
        with open(full, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load {full}: {e}")
        return None


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    """Load a CSV file, return None on failure."""
    full = _resolve(path)
    try:
        return pd.read_csv(full)
    except Exception as e:
        logger.warning(f"Could not load {full}: {e}")
        return None


def _load_json(path: str) -> Optional[dict]:
    """Load a JSON file, return None on failure."""
    full = _resolve(path)
    try:
        with open(full, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {full}: {e}")
        return None


def _load_text(path: str) -> Optional[str]:
    """Load a text file, return None on failure."""
    full = _resolve(path)
    try:
        with open(full, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Could not load {full}: {e}")
        return None


# ============================================================================
# MAIN SERVICE CLASS
# ============================================================================

class ModelEvaluationService:
    """
    Loads all Kedro pipeline outputs and computes the complete evaluation
    data structure needed by the frontend.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        logger.info(f"ModelEvaluationService initialized. KEDRO_PROJECT_PATH={KEDRO_PROJECT_PATH}")

    # ------------------------------------------------------------------
    # DATA LOADING  (lazy, cached)
    # ------------------------------------------------------------------

    def _get_problem_type(self) -> str:
        """Get the detected problem type from Phase 3."""
        if 'problem_type' not in self._cache:
            pt = _load_text("data/08_reporting/problem_type.txt")
            self._cache['problem_type'] = pt or 'classification'
        return self._cache['problem_type']

    def _get_all_trained_models(self) -> Optional[Dict[str, Any]]:
        """Load ALL trained models from Phase 4 (Dict[algo_name, model])."""
        if 'all_trained_models' not in self._cache:
            # Primary: all_trained_models.pkl has ALL 50+ models
            models = _load_pickle("data/06_models/phase4/all_trained_models.pkl")
            if models is None:
                # Fallback: best_models.pkl has top 5
                models = _load_pickle("data/06_models/phase4/best_models.pkl")
            self._cache['all_trained_models'] = models
            if models:
                logger.info(f"Loaded {len(models)} trained models: {list(models.keys())[:5]}...")
        return self._cache.get('all_trained_models')

    def _resolve_model_id(self, model_id: Optional[str]) -> Optional[str]:
        """
        Map a sanitized model_id (e.g. 'adaboostclassifier') back to
        the original dict key (e.g. 'AdaBoostClassifier').
        Returns None if model_id is 'best' or None (use default best model).
        """
        if not model_id or model_id in ('best', 'best_model'):
            return None  # signals: use data/06_models/best_model.pkl

        all_models = self._get_all_trained_models()
        if not all_models:
            return None

        # Direct match
        if model_id in all_models:
            return model_id

        # Sanitized match: compare lowercased, stripped versions
        sanitized_to_original = {}
        for key in all_models:
            sanitized = key.lower().replace(' ', '_').replace('(', '').replace(')', '')
            sanitized_to_original[sanitized] = key

        return sanitized_to_original.get(model_id.lower(), None)

    def _get_best_model(self, model_id: Optional[str] = None):
        """
        Load a specific trained model by ID, or the overall best model.
        - model_id=None or 'best' → data/06_models/best_model.pkl
        - model_id='adaboostclassifier' → looks up in all_trained_models.pkl
        """
        cache_key = f"model:{model_id or 'best'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        resolved_key = self._resolve_model_id(model_id)

        if resolved_key is not None:
            # Load specific model from Phase 4 dict
            all_models = self._get_all_trained_models()
            if all_models and resolved_key in all_models:
                model = all_models[resolved_key]
                logger.info(f"Loaded model '{resolved_key}' from all_trained_models")
                self._cache[cache_key] = model
                return model
            else:
                logger.warning(f"Model '{model_id}' (resolved: '{resolved_key}') not found in trained models")

        # Fallback: load the single best model
        if 'model:best' not in self._cache:
            self._cache['model:best'] = _load_pickle("data/06_models/best_model.pkl")
        self._cache[cache_key] = self._cache['model:best']
        return self._cache.get(cache_key)

    def _get_model_evaluation(self) -> Optional[dict]:
        """Load Phase 3 model_evaluation dict."""
        if 'model_evaluation' not in self._cache:
            self._cache['model_evaluation'] = _load_pickle("data/06_models/model_evaluation.pkl")
        return self._cache['model_evaluation']

    def _get_predictions_df(self) -> Optional[pd.DataFrame]:
        """Load Phase 3 predictions CSV (actual, predicted, correct)."""
        if 'predictions_df' not in self._cache:
            self._cache['predictions_df'] = _load_csv("data/07_model_output/phase3_predictions.csv")
        return self._cache['predictions_df']

    def _get_test_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load X_test and y_test used during training."""
        if 'X_test' not in self._cache:
            self._cache['X_test'] = _load_csv("data/03_primary/X_test_selected.csv")
        if 'y_test' not in self._cache:
            self._cache['y_test'] = _load_pickle("data/03_primary/y_test_raw.pkl")
        return self._cache.get('X_test'), self._cache.get('y_test')

    def _get_X_test_scaled(self) -> Optional[pd.DataFrame]:
        """Load scaled test data (what the model actually sees)."""
        if 'X_test_scaled' not in self._cache:
            self._cache['X_test_scaled'] = _load_csv("data/03_primary/X_test_scaled.csv")
            # Fallback to X_test_selected if scaled doesn't exist
            if self._cache['X_test_scaled'] is None:
                self._cache['X_test_scaled'] = _load_csv("data/03_primary/X_test_selected.csv")
        return self._cache.get('X_test_scaled')

    def _get_train_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load X_train and y_train for learning curve data."""
        if 'X_train' not in self._cache:
            self._cache['X_train'] = _load_csv("data/03_primary/X_train_selected.csv")
        if 'y_train' not in self._cache:
            self._cache['y_train'] = _load_pickle("data/03_primary/y_train_raw.pkl")
            if self._cache['y_train'] is None:
                self._cache['y_train'] = _load_pickle("data/03_primary/y_train_balanced.pkl")
        return self._cache.get('X_train'), self._cache.get('y_train')

    def _get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Load Phase 2 feature importance CSV."""
        if 'feature_importance' not in self._cache:
            self._cache['feature_importance'] = _load_csv("data/08_reporting/feature_importance.csv")
        return self._cache.get('feature_importance')

    def _get_phase4_report(self) -> Optional[pd.DataFrame]:
        """Load Phase 4 ranked report (all algorithms)."""
        if 'phase4_report' not in self._cache:
            self._cache['phase4_report'] = _load_csv("data/07_model_output/phase4_report.csv")
        return self._cache.get('phase4_report')

    def _get_phase4_summary(self) -> Optional[dict]:
        """Load Phase 4 summary (pickle file per catalog, fallback to JSON)."""
        if 'phase4_summary' not in self._cache:
            # Catalog defines phase4_summary as PickleDataset
            result = _load_pickle("data/07_model_output/phase4_summary.pkl")
            if result is None:
                # Fallback: try JSON in case it was saved differently
                result = _load_json("data/07_model_output/phase4_summary.json")
            self._cache['phase4_summary'] = result
        return self._cache.get('phase4_summary')

    def _get_phase5_metrics(self) -> Optional[dict]:
        """Load Phase 5 comprehensive metrics (pickle file per catalog, fallback to JSON)."""
        if 'phase5_metrics' not in self._cache:
            # Catalog defines phase5_metrics as PickleDataset
            result = _load_pickle("data/08_reporting/phase5_metrics.pkl")
            if result is None:
                # Fallback: try JSON
                result = _load_json("data/08_reporting/phase5_metrics.json")
            self._cache['phase5_metrics'] = result
        return self._cache.get('phase5_metrics')

    def _get_cross_validation(self) -> Optional[dict]:
        """Load cross-validation results from Phase 3."""
        if 'cv_results' not in self._cache:
            self._cache['cv_results'] = _load_pickle("data/08_reporting/cross_validation_results.pkl")
        return self._cache.get('cv_results')

    def _get_scalers(self) -> Optional[dict]:
        """Load scalers from Phase 3."""
        if 'scalers' not in self._cache:
            self._cache['scalers'] = _load_pickle("data/03_primary/scalers.pkl")
        return self._cache.get('scalers')

    def clear_cache(self):
        """Clear all cached data (call when pipeline is re-run)."""
        self._cache.clear()
        logger.info("Evaluation service cache cleared")

    # ------------------------------------------------------------------
    # PREDICTION PROBABILITIES (computed from model + test data)
    # ------------------------------------------------------------------

    def _get_y_pred_proba(self, model_id: Optional[str] = None) -> Optional[np.ndarray]:
        """Get prediction probabilities from a specific model on test data."""
        cache_key = f"y_pred_proba:{model_id or 'best'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        model = self._get_best_model(model_id)
        X_test = self._get_X_test_scaled()

        if model is None or X_test is None:
            logger.warning(f"Cannot compute y_pred_proba for '{model_id}': model or X_test missing")
            return None

        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                # For binary classification, take probability of positive class
                if proba.ndim == 2 and proba.shape[1] == 2:
                    self._cache[cache_key] = proba[:, 1]
                else:
                    self._cache[cache_key] = proba
            elif hasattr(model, 'decision_function'):
                self._cache[cache_key] = model.decision_function(X_test)
            else:
                logger.warning(f"Model {type(model).__name__} has no predict_proba or decision_function")
                return None

            return self._cache[cache_key]
        except Exception as e:
            logger.error(f"Error computing y_pred_proba for '{model_id}': {e}")
            return None

    def _get_y_test_numeric(self) -> Optional[np.ndarray]:
        """Get numeric y_test (handle string labels)."""
        if 'y_test_numeric' in self._cache:
            return self._cache['y_test_numeric']

        _, y_test = self._get_test_data()
        if y_test is None:
            # Fallback: try from predictions CSV
            pred_df = self._get_predictions_df()
            if pred_df is not None and 'actual' in pred_df.columns:
                y_test = pred_df['actual']
            else:
                return None

        y = np.asarray(y_test)
        # Convert string labels to numeric
        if y.dtype.kind in ('U', 'S', 'O'):
            classes = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(classes)}
            y = np.array([label_map[v] for v in y])
            self._cache['label_map'] = label_map
            self._cache['reverse_label_map'] = {v: k for k, v in label_map.items()}

        self._cache['y_test_numeric'] = y.astype(float)
        return self._cache['y_test_numeric']

    # ------------------------------------------------------------------
    # TAB 1: OVERVIEW - Threshold Evaluation
    # ------------------------------------------------------------------

    def compute_threshold_evaluation(self, threshold: float = 0.5, model_id: Optional[str] = None) -> Optional[dict]:
        """
        Compute confusion matrix, metrics, and rates at a given threshold.
        Returns the thresholdEvaluation object for the frontend.
        """
        y_test = self._get_y_test_numeric()
        y_proba = self._get_y_pred_proba(model_id)

        if y_test is None or y_proba is None:
            logger.error("Cannot compute threshold evaluation: missing y_test or y_pred_proba")
            return None

        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)
        y_true = y_test.astype(int)

        # Confusion matrix
        from sklearn.metrics import (
            confusion_matrix, accuracy_score, precision_score,
            recall_score, f1_score, roc_auc_score
        )

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Multiclass - flatten to binary summary
            tp = int(np.diag(cm).sum())
            total = int(cm.sum())
            fp = fn = (total - tp) // 2
            tn = total - tp - fp - fn

        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        total = tn + fp + fn + tp

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall_val = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

        try:
            auc_roc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc_roc = 0.0

        # Rates
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "confusionMatrix": {
                "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total": total
            },
            "metrics": {
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall_val, 6),
                "f1Score": round(f1, 6),
                "aucRoc": round(auc_roc, 6),
            },
            "rates": {
                "falsePositiveRate": round(fpr_rate, 6),
                "falseNegativeRate": round(fnr_rate, 6),
            }
        }

    # ------------------------------------------------------------------
    # TAB 2: BUSINESS IMPACT
    # ------------------------------------------------------------------

    def compute_business_impact(
            self,
            threshold: float = 0.5,
            cost_fp: float = 500,
            cost_fn: float = 2000,
            revenue_tp: float = 1000,
            volume: Optional[int] = None,
            model_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Compute business impact from confusion matrix.
        Returns the businessImpact object for the frontend.
        """
        eval_data = self.compute_threshold_evaluation(threshold, model_id)
        if eval_data is None:
            return None

        cm = eval_data["confusionMatrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        total = cm["total"]

        # Scale to volume if provided
        if volume and total > 0:
            scale = volume / total
            tp_scaled = int(round(tp * scale))
            fp_scaled = int(round(fp * scale))
            fn_scaled = int(round(fn * scale))
            tn_scaled = int(round(tn * scale))
        else:
            tp_scaled, fp_scaled, fn_scaled, tn_scaled = tp, fp, fn, tn
            volume = total

        # Costs
        fp_cost = fp_scaled * cost_fp
        fn_cost = fn_scaled * cost_fn
        total_cost = fp_cost + fn_cost

        # Revenue
        tp_revenue = tp_scaled * revenue_tp

        # Profit
        profit = tp_revenue - total_cost

        # Baseline comparison (predict all positive = no model)
        baseline_total_positive = tp_scaled + fn_scaled
        baseline_total_negative = tn_scaled + fp_scaled
        baseline_revenue = baseline_total_positive * revenue_tp  # all positives caught
        baseline_cost = baseline_total_negative * cost_fp  # all negatives are false positives
        baseline_profit = baseline_revenue - baseline_cost

        improvement = 0.0
        if baseline_profit != 0:
            improvement = ((profit - baseline_profit) / abs(baseline_profit)) * 100

        # Approval rate
        predicted_positive = tp_scaled + fp_scaled
        approval_rate = predicted_positive / volume if volume > 0 else 0.0

        result = {
            "costs": {
                "totalCost": round(total_cost, 2),
                "falsePositiveCost": round(fp_cost, 2),
                "falseNegativeCost": round(fn_cost, 2),
                "costPerFalsePositive": cost_fp,
                "costPerFalseNegative": cost_fn,
            },
            "revenue": {
                "truePositiveRevenue": round(tp_revenue, 2),
                "revenuePerTruePositive": revenue_tp,
            },
            "financial": {
                "profit": round(profit, 2),
                "improvementVsBaseline": round(improvement, 2),
                "atVolume": volume,
                "approvalRate": round(approval_rate, 4),
            },
        }

        if volume and total > 0 and volume != total:
            result["scaledCounts"] = {
                "truePositives": tp_scaled,
                "falsePositives": fp_scaled,
                "falseNegatives": fn_scaled,
            }

        return result

    # ------------------------------------------------------------------
    # TAB 3: CURVES & THRESHOLD
    # ------------------------------------------------------------------

    def compute_curves(self, model_id: Optional[str] = None) -> Optional[dict]:
        """
        Compute ROC and Precision-Recall curve arrays.
        Returns the curves object for the frontend.
        """
        y_test = self._get_y_test_numeric()
        y_proba = self._get_y_pred_proba(model_id)

        if y_test is None or y_proba is None:
            return None

        y_true = y_test.astype(int)

        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

        try:
            # ROC Curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            # Downsample to max ~100 points for frontend performance
            roc_data = self._downsample_curve(fpr, tpr, roc_thresholds, max_points=100)

            # Precision-Recall Curve
            pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            ap_score = average_precision_score(y_true, y_proba)

            pr_data = self._downsample_curve(pr_recall, pr_precision, pr_thresholds, max_points=100)

            return {
                "rocCurve": {
                    "fpr": roc_data[0],
                    "tpr": roc_data[1],
                    "thresholds": roc_data[2],
                    "auc": round(float(roc_auc), 6),
                },
                "prCurve": {
                    "recall": pr_data[0],
                    "precision": pr_data[1],
                    "thresholds": pr_data[2],
                    "ap": round(float(ap_score), 6),
                }
            }
        except Exception as e:
            logger.error(f"Error computing curves: {e}")
            return None

    def compute_optimal_threshold(
            self,
            cost_fp: float = 500,
            cost_fn: float = 2000,
            revenue_tp: float = 1000,
            model_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Search for the optimal threshold that maximizes profit.
        Returns the optimalThreshold object for the frontend.
        """
        y_test = self._get_y_test_numeric()
        y_proba = self._get_y_pred_proba(model_id)

        if y_test is None or y_proba is None:
            return None

        y_true = y_test.astype(int)
        best_threshold = 0.5
        best_profit = float('-inf')

        for t in np.arange(0.1, 0.91, 0.01):
            y_pred = (y_proba >= t).astype(int)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape != (2, 2):
                continue
            tn, fp, fn, tp = cm.ravel()

            profit = (tp * revenue_tp) - (fp * cost_fp) - (fn * cost_fn)
            if profit > best_profit:
                best_profit = profit
                best_threshold = t

        # Generate recommendation text
        recommendation = (
            f"Adjust threshold to {best_threshold:.2f} for maximum profit of "
            f"${best_profit:,.0f}. This optimizes the trade-off between "
            f"false positive cost (${cost_fp:,.0f}) and false negative cost (${cost_fn:,.0f})."
        )

        return {
            "optimalThreshold": round(float(best_threshold), 2),
            "optimalProfit": round(float(best_profit), 2),
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # TAB 4: ADVANCED ANALYSIS - Learning Curve & Feature Importance
    # ------------------------------------------------------------------

    def compute_learning_curve(self, model_id: Optional[str] = None) -> Optional[dict]:
        """
        Compute learning curve data (train vs test accuracy + overfitting).
        Returns the learningCurve object for the frontend.
        """
        model = self._get_best_model(model_id)
        eval_data = self._get_model_evaluation()

        # For non-best models, compute train/test scores directly
        resolved_key = self._resolve_model_id(model_id)
        if resolved_key is not None:
            # This is a specific Phase 4 model - get scores from report
            phase4_report = self._get_phase4_report()
            if phase4_report is not None:
                row = phase4_report[phase4_report['Algorithm'] == resolved_key]
                if not row.empty:
                    train_score = float(row.iloc[0].get('Train_Score', 0))
                    test_score = float(row.iloc[0].get('Test_Score', 0))
                    gap = abs(train_score - test_score)
                    status = "acceptable" if gap < 0.03 else "moderate" if gap < 0.08 else "overfitting"
                    return {
                        "trainAccuracy": round(train_score, 6),
                        "testAccuracy": round(test_score, 6),
                        "overfittingRatio": round(gap, 6),
                        "status": status,
                    }

        if eval_data is None:
            return None

        train_score = eval_data.get('train_score', 0)
        test_score = eval_data.get('test_score', 0)

        # Also check for accuracy keys
        if 'accuracy' in eval_data:
            test_score = eval_data['accuracy']

        # Overfitting ratio = gap between train and test
        gap = abs(train_score - test_score)
        overfitting_ratio = gap

        if gap < 0.03:
            status = "acceptable"
        elif gap < 0.08:
            status = "moderate"
        else:
            status = "overfitting"

        return {
            "trainAccuracy": round(float(train_score), 6),
            "testAccuracy": round(float(test_score), 6),
            "overfittingRatio": round(float(overfitting_ratio), 6),
            "status": status,
        }

    def compute_feature_importance(self, model_id: Optional[str] = None) -> Optional[dict]:
        """
        Compute feature importance with correlations.
        Returns the featureImportance object for the frontend.
        """
        # Try Phase 2 feature importance CSV first
        fi_df = self._get_feature_importance()

        # Also try to get model's built-in feature importance
        model = self._get_best_model(model_id)
        X_test = self._get_X_test_scaled()
        _, y_test = self._get_test_data()

        features_list = []
        interactions_list = []

        if fi_df is not None and 'feature' in fi_df.columns and 'importance' in fi_df.columns:
            # Use Phase 2 feature importance
            total_importance = fi_df['importance'].sum()

            for _, row in fi_df.iterrows():
                name = str(row['feature'])
                imp = float(row['importance'])
                imp_pct = (imp / total_importance * 100) if total_importance > 0 else 0

                # Compute correlation with target if possible
                corr = 0.0
                if X_test is not None and y_test is not None and name in X_test.columns:
                    try:
                        y_arr = np.asarray(y_test)
                        if y_arr.dtype.kind in ('U', 'S', 'O'):
                            classes = np.unique(y_arr)
                            label_map = {label: idx for idx, label in enumerate(classes)}
                            y_arr = np.array([label_map[v] for v in y_arr]).astype(float)
                        corr = float(np.corrcoef(X_test[name].values.astype(float), y_arr.astype(float))[0, 1])
                        if np.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0

                strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak" if abs(corr) > 0.1 else "negligible"

                features_list.append({
                    "name": name,
                    "importancePercent": round(imp_pct, 2),
                    "correlationWithTarget": round(corr, 4),
                    "correlationStrength": strength,
                })

        elif model is not None and hasattr(model, 'feature_importances_') and X_test is not None:
            # Use model's built-in feature importance
            importances = model.feature_importances_
            feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else [f"feature_{i}" for i in range(len(importances))]
            total = sum(importances)

            sorted_indices = np.argsort(importances)[::-1]
            for idx in sorted_indices:
                name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                imp = importances[idx]
                imp_pct = (imp / total * 100) if total > 0 else 0

                corr = 0.0
                if y_test is not None and name in (X_test.columns if hasattr(X_test, 'columns') else []):
                    try:
                        y_arr = np.asarray(y_test)
                        if y_arr.dtype.kind in ('U', 'S', 'O'):
                            classes = np.unique(y_arr)
                            label_map = {label: idx2 for idx2, label in enumerate(classes)}
                            y_arr = np.array([label_map[v] for v in y_arr]).astype(float)
                        corr = float(np.corrcoef(X_test[name].values.astype(float), y_arr.astype(float))[0, 1])
                        if np.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0

                strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak" if abs(corr) > 0.1 else "negligible"

                features_list.append({
                    "name": name,
                    "importancePercent": round(imp_pct, 2),
                    "correlationWithTarget": round(corr, 4),
                    "correlationStrength": strength,
                })

        if not features_list:
            return None

        # Compute feature interactions from correlation matrix
        if X_test is not None and len(features_list) >= 2:
            try:
                top_features = [f["name"] for f in features_list[:10]]
                available = [f for f in top_features if f in X_test.columns]

                if len(available) >= 2:
                    corr_matrix = X_test[available].corr()
                    pairs_seen = set()

                    for i, f1 in enumerate(available):
                        for j, f2 in enumerate(available):
                            if i >= j:
                                continue
                            pair_key = tuple(sorted([f1, f2]))
                            if pair_key in pairs_seen:
                                continue
                            pairs_seen.add(pair_key)

                            corr_val = corr_matrix.loc[f1, f2]
                            if abs(corr_val) > 0.05:
                                interactions_list.append({
                                    "feature1": f1,
                                    "feature2": f2,
                                    "interactionStrength": round(abs(float(corr_val)), 4),
                                    "interactionDirection": "positive" if corr_val > 0 else "negative",
                                })

                    interactions_list.sort(key=lambda x: x["interactionStrength"], reverse=True)
                    interactions_list = interactions_list[:10]  # Top 10 interactions
            except Exception as e:
                logger.warning(f"Error computing feature interactions: {e}")

        return {
            "features": features_list,
            "interactions": interactions_list,
        }

    # ------------------------------------------------------------------
    # TAB 5: PRODUCTION READINESS
    # ------------------------------------------------------------------

    def compute_production_readiness(self, threshold: float = 0.5, model_id: Optional[str] = None) -> dict:
        """
        Evaluate production readiness criteria.
        Returns the productionReadiness object for the frontend.
        """
        eval_data = self.compute_threshold_evaluation(threshold, model_id)
        learning = self.compute_learning_curve(model_id)
        fi = self.compute_feature_importance(model_id)
        cv = self._get_cross_validation()
        phase4_summary = self._get_phase4_summary()
        model = self._get_best_model(model_id)

        criteria: List[dict] = []

        # --- Performance Criteria ---
        if eval_data:
            m = eval_data["metrics"]

            criteria.append({
                "name": f"AUC-ROC ({m['aucRoc']:.3f})",
                "description": f"AUC-ROC score is {m['aucRoc']:.3f}. Threshold: ≥ 0.70",
                "passed": m['aucRoc'] >= 0.70,
                "category": "Performance",
            })
            criteria.append({
                "name": f"F1 Score ({m['f1Score']*100:.1f}%)",
                "description": f"F1 Score is {m['f1Score']*100:.1f}%. Threshold: ≥ 60%",
                "passed": m['f1Score'] >= 0.60,
                "category": "Performance",
            })
            criteria.append({
                "name": f"Precision ({m['precision']*100:.1f}%)",
                "description": f"Precision is {m['precision']*100:.1f}%. Threshold: ≥ 50%",
                "passed": m['precision'] >= 0.50,
                "category": "Performance",
            })
            criteria.append({
                "name": f"Recall ({m['recall']*100:.1f}%)",
                "description": f"Recall is {m['recall']*100:.1f}%. Threshold: ≥ 50%",
                "passed": m['recall'] >= 0.50,
                "category": "Performance",
            })

        # --- Stability & Robustness ---
        if learning:
            criteria.append({
                "name": f"Overfitting Risk ({learning['status']})",
                "description": f"Train-test gap: {learning['overfittingRatio']*100:.1f}%. Status: {learning['status']}",
                "passed": learning['status'] in ('acceptable', 'moderate'),
                "category": "Stability & Robustness",
            })

        if cv:
            cv_mean = cv.get('mean', 0)
            cv_std = cv.get('std', 0)
            criteria.append({
                "name": f"Cross-Validation (mean={cv_mean:.3f})",
                "description": f"5-fold CV mean: {cv_mean:.3f} ± {cv_std:.3f}",
                "passed": cv_std < 0.05,
                "category": "Stability & Robustness",
            })

        criteria.append({
            "name": "Model Convergence",
            "description": "Model training completed without errors",
            "passed": model is not None,
            "category": "Stability & Robustness",
        })

        # --- Data Quality ---
        X_test = self._get_X_test_scaled()
        if X_test is not None:
            criteria.append({
                "name": f"Test Set Size ({len(X_test)} samples)",
                "description": f"Test set has {len(X_test)} samples. Minimum recommended: 100",
                "passed": len(X_test) >= 100,
                "category": "Data Quality",
            })

        criteria.append({
            "name": "Feature Quality",
            "description": "Feature importance analysis available" if fi else "Feature importance not computed",
            "passed": fi is not None and len(fi.get('features', [])) > 0,
            "category": "Data Quality",
        })

        # --- Explainability ---
        criteria.append({
            "name": "Feature Importance Available",
            "description": "Feature importance scores computed for model interpretability",
            "passed": fi is not None and len(fi.get('features', [])) > 0,
            "category": "Explainability",
        })

        model_name = type(model).__name__ if model else "Unknown"
        is_interpretable = any(kw in model_name.lower() for kw in ['forest', 'tree', 'logistic', 'linear', 'boost', 'xgb', 'lgbm', 'catboost'])
        criteria.append({
            "name": f"Model Interpretability ({model_name})",
            "description": f"{model_name} {'is' if is_interpretable else 'may not be'} inherently interpretable",
            "passed": is_interpretable,
            "category": "Explainability",
        })

        # --- Infrastructure ---
        criteria.append({
            "name": "Model Serialization",
            "description": "Model is serialized as pickle file and can be loaded for inference",
            "passed": model is not None,
            "category": "Infrastructure",
        })

        scalers = self._get_scalers()
        criteria.append({
            "name": "Preprocessing Pipeline",
            "description": "Scalers and preprocessors saved for consistent inference",
            "passed": scalers is not None,
            "category": "Infrastructure",
        })

        # Check if multiple algorithms were compared (Phase 4)
        criteria.append({
            "name": "Algorithm Comparison",
            "description": f"Compared {phase4_summary.get('total_models', 0)} algorithms" if phase4_summary else "Phase 4 algorithm comparison not available",
            "passed": phase4_summary is not None and phase4_summary.get('total_models', 0) > 1,
            "category": "Infrastructure",
        })

        # --- Summary ---
        passed = sum(1 for c in criteria if c["passed"])
        total = len(criteria)
        pct = (passed / total * 100) if total > 0 else 0

        if pct >= 80:
            status = "READY"
        elif pct >= 50:
            status = "WARNING"
        else:
            status = "NOT_READY"

        return {
            "overallStatus": status,
            "summary": {
                "passed": passed,
                "totalCriteria": total,
                "passPercentage": round(pct, 1),
            },
            "criteria": criteria,
        }

    # ------------------------------------------------------------------
    # OVERALL SCORE
    # ------------------------------------------------------------------

    def compute_overall_score(self, threshold: float = 0.5, model_id: Optional[str] = None) -> float:
        """Compute a single overall quality score (0-100)."""
        eval_data = self.compute_threshold_evaluation(threshold, model_id)
        if eval_data is None:
            return 0.0

        m = eval_data["metrics"]
        # Weighted average of key metrics
        score = (
                m['accuracy'] * 15 +
                m['precision'] * 20 +
                m['recall'] * 20 +
                m['f1Score'] * 25 +
                m['aucRoc'] * 20
        )
        return round(min(score, 100.0), 1)

    # ------------------------------------------------------------------
    # COMBINED: Returns everything for all 5 tabs
    # ------------------------------------------------------------------

    def get_complete_evaluation(
            self,
            threshold: float = 0.5,
            cost_fp: float = 500,
            cost_fn: float = 2000,
            revenue_tp: float = 1000,
            volume: Optional[int] = None,
            model_id: Optional[str] = None,
    ) -> dict:
        """
        Returns the complete CompleteEvaluationResponse for the frontend.
        This single call feeds all 5 tabs of the Model Evaluation screen.

        Args:
            model_id: Algorithm ID (e.g. 'adaboostclassifier') or None/'best' for overall best.
        """
        logger.info(f"Computing complete evaluation (model={model_id or 'best'}, threshold={threshold})")

        result = {}

        # Tab 1: Overview
        result["thresholdEvaluation"] = self.compute_threshold_evaluation(threshold, model_id)

        # Tab 2: Business Impact
        result["businessImpact"] = self.compute_business_impact(
            threshold, cost_fp, cost_fn, revenue_tp, volume, model_id
        )

        # Tab 3: Curves & Threshold
        result["curves"] = self.compute_curves(model_id)
        result["optimalThreshold"] = self.compute_optimal_threshold(cost_fp, cost_fn, revenue_tp, model_id)

        # Tab 4: Advanced Analysis
        result["learningCurve"] = self.compute_learning_curve(model_id)
        result["featureImportance"] = self.compute_feature_importance(model_id)

        # Tab 5: Production Readiness
        result["productionReadiness"] = self.compute_production_readiness(threshold, model_id)

        # Overall Score
        result["overallScore"] = self.compute_overall_score(threshold, model_id)

        # Model info (tell the frontend which model this is)
        resolved = self._resolve_model_id(model_id)
        model_obj = self._get_best_model(model_id)
        result["modelInfo"] = {
            "requestedId": model_id,
            "resolvedName": resolved or (type(model_obj).__name__ if model_obj else "Unknown"),
            "algorithmType": type(model_obj).__name__ if model_obj else "Unknown",
        }

        logger.info(f"Complete evaluation for '{model_id or 'best'}': "
                    f"threshold={result.get('thresholdEvaluation') is not None}, "
                    f"business={result.get('businessImpact') is not None}, "
                    f"curves={result.get('curves') is not None}, "
                    f"learning={result.get('learningCurve') is not None}, "
                    f"features={result.get('featureImportance') is not None}, "
                    f"production={result.get('productionReadiness') is not None}")

        return _sanitize_for_json(result)

    # ------------------------------------------------------------------
    # TRAINED MODELS LIST (for model selector)
    # ------------------------------------------------------------------

    def list_trained_models(self) -> dict:
        """
        List all trained models available for evaluation.
        Combines Phase 3 best model + Phase 4 all algorithms.
        """
        models = []
        problem_type = self._get_problem_type()
        phase4_report = self._get_phase4_report()
        phase4_summary = self._get_phase4_summary()

        best_model_name = None
        if phase4_summary:
            best_model_name = phase4_summary.get('best_model')

        if phase4_report is not None:
            for _, row in phase4_report.iterrows():
                algo = str(row.get('Algorithm', 'Unknown'))
                test_score = float(row.get('Test_Score', 0)) if pd.notna(row.get('Test_Score')) else None
                train_score = float(row.get('Train_Score', 0)) if pd.notna(row.get('Train_Score')) else None

                # Create a stable ID from algorithm name
                model_id = algo.lower().replace(' ', '_').replace('(', '').replace(')', '')

                models.append({
                    "id": model_id,
                    "name": algo,
                    "algorithm": algo,
                    "accuracy": test_score,
                    "testScore": test_score,
                    "problemType": problem_type,
                    "trainedAt": None,
                })

        # If no Phase 4, at least list the Phase 3 best model
        if not models:
            model = self._get_best_model()
            if model is not None:
                eval_data = self._get_model_evaluation()
                model_name = type(model).__name__
                models.append({
                    "id": "best_model",
                    "name": model_name,
                    "algorithm": model_name,
                    "accuracy": eval_data.get('accuracy') if eval_data else None,
                    "testScore": eval_data.get('test_score') if eval_data else None,
                    "problemType": problem_type,
                    "trainedAt": None,
                })
                best_model_name = model_name

        best_id = None
        if best_model_name:
            best_id = best_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

        return {
            "models": models,
            "bestModelId": best_id,
            "totalModels": len(models),
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample_curve(
            x: np.ndarray, y: np.ndarray, thresholds: np.ndarray,
            max_points: int = 100
    ) -> Tuple[List[float], List[float], List[float]]:
        """Downsample curve arrays to max_points for frontend performance."""
        n = len(x)
        if n <= max_points:
            t_list = thresholds.tolist() if len(thresholds) == n else (thresholds.tolist() + [None] * (n - len(thresholds)))
            return (
                [round(float(v), 6) for v in x],
                [round(float(v), 6) for v in y],
                [round(float(v), 6) if v is not None else None for v in t_list[:n]],
            )

        indices = np.linspace(0, n - 1, max_points, dtype=int)
        # Always include first and last point
        indices = np.unique(np.concatenate([[0], indices, [n - 1]]))

        x_down = x[indices]
        y_down = y[indices]
        t_down = thresholds[indices] if len(thresholds) >= n else np.interp(
            indices, np.arange(len(thresholds)), thresholds
        )

        return (
            [round(float(v), 6) for v in x_down],
            [round(float(v), 6) for v in y_down],
            [round(float(v), 6) for v in t_down],
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_service_instance: Optional[ModelEvaluationService] = None


def get_evaluation_service() -> ModelEvaluationService:
    """Get or create the singleton evaluation service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelEvaluationService()
    return _service_instance