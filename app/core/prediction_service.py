"""
Prediction Service
==================
Handles real-time and batch predictions using Kedro pipeline artifacts.
All response shapes match the frontend mock data contracts exactly.

Frontend components served:
  PredictionsDashboard.tsx → model.accuracy (0-1), output.predictionLabel, etc.
  SinglePredictionTab.tsx  → predictionResult.prediction, .confidence, .explanation[]
  BatchPredictionTab.tsx   → batchResults.approved, .rejected, .total, .avgConfidence
  HistoryTab.tsx           → item.predictedLabel, .predictedClass, .confidence
  MonitoringTab            → stats.totalPredictions, confidenceDistribution[{range,count}]

Data sources (from KEDRO_PROJECT_PATH):
  data/06_models/phase4/all_trained_models.pkl
  data/06_models/best_model.pkl
  data/07_model_output/phase4_report.csv
  data/03_primary/scalers.pkl
  data/06_models/encoder.pkl, scaler.pkl, imputer.pkl
  data/03_primary/X_test_scaled.csv
"""

import os
import io
import pickle
import time
import uuid
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KEDRO_PROJECT_PATH = Path(os.getenv(
    'KEDRO_PROJECT_PATH',
    '/home/ashok/work/latest/full/kedro-ml-engine-integrated'
))

# Class labels for the Adult Census Income dataset
POSITIVE_CLASS = ">50K"
NEGATIVE_CLASS = "<=50K"
POSITIVE_LABEL = "High Income (>$50K)"
NEGATIVE_LABEL = "Low Income (≤$50K)"


def _resolve(relative_path: str) -> str:
    return str(KEDRO_PROJECT_PATH / relative_path)


def _load_pickle(path: str):
    full = _resolve(path)
    try:
        with open(full, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load {full}: {e}")
        return None


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    full = _resolve(path)
    try:
        return pd.read_csv(full)
    except Exception as e:
        logger.warning(f"Could not load {full}: {e}")
        return None


class PredictionService:
    """
    Handles real-time and batch predictions using Kedro pipeline artifacts.
    All response shapes are frontend-compatible.
    """

    def __init__(self):
        self._all_models: Dict[str, Any] = {}
        self._best_model = None
        self._report_df = pd.DataFrame()
        self._preprocessor_cache: Dict[str, Any] = {}
        self._phase3_scalers: Dict[str, Any] = {}
        self._expected_columns: List[str] = []

        # In-memory storage (swap to DB later)
        self._prediction_history: List[Dict] = []
        self._batch_jobs: Dict[str, Dict] = {}
        self._monitoring = {
            "total_predictions": 0,
            "total_latency_ms": 0.0,
            "errors": 0,
            "class_counts": {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 0},
            "hourly_counts": {},
            "confidence_buckets": {
                "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                "0.6-0.8": 0, "0.8-1.0": 0
            },
            "hourly_confidence_sums": {},  # for computing avg confidence per hour
        }

        self._load_artifacts()

    # ========================================================================
    # ARTIFACT LOADING
    # ========================================================================

    def _load_artifacts(self):
        logger.info("Loading Kedro artifacts for prediction service...")

        all_models = _load_pickle("data/06_models/phase4/all_trained_models.pkl")
        if all_models and isinstance(all_models, dict):
            self._all_models = all_models
            logger.info(f"Loaded {len(self._all_models)} trained models from Phase 4")

        self._best_model = _load_pickle("data/06_models/best_model.pkl")
        if self._best_model:
            logger.info("Loaded best model from Phase 3")

        # ✅ FIXED — explicit None check avoids __bool__() on DataFrame
        _report = _load_csv("data/07_model_output/phase4_report.csv")
        self._report_df = _report if _report is not None else pd.DataFrame()

        for name in ["encoder", "scaler", "imputer"]:
            artifact = _load_pickle(f"data/06_models/{name}.pkl")
            if artifact is not None:
                self._preprocessor_cache[name] = artifact
                logger.info(f"Loaded preprocessor: {name}")

        X_ref = _load_csv("data/03_primary/X_test_scaled.csv")
        if X_ref is not None:
            self._expected_columns = list(X_ref.columns)
            logger.info(f"Model expects {len(self._expected_columns)} features")

        self._phase3_scalers = _load_pickle("data/03_primary/scalers.pkl") or {}
        logger.info("Prediction service initialization complete")

    # ========================================================================
    # INPUT FEATURE SCHEMA — drives the frontend form
    # ========================================================================

    def get_input_feature_schema(self) -> List[Dict[str, Any]]:
        """
        Returns input features for the frontend form.
        Frontend uses: feature.name, feature.type ("numeric"/"categorical"),
                       feature.min, feature.max, feature.options, feature.description
        """
        return [
            {
                "name": "age", "displayName": "Age",
                "type": "numeric", "inputType": "number",
                "min": 17, "max": 90, "step": 1,
                "description": "Person's age in years",
                "defaultValue": 35, "required": True,
                "typicalRange": "25-65"
            },
            {
                "name": "workclass", "displayName": "Work Class",
                "type": "categorical", "inputType": "select",
                "options": [
                    "Private", "Self-emp-not-inc", "Self-emp-inc",
                    "Federal-gov", "Local-gov", "State-gov",
                    "Without-pay", "Never-worked"
                ],
                "description": "Type of employment",
                "defaultValue": "Private", "required": True
            },
            {
                "name": "education", "displayName": "Education",
                "type": "categorical", "inputType": "select",
                "options": [
                    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
                    "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
                    "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"
                ],
                "description": "Highest education level",
                "defaultValue": "HS-grad", "required": True
            },
            {
                "name": "education_num", "displayName": "Education Years",
                "type": "numeric", "inputType": "number",
                "min": 1, "max": 16, "step": 1,
                "description": "Number of years of education",
                "defaultValue": 10, "required": True,
                "typicalRange": "9-14"
            },
            {
                "name": "marital_status", "displayName": "Marital Status",
                "type": "categorical", "inputType": "select",
                "options": [
                    "Married-civ-spouse", "Divorced", "Never-married",
                    "Separated", "Widowed", "Married-spouse-absent",
                    "Married-AF-spouse"
                ],
                "description": "Current marital status",
                "defaultValue": "Never-married", "required": True
            },
            {
                "name": "occupation", "displayName": "Occupation",
                "type": "categorical", "inputType": "select",
                "options": [
                    "Tech-support", "Craft-repair", "Other-service", "Sales",
                    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                    "Transport-moving", "Priv-house-serv", "Protective-serv",
                    "Armed-Forces"
                ],
                "description": "Type of occupation",
                "defaultValue": "Other-service", "required": True
            },
            {
                "name": "relationship", "displayName": "Relationship",
                "type": "categorical", "inputType": "select",
                "options": [
                    "Wife", "Own-child", "Husband",
                    "Not-in-family", "Other-relative", "Unmarried"
                ],
                "description": "Family relationship role",
                "defaultValue": "Not-in-family", "required": True
            },
            {
                "name": "race", "displayName": "Race",
                "type": "categorical", "inputType": "select",
                "options": [
                    "White", "Asian-Pac-Islander",
                    "Amer-Indian-Eskimo", "Other", "Black"
                ],
                "description": "Racial category",
                "defaultValue": "White", "required": True
            },
            {
                "name": "sex", "displayName": "Sex",
                "type": "categorical", "inputType": "select",
                "options": ["Male", "Female"],
                "description": "Biological sex",
                "defaultValue": "Male", "required": True
            },
            {
                "name": "capital_gain", "displayName": "Capital Gain",
                "type": "numeric", "inputType": "number",
                "min": 0, "max": 99999, "step": 100,
                "description": "Capital gains in USD",
                "defaultValue": 0, "required": True,
                "typicalRange": "0-5000"
            },
            {
                "name": "capital_loss", "displayName": "Capital Loss",
                "type": "numeric", "inputType": "number",
                "min": 0, "max": 4356, "step": 100,
                "description": "Capital losses in USD",
                "defaultValue": 0, "required": True,
                "typicalRange": "0-2000"
            },
            {
                "name": "hours_per_week", "displayName": "Hours per Week",
                "type": "numeric", "inputType": "number",
                "min": 1, "max": 99, "step": 1,
                "description": "Working hours per week",
                "defaultValue": 40, "required": True,
                "typicalRange": "30-50"
            },
            {
                "name": "native_country", "displayName": "Native Country",
                "type": "categorical", "inputType": "select",
                "options": [
                    "United-States", "Cambodia", "England", "Puerto-Rico",
                    "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
                    "Japan", "Greece", "South", "China", "Cuba", "Iran",
                    "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                    "Vietnam", "Mexico", "Portugal", "Ireland", "France",
                    "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
                    "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
                    "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                    "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
                ],
                "description": "Country of origin",
                "defaultValue": "United-States", "required": True
            }
        ]

    # ========================================================================
    # PREPROCESSING PIPELINE (replicates Kedro)
    # ========================================================================

    _COLUMN_MAPPING = {
        "age": "age",
        "workclass": "workclass",
        "education": "education",
        "education_num": "education.num",
        "marital_status": "marital.status",
        "occupation": "occupation",
        "relationship": "relationship",
        "race": "race",
        "sex": "sex",
        "capital_gain": "capital.gain",
        "capital_loss": "capital.loss",
        "hours_per_week": "hours.per.week",
        "native_country": "native.country",
    }

    def _preprocess_raw_input(self, raw_features: Dict[str, Any]) -> pd.DataFrame:
        raw_row = {}
        for fe_name, csv_name in self._COLUMN_MAPPING.items():
            raw_row[csv_name] = raw_features.get(fe_name)
        raw_df = pd.DataFrame([raw_row])

        expected_cols = self._expected_columns

        if "scaler" in self._preprocessor_cache and expected_cols:
            return self._transform_with_saved_preprocessors(raw_df, expected_cols)
        return self._transform_manually(raw_df, expected_cols)

    def _transform_with_saved_preprocessors(
            self, raw_df: pd.DataFrame, expected_cols: List[str]
    ) -> pd.DataFrame:
        encoder = self._preprocessor_cache.get("encoder")
        scaler = self._preprocessor_cache.get("scaler")
        imputer = self._preprocessor_cache.get("imputer")

        if imputer and hasattr(imputer, "transform"):
            try:
                raw_df = pd.DataFrame(imputer.transform(raw_df), columns=raw_df.columns)
            except Exception as e:
                logger.warning(f"Imputer transform failed: {e}")

        if encoder and hasattr(encoder, "transform"):
            try:
                cat_cols = raw_df.select_dtypes(include=["object"]).columns
                if len(cat_cols) > 0:
                    encoded = encoder.transform(raw_df[cat_cols])
                    if hasattr(encoded, "toarray"):
                        encoded = encoded.toarray()
                    encoded_df = pd.DataFrame(encoded)
                    num_df = raw_df.drop(columns=cat_cols).reset_index(drop=True)
                    raw_df = pd.concat([num_df, encoded_df], axis=1)
            except Exception as e:
                logger.warning(f"Encoder transform failed: {e}")

        if scaler and hasattr(scaler, "transform"):
            try:
                num_cols = raw_df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    raw_df[num_cols] = scaler.transform(raw_df[num_cols])
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")

        if self._phase3_scalers and "numeric_scaler" in self._phase3_scalers:
            try:
                phase3_scaler = self._phase3_scalers["numeric_scaler"]
                phase3_cols = self._phase3_scalers.get("numeric_cols", [])
                matching = [c for c in phase3_cols if c in raw_df.columns]
                if matching:
                    raw_df[matching] = phase3_scaler.transform(raw_df[matching])
            except Exception as e:
                logger.warning(f"Phase 3 scaler failed: {e}")

        for col in expected_cols:
            if col not in raw_df.columns:
                raw_df[col] = 0
        raw_df = raw_df[expected_cols]
        return raw_df

    def _transform_manually(
            self, raw_df: pd.DataFrame, expected_cols: List[str]
    ) -> pd.DataFrame:
        from sklearn.preprocessing import LabelEncoder

        numeric_cols = ["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
        categorical_cols = [
            "workclass", "education", "marital.status", "occupation",
            "relationship", "race", "sex", "native.country"
        ]

        X_train_ref = _load_csv("data/03_primary/X_train_raw.csv")
        result = pd.DataFrame()

        for col in numeric_cols:
            if col in raw_df.columns:
                if X_train_ref is not None and col in X_train_ref.columns:
                    mean_val = X_train_ref[col].mean()
                    std_val = X_train_ref[col].std()
                    result[f"{col}_scaled"] = (
                        (raw_df[col].astype(float) - mean_val) / std_val if std_val > 0 else 0
                    )
                else:
                    result[f"{col}_scaled"] = raw_df[col].astype(float)

        for col in categorical_cols:
            if col not in raw_df.columns:
                continue
            n_unique = (
                X_train_ref[col].nunique() if X_train_ref is not None and col in X_train_ref.columns else 5
            )
            if n_unique <= 10:
                if X_train_ref is not None and col in X_train_ref.columns:
                    categories = sorted(X_train_ref[col].dropna().unique())
                    for cat in categories[1:]:
                        result[f"{col}_{cat}"] = int(raw_df[col].values[0] == cat)
                else:
                    result[col] = 0
            else:
                if X_train_ref is not None and col in X_train_ref.columns:
                    le = LabelEncoder()
                    le.fit(X_train_ref[col].astype(str))
                    val = raw_df[col].values[0]
                    result[col] = le.transform([str(val)])[0] if val in le.classes_ else -1
                else:
                    result[col] = 0

        if expected_cols:
            for col in expected_cols:
                if col not in result.columns:
                    result[col] = 0
            result = result[expected_cols]
        return result

    # ========================================================================
    # SINGLE PREDICTION
    # Response shape matches MOCK_SINGLE_PREDICTION from PredictionsContainer.tsx
    # ========================================================================

    def predict_single(
            self,
            raw_features: Dict[str, Any],
            model_id: Optional[str] = None,
            threshold: float = 0.5
    ) -> Dict[str, Any]:
        start = time.time()
        prediction_id = f"pred-{uuid.uuid4().hex[:8]}"

        try:
            model, model_name = self._resolve_model(model_id)
            X = self._preprocess_raw_input(raw_features)

            y_pred = model.predict(X)[0]
            y_proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                y_proba = float(proba[1])
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(X)[0]
                y_proba = float(1 / (1 + np.exp(-decision)))

            if y_proba is not None:
                prediction_class = 1 if y_proba >= threshold else 0
            else:
                prediction_class = int(y_pred)

            label = POSITIVE_CLASS if prediction_class == 1 else NEGATIVE_CLASS
            display_label = POSITIVE_LABEL if prediction_class == 1 else NEGATIVE_LABEL

            # Confidence (how sure the model is of its prediction)
            if y_proba is not None:
                raw_confidence = y_proba if prediction_class == 1 else (1 - y_proba)
            else:
                raw_confidence = 0.5

            if raw_confidence >= 0.85:
                confidence_label = "high"
            elif raw_confidence >= 0.65:
                confidence_label = "moderate"
            else:
                confidence_label = "low"

            # Probabilities dict for both classes
            prob_positive = y_proba if y_proba is not None else (1.0 if prediction_class == 1 else 0.0)
            prob_negative = 1 - prob_positive
            probabilities = {
                NEGATIVE_CLASS: round(prob_negative, 4),
                POSITIVE_CLASS: round(prob_positive, 4),
            }

            # Feature contributions
            contributions = self._compute_feature_contributions(model, X, raw_features)

            # Build explanation text
            top_positive = [c for c in contributions if c["direction"] == "positive"][:2]
            top_negative = [c for c in contributions if c["direction"] == "negative"][:1]
            explanation_text = self._build_explanation_text(
                label, raw_features, top_positive, top_negative
            )

            latency_ms = round((time.time() - start) * 1000, 1)
            now = datetime.utcnow().isoformat() + "Z"

            # ── FRONTEND-COMPATIBLE RESPONSE ──
            result = {
                "predictionId": prediction_id,
                "modelId": model_id or "best",
                "modelName": model_name,
                "timestamp": now,
                "input": raw_features,

                # Nested output block (PredictionsDashboard.tsx)
                "output": {
                    "prediction": label,
                    "predictionLabel": display_label,
                    "predictionValue": prediction_class,
                    "probability": round(prob_positive, 4),
                    "probabilities": probabilities,
                    "confidence": confidence_label,
                    "threshold": threshold,
                },

                # Explanation block (PredictionsDashboard.tsx)
                "explanation": {
                    "topFeatures": contributions,
                    "baselineScore": round(prob_negative, 4),
                    "explanation": explanation_text,
                },

                # Metadata block (PredictionsDashboard.tsx)
                "metadata": {
                    "processingTimeMs": latency_ms,
                    "modelVersion": "v1.0",
                },

                # Top-level aliases (SinglePredictionTab.tsx compatibility)
                "prediction": label,
                "confidence": round(raw_confidence, 4),
                "probabilities": {
                    label.lower().replace(" ", "_").replace(">", "gt").replace("<=", "lte"):
                        round(raw_confidence, 4),
                    (NEGATIVE_CLASS if prediction_class == 1 else POSITIVE_CLASS)
                    .lower().replace(" ", "_").replace(">", "gt").replace("<=", "lte"):
                        round(1 - raw_confidence, 4),
                },
            }

            self._update_monitoring(result)
            self._add_to_history(result, raw_features)
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            self._monitoring["errors"] += 1
            return {
                "predictionId": prediction_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def _resolve_model(self, model_id: Optional[str]) -> Tuple[Any, str]:
        if model_id is None or model_id == "best":
            if not self._report_df.empty:
                best_name = self._report_df.iloc[0].get("Algorithm", "AdaBoostClassifier")
            else:
                best_name = "AdaBoostClassifier"
            model_key = best_name
        else:
            model_key = None
            for key in self._all_models:
                if key.lower().replace(" ", "") == model_id.lower().replace(" ", ""):
                    model_key = key
                    break
            if not model_key:
                model_key = model_id

        if model_key in self._all_models:
            model_data = self._all_models[model_key]
            if isinstance(model_data, dict) and "model" in model_data:
                return model_data["model"], model_key
            return model_data, model_key

        if self._best_model is not None:
            return self._best_model, "BestModel"

        raise ValueError(f"Model '{model_id}' not found. Available: {list(self._all_models.keys())}")

    def _compute_feature_contributions(
            self, model, X: pd.DataFrame, raw_features: Dict
    ) -> List[Dict]:
        """
        Returns list of { feature, contribution, impact, direction, value }.
        Both 'contribution' and 'impact' have the same value (compat with both dashboard components).
        """
        contributions = []

        # Try SHAP
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            for i, col in enumerate(X.columns):
                sv = float(shap_values[0][i])
                impact_pct = round(abs(sv) * 100, 1)
                contributions.append({
                    "feature": col,
                    "contribution": round(abs(sv), 4),
                    "impact": impact_pct,
                    "direction": "positive" if sv > 0 else "negative",
                    "value": raw_features.get(col, raw_features.get(col.replace(".", "_"), "")),
                    "shapValue": round(sv, 4),
                })
            contributions.sort(key=lambda x: x["contribution"], reverse=True)
            return contributions[:8]
        except Exception:
            pass

        # Fallback: feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for i, col in enumerate(X.columns):
                if i < len(importances):
                    imp = float(importances[i])
                    impact_pct = round(imp * 100, 1)
                    contributions.append({
                        "feature": col,
                        "contribution": round(imp, 4),
                        "impact": impact_pct,
                        "direction": "positive" if float(X.iloc[0, i]) > 0 else "negative",
                        "value": raw_features.get(col, raw_features.get(col.replace(".", "_"), "")),
                    })
            contributions.sort(key=lambda x: x["contribution"], reverse=True)
            return contributions[:8]

        # Fallback: coef_
        if hasattr(model, "coef_"):
            coefs = model.coef_.flatten()
            for i, col in enumerate(X.columns):
                if i < len(coefs):
                    c = float(coefs[i])
                    impact_pct = round(abs(c) * 100, 1)
                    contributions.append({
                        "feature": col,
                        "contribution": round(abs(c), 4),
                        "impact": impact_pct,
                        "direction": "positive" if c > 0 else "negative",
                        "value": raw_features.get(col, raw_features.get(col.replace(".", "_"), "")),
                    })
            contributions.sort(key=lambda x: x["contribution"], reverse=True)
            return contributions[:8]

        return []

    def _build_explanation_text(
            self, prediction: str, raw_features: Dict,
            top_positive: List, top_negative: List
    ) -> str:
        parts = []
        if top_positive:
            feat_names = [f"{c['feature']} ({c.get('value', '')})" for c in top_positive]
            parts.append(f"Key factors supporting {prediction}: {', '.join(feat_names)}")
        if top_negative:
            feat_names = [f"{c['feature']} ({c.get('value', '')})" for c in top_negative]
            parts.append(f"Factors against: {', '.join(feat_names)}")
        return ". ".join(parts) + "." if parts else f"Prediction: {prediction}"

    # ========================================================================
    # BATCH PREDICTION
    # Response shape matches MOCK_BATCH_JOB from PredictionsContainer.tsx
    # ========================================================================

    _BATCH_COLUMN_MAPPING = {
        "age": "age", "workclass": "workclass", "education": "education",
        "education.num": "education_num", "education_num": "education_num",
        "marital.status": "marital_status", "marital_status": "marital_status",
        "occupation": "occupation", "relationship": "relationship",
        "race": "race", "sex": "sex",
        "capital.gain": "capital_gain", "capital_gain": "capital_gain",
        "capital.loss": "capital_loss", "capital_loss": "capital_loss",
        "hours.per.week": "hours_per_week", "hours_per_week": "hours_per_week",
        "native.country": "native_country", "native_country": "native_country",
    }

    def start_batch_prediction(
            self, csv_content: bytes, filename: str, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        job_id = f"batch-{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()

        try:
            df = pd.read_csv(io.BytesIO(csv_content))
            total_records = len(df)

            _, model_name = self._resolve_model(model_id)

            job = {
                "jobId": job_id,
                "modelId": model_id or "best",
                "modelName": model_name,
                "status": "processing",
                "progress": 0,
                "totalRecords": total_records,
                "processedRecords": 0,
                "successfulRecords": 0,
                "failedRecords": 0,
                "startedAt": start_time.isoformat() + "Z",
                "completedAt": None,
                "durationSeconds": None,
                "inputFile": {
                    "name": filename,
                    "size": len(csv_content),
                    "records": total_records,
                },
                "outputFile": None,
                "summary": None,
                "errors": [],
                "results": None,  # internal — not sent to frontend
            }
            self._batch_jobs[job_id] = job

            self._process_batch(job_id, df, model_id)
            return job

        except Exception as e:
            logger.error(f"Batch job creation failed: {e}", exc_info=True)
            return {"jobId": job_id, "status": "failed", "error": str(e)}

    def _process_batch(self, job_id: str, df: pd.DataFrame, model_id: Optional[str]):
        job = self._batch_jobs[job_id]
        results = []
        errors = []
        start_ts = time.time()

        for idx, row in df.iterrows():
            try:
                features = {}
                for csv_col, fe_name in self._BATCH_COLUMN_MAPPING.items():
                    if csv_col in df.columns:
                        val = row[csv_col]
                        features[fe_name] = val if pd.notna(val) else None

                if "age" not in features or features["age"] is None:
                    errors.append({"row": idx + 1, "error": "Missing required field: age"})
                    job["failedRecords"] += 1
                    continue

                result = self.predict_single(features, model_id)
                if "error" in result:
                    errors.append({"row": idx + 1, "error": result["error"]})
                    job["failedRecords"] += 1
                else:
                    row_result = {
                        "row": idx + 1,
                        "prediction": result["output"]["prediction"],
                        "probability": result["output"]["probability"],
                        "confidence": result["output"]["confidence"],
                    }
                    for k in df.columns:
                        if k not in row_result:
                            val = row[k]
                            row_result[k] = val if pd.notna(val) else None
                    results.append(row_result)
                    job["successfulRecords"] += 1

            except Exception as e:
                errors.append({"row": idx + 1, "error": str(e)})
                job["failedRecords"] += 1

            job["processedRecords"] = idx + 1
            job["progress"] = round((idx + 1) / len(df) * 100, 1)

        elapsed = time.time() - start_ts
        job["status"] = "completed"
        job["progress"] = 100
        job["completedAt"] = datetime.utcnow().isoformat() + "Z"
        job["durationSeconds"] = int(elapsed)
        job["errors"] = errors[:50]
        job["results"] = results

        # Output file reference
        job["outputFile"] = {
            "url": f"/api/v1/predictions/batch/{job_id}/download",
            "name": f"predictions_{job_id}.csv",
        }

        # Summary — frontend uses both formats
        if results:
            predictions = [r["prediction"] for r in results]
            probabilities = [r["probability"] for r in results if r.get("probability") is not None]

            positive_count = sum(1 for p in predictions if p == POSITIVE_CLASS)
            negative_count = len(predictions) - positive_count
            avg_conf = round(np.mean(probabilities), 4) if probabilities else None
            high_conf = sum(1 for p in probabilities if abs(p - 0.5) > 0.3)

            # Format processing time
            if elapsed >= 60:
                proc_time = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            else:
                proc_time = f"{elapsed:.1f}s"

            job["summary"] = {
                # PredictionsDashboard format
                "predictions": {
                    POSITIVE_CLASS: positive_count,
                    NEGATIVE_CLASS: negative_count,
                },
                "averageConfidence": avg_conf,
                "highConfidencePredictions": high_conf,
                "lowConfidencePredictions": len(probabilities) - high_conf,
                # BatchPredictionTab aliases
                "approved": positive_count,
                "rejected": negative_count,
                "total": len(predictions),
                "avgConfidence": round(avg_conf * 100, 1) if avg_conf else None,
                "processingTime": proc_time,
            }

        # Add batch to history
        self._prediction_history.insert(0, {
            "id": job["jobId"],
            "type": "batch",
            "modelName": job["modelName"],
            "model": job["modelName"],
            "timestamp": job["startedAt"],
            "timestampLabel": job["startedAt"],
            "recordsProcessed": job["totalRecords"],
            "status": "completed",
            "prediction": None,
            "confidence": None,
            "predictedLabel": f"{job['totalRecords']} records processed",
            "predictedClass": "batch",
            "details": job.get("summary"),
        })

    def get_batch_status(self, job_id: str) -> Optional[Dict]:
        job = self._batch_jobs.get(job_id)
        if job:
            # Return without internal results list
            return {k: v for k, v in job.items() if k != "results"}
        return None

    def get_batch_results_csv(self, job_id: str) -> Optional[bytes]:
        job = self._batch_jobs.get(job_id)
        if not job or not job.get("results"):
            return None
        df = pd.DataFrame(job["results"])
        return df.to_csv(index=False).encode("utf-8")

    # ========================================================================
    # DEPLOYED MODELS
    # Response shape: accuracy as 0-1, includes outputSchema + endpoint
    # PredictionsDashboard.tsx: (selectedModel.accuracy * 100).toFixed(1)
    # ========================================================================

    def get_deployed_models(self) -> Dict[str, Any]:
        models = []
        output_schema = {
            "prediction": "binary",
            "classes": [NEGATIVE_CLASS, POSITIVE_CLASS],
            "includesProbability": True,
            "includesExplanation": True,
        }

        if not self._report_df.empty:
            for _, row in self._report_df.iterrows():
                algo = row.get("Algorithm", "Unknown")
                test_score = row.get("Test_Score", row.get("test_score", 0))
                # Ensure 0-1 scale
                accuracy = float(test_score) if test_score <= 1 else float(test_score) / 100.0
                is_best = (algo == self._report_df.iloc[0].get("Algorithm"))

                models.append({
                    "id": algo.lower().replace(" ", ""),
                    "name": f"Income Prediction — {algo}",
                    "algorithm": algo,
                    "version": "v1.0",
                    "deployedAt": "2025-02-07T00:00:00Z",
                    "accuracy": round(accuracy, 4),
                    "status": "active" if is_best else "available",
                    "endpoint": "/api/v1/predictions/predict",
                    "inputFeatures": self.get_input_feature_schema(),
                    "outputSchema": output_schema,
                })
        else:
            for name, model_data in self._all_models.items():
                score = 0
                if isinstance(model_data, dict):
                    score = model_data.get("test_score", 0)
                accuracy = float(score) if score <= 1 else float(score) / 100.0
                models.append({
                    "id": name.lower().replace(" ", ""),
                    "name": f"Income Prediction — {name}",
                    "algorithm": name,
                    "version": "v1.0",
                    "deployedAt": "2025-02-07T00:00:00Z",
                    "accuracy": round(accuracy, 4),
                    "status": "available",
                    "endpoint": "/api/v1/predictions/predict",
                    "inputFeatures": self.get_input_feature_schema(),
                    "outputSchema": output_schema,
                })

        models.sort(key=lambda x: x["accuracy"], reverse=True)
        if models:
            models[0]["status"] = "active"

        return {
            "models": models,
            "totalModels": len(models),
            "activeModel": models[0] if models else None,
        }

    # ========================================================================
    # HISTORY
    # Response shape matches MOCK_PREDICTION_HISTORY
    # ========================================================================

    def _add_to_history(self, result: Dict, raw_features: Dict):
        """Add a single prediction to history in the frontend-expected shape."""
        label = result.get("output", {}).get("prediction", "")
        display_label = result.get("output", {}).get("predictionLabel", label)
        prob = result.get("output", {}).get("probability", 0)
        confidence_val = result.get("confidence", prob)

        self._prediction_history.insert(0, {
            "id": result["predictionId"],
            "type": "single",
            "modelName": result.get("modelName", ""),
            "model": result.get("modelName", ""),
            "timestamp": result.get("timestamp", ""),
            "timestampLabel": result.get("timestamp", ""),
            "prediction": label,
            "confidence": round(confidence_val, 4),
            "status": "success",
            # HistoryTab detail modal fields
            "predictedLabel": display_label,
            "predictedClass": label,
            "inputs": raw_features,
            "details": {
                "probability": prob,
                "threshold": result.get("output", {}).get("threshold", 0.5),
                "explanation": result.get("explanation", {}),
            },
        })

    def get_prediction_history(
            self, page: int = 1, page_size: int = 20,
            type_filter: Optional[str] = None, model_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        history = self._prediction_history

        if type_filter and type_filter != "all":
            history = [h for h in history if h.get("type") == type_filter]
        if model_filter and model_filter != "all":
            history = [h for h in history if h.get("modelName") == model_filter]

        total = len(history)
        start = (page - 1) * page_size
        page_items = history[start:start + page_size]

        return {
            "predictions": page_items,
            "pagination": {
                "page": page,
                "pageSize": page_size,
                "totalItems": total,
                "totalPages": max(1, math.ceil(total / page_size)),
            }
        }

    # ========================================================================
    # MONITORING
    # Response shape matches MOCK_MONITORING_STATS
    # Key: stats is NESTED under "stats" key, confidenceDistribution is ARRAY
    # ========================================================================

    def _update_monitoring(self, result: Dict):
        m = self._monitoring
        m["total_predictions"] += 1
        m["total_latency_ms"] += result.get("metadata", {}).get("processingTimeMs", 0)

        pred = result.get("output", {}).get("prediction", "")
        if pred in m["class_counts"]:
            m["class_counts"][pred] += 1

        prob = result.get("output", {}).get("probability")
        if prob is not None:
            if prob < 0.2:
                m["confidence_buckets"]["0.0-0.2"] += 1
            elif prob < 0.4:
                m["confidence_buckets"]["0.2-0.4"] += 1
            elif prob < 0.6:
                m["confidence_buckets"]["0.4-0.6"] += 1
            elif prob < 0.8:
                m["confidence_buckets"]["0.6-0.8"] += 1
            else:
                m["confidence_buckets"]["0.8-1.0"] += 1

        hour = datetime.utcnow().hour
        m["hourly_counts"][hour] = m["hourly_counts"].get(hour, 0) + 1
        if prob is not None:
            m["hourly_confidence_sums"][hour] = (
                    m["hourly_confidence_sums"].get(hour, 0) + prob
            )

    def get_monitoring_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns monitoring stats in the exact shape the frontend Monitoring tab expects.
        Key: confidenceDistribution is an ARRAY of {range, count}, not a dict.
        """
        m = self._monitoring
        total = m["total_predictions"]

        avg_latency = round(m["total_latency_ms"] / total, 1) if total > 0 else 0
        total_with_errors = total + m["errors"]
        error_rate = round(m["errors"] / total_with_errors, 4) if total_with_errors > 0 else 0
        active_hours = max(len(m["hourly_counts"]), 1)
        throughput = round(total / active_hours, 0)

        # Resolve model name
        model_name = None
        if model_id:
            try:
                _, model_name = self._resolve_model(model_id)
            except Exception:
                model_name = model_id

        # Hourly trend — array of { hour, predictions, avgConfidence }
        hourly_trend = []
        for h in range(24):
            count = m["hourly_counts"].get(h, 0)
            conf_sum = m["hourly_confidence_sums"].get(h, 0)
            avg_conf = round(conf_sum / count, 3) if count > 0 else 0.0
            hourly_trend.append({
                "hour": h,
                "predictions": count,
                "avgConfidence": avg_conf,
            })

        # Confidence distribution — ARRAY of { range, count }
        confidence_distribution = [
            {"range": bucket_range, "count": count}
            for bucket_range, count in m["confidence_buckets"].items()
        ]

        # Prediction distribution — { ">50K": N, "<=50K": N }
        prediction_distribution = dict(m["class_counts"])

        # Alerts
        alerts = self._generate_alerts()

        return {
            "modelId": model_id,
            "modelName": model_name,
            "timeRange": "last_24_hours",

            # NESTED under "stats" — this is what the frontend reads
            "stats": {
                "totalPredictions": total,
                "averageLatencyMs": avg_latency,
                "errorRate": error_rate,
                "throughput": throughput,
            },

            "predictionDistribution": prediction_distribution,
            "confidenceDistribution": confidence_distribution,
            "hourlyTrend": hourly_trend,
            "alerts": alerts,
        }

    def _generate_alerts(self) -> List[Dict]:
        alerts = []
        m = self._monitoring
        total = m["total_predictions"] + m["errors"]

        if total > 10 and m["errors"] / total > 0.05:
            alerts.append({
                "id": f"alert-err-{uuid.uuid4().hex[:4]}",
                "severity": "warning",
                "message": f"Error rate above 5%: {m['errors']}/{total} failed predictions",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

        low_conf = m["confidence_buckets"]["0.4-0.6"]
        if m["total_predictions"] > 10 and low_conf / m["total_predictions"] > 0.3:
            alerts.append({
                "id": f"alert-conf-{uuid.uuid4().hex[:4]}",
                "severity": "warning",
                "message": f"High proportion of borderline predictions ({low_conf} in 0.4-0.6 range)",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

        return alerts

    # ========================================================================
    # CSV TEMPLATE
    # ========================================================================

    def get_csv_template(self) -> str:
        return (
            "age,workclass,education,education.num,marital.status,"
            "occupation,relationship,race,sex,capital.gain,"
            "capital.loss,hours.per.week,native.country\n"
            "35,Private,Bachelors,13,Married-civ-spouse,"
            "Exec-managerial,Husband,White,Male,5000,"
            "0,45,United-States\n"
            "28,Private,HS-grad,9,Never-married,"
            "Other-service,Not-in-family,Black,Female,0,"
            "0,35,United-States\n"
        )


# ============================================================================
# SINGLETON
# ============================================================================

_service_instance: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance