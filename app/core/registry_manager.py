"""
Model Registry Manager
========================
Business logic for registering, listing, and managing models.

Reads trained model artifacts from Kedro output directories:
- data/06_models/best_model.pkl
- data/07_model_output/phase4_summary.json
- data/07_model_output/phase4_ranked_report.csv
- data/03_primary/scalers.pkl
- data/08_reporting/problem_type.txt

Follows same patterns as JobManager (SessionLocal, try/except/finally).
"""

import json
import csv
import os
import uuid
import pickle
import logging
from io import StringIO
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import desc, or_
from sqlalchemy.orm import Session, joinedload

from app.core.database import SessionLocal
from app.models.models import RegisteredModel, ModelVersion, ModelArtifact

logger = logging.getLogger(__name__)

# Kedro project path
KEDRO_PROJECT_PATH = Path(os.getenv(
    'KEDRO_PROJECT_PATH',
    '/home/ashok/work/latest/full/kedro-ml-engine-integrated'
))


class RegistryManager:
    """Manage Model Registry operations using SQLAlchemy"""

    def __init__(self):
        logger.info("✅ RegistryManager initialized")

    # ========================================================================
    # HELPERS - Read training output files
    # ========================================================================

    def _read_json_file(self, path: Path) -> Optional[dict]:
        """Read a JSON file safely"""
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read JSON {path}: {e}")
        return None

    def _read_pkl_file(self, path: Path) -> Optional[Any]:
        """Read a pickle file safely"""
        try:
            if path.exists():
                with open(path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not read pickle {path}: {e}")
        return None

    def _read_csv_as_dicts(self, path: Path) -> Optional[List[dict]]:
        """Read CSV file as list of dicts"""
        try:
            if path.exists():
                raw = path.read_text(encoding="utf-8")
                reader = csv.DictReader(StringIO(raw))
                return list(reader)
        except Exception as e:
            logger.warning(f"Could not read CSV {path}: {e}")
        return None

    def _read_text_file(self, path: Path) -> Optional[str]:
        """Read text file safely"""
        try:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Could not read text {path}: {e}")
        return None

    def _get_file_size(self, path: Path) -> Optional[int]:
        """Get file size in bytes"""
        try:
            if path.exists():
                return path.stat().st_size
        except Exception:
            pass
        return None

    def _to_float(self, val, default=None) -> Optional[float]:
        """Safe float conversion"""
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # ========================================================================
    # AUTO-DETECT from Kedro output files
    # ========================================================================

    def auto_detect_training_results(self) -> Dict[str, Any]:
        """
        Read training results from Kedro output directories.
        Returns a dict with all detected info.

        Reads from:
        - data/07_model_output/phase4_summary.json
        - data/07_model_output/phase4_ranked_report.csv
        - data/06_models/model_evaluation_classification.json
        - data/08_reporting/problem_type.txt
        - data/06_models/best_model.pkl (size only)
        - data/03_primary/scalers.pkl (size only)
        """
        base = KEDRO_PROJECT_PATH

        result = {
            "problem_type": None,
            "best_algorithm": None,
            "best_accuracy": None,
            "best_train_score": None,
            "best_test_score": None,
            "all_algorithms": [],
            "evaluation_metrics": {},
            "model_file_path": None,
            "model_file_size": None,
            "scaler_file_path": None,
            "artifacts": [],
        }

        # 1. Problem type
        problem_type = self._read_text_file(base / "data/08_reporting/problem_type.txt")
        if problem_type:
            result["problem_type"] = problem_type
            logger.info(f"  Detected problem_type: {problem_type}")

        # 2. Phase4 summary (best model info)
        summary = self._read_json_file(base / "data/07_model_output/phase4_summary.json")
        if summary:
            result["best_algorithm"] = summary.get("best_model")
            result["best_accuracy"] = self._to_float(summary.get("best_score"))
            logger.info(f"  Detected best_algorithm: {result['best_algorithm']}")
            logger.info(f"  Detected best_accuracy: {result['best_accuracy']}")

        # 3. Phase4 ranked report (all algorithm results)
        ranked = self._read_csv_as_dicts(base / "data/07_model_output/phase4_ranked_report.csv")
        if ranked:
            result["all_algorithms"] = ranked
            # Get best model's train/test scores
            best_algo = result["best_algorithm"]
            if best_algo:
                best_row = next((r for r in ranked if r.get("Algorithm") == best_algo), None)
                if best_row:
                    result["best_train_score"] = self._to_float(best_row.get("Train_Score"))
                    result["best_test_score"] = self._to_float(best_row.get("Test_Score"))

        # 4. Evaluation metrics
        eval_metrics = self._read_json_file(
            base / "data/06_models/model_evaluation_classification.json"
        )
        if eval_metrics:
            result["evaluation_metrics"] = eval_metrics

        # 5. Model file
        model_path = base / "data/06_models/best_model.pkl"
        if model_path.exists():
            result["model_file_path"] = str(model_path)
            result["model_file_size"] = self._get_file_size(model_path)
            result["artifacts"].append({
                "name": "best_model.pkl",
                "type": "model",
                "path": str(model_path),
                "size": result["model_file_size"]
            })

        # Also check classification-specific model
        cls_model_path = base / "data/06_models/best_model_classification.pkl"
        if cls_model_path.exists():
            result["artifacts"].append({
                "name": "best_model_classification.pkl",
                "type": "model",
                "path": str(cls_model_path),
                "size": self._get_file_size(cls_model_path)
            })

        # 6. Scaler
        scaler_path = base / "data/03_primary/scalers.pkl"
        if scaler_path.exists():
            result["scaler_file_path"] = str(scaler_path)
            result["artifacts"].append({
                "name": "scalers.pkl",
                "type": "scaler",
                "path": str(scaler_path),
                "size": self._get_file_size(scaler_path)
            })

        # 7. Reports as artifacts
        for report_file in [
            "phase4_summary.json",
            "phase4_report.csv",
            "phase4_ranked_report.csv",
            "phase4_results.csv",
            "phase3_predictions.csv",
        ]:
            report_path = base / "data/07_model_output" / report_file
            if report_path.exists():
                result["artifacts"].append({
                    "name": report_file,
                    "type": "report",
                    "path": str(report_path),
                    "size": self._get_file_size(report_path)
                })

        # 8. Evaluation JSON as artifact
        eval_path = base / "data/06_models/model_evaluation_classification.json"
        if eval_path.exists():
            result["artifacts"].append({
                "name": "model_evaluation_classification.json",
                "type": "report",
                "path": str(eval_path),
                "size": self._get_file_size(eval_path)
            })

        logger.info(f"  Total artifacts detected: {len(result['artifacts'])}")
        return result

    # ========================================================================
    # REGISTER MODEL
    # ========================================================================

    def register_model(
            self,
            project_id: str,
            name: str,
            description: Optional[str] = None,
            job_id: Optional[str] = None,
            algorithm: Optional[str] = None,
            problem_type: Optional[str] = None,
            dataset_id: Optional[str] = None,
            dataset_name: Optional[str] = None,
            accuracy: Optional[float] = None,
            train_score: Optional[float] = None,
            test_score: Optional[float] = None,
            precision: Optional[float] = None,
            recall: Optional[float] = None,
            f1_score: Optional[float] = None,
            tags: Optional[List[str]] = None,
            created_by: Optional[str] = None,
            collection_id: Optional[str] = None,
            dataset_path: Optional[str] = None,

    ) -> dict:
        """
        Register a new model in the registry.

        Auto-detects training results from Kedro output files if metrics
        are not explicitly provided.
        """
        model_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        db = None

        try:
            db = SessionLocal()

            # Auto-detect from training output
            logger.info(f"📊 Auto-detecting training results...")
            detected = self.auto_detect_training_results()

            # Use provided values or fall back to detected
            final_algorithm = algorithm or detected.get("best_algorithm")
            final_problem_type = problem_type or detected.get("problem_type")
            final_accuracy = accuracy if accuracy is not None else detected.get("best_accuracy")
            final_train_score = train_score if train_score is not None else detected.get("best_train_score")
            final_test_score = test_score if test_score is not None else detected.get("best_test_score")

            # Extract evaluation metrics
            eval_metrics = detected.get("evaluation_metrics", {})
            final_precision = precision if precision is not None else self._to_float(eval_metrics.get("precision"))
            final_recall = recall if recall is not None else self._to_float(eval_metrics.get("recall"))
            final_f1 = f1_score if f1_score is not None else self._to_float(eval_metrics.get("f1_score"))
            final_roc_auc = self._to_float(eval_metrics.get("roc_auc"))

            # Model file size in MB
            model_size_mb = None
            if detected.get("model_file_size"):
                model_size_mb = round(detected["model_file_size"] / (1024 * 1024), 2)

            logger.info(f"  Algorithm: {final_algorithm}")
            logger.info(f"  Problem type: {final_problem_type}")
            logger.info(f"  Accuracy: {final_accuracy}")

            # --- Check if model with same name exists for this project ---
            existing = db.query(RegisteredModel).filter(
                RegisteredModel.project_id == project_id,
                RegisteredModel.name == name
            ).first()

            if existing:
                # Add a new VERSION to existing model
                logger.info(f"  Model '{name}' already exists, adding new version...")
                return self._add_version_to_existing(
                    db=db,
                    model=existing,
                    detected=detected,
                    algorithm=final_algorithm,
                    accuracy=final_accuracy,
                    train_score=final_train_score,
                    test_score=final_test_score,
                    precision_val=final_precision,
                    recall_val=final_recall,
                    f1_val=final_f1,
                    roc_auc=final_roc_auc,
                    model_size_mb=model_size_mb,
                    job_id=job_id,
                    created_by=created_by,
                    tags=tags,
                )

            # --- Create NEW registered model ---
            registered_model = RegisteredModel(
                id=model_id,
                project_id=project_id,
                name=name,
                description=description or f"{final_algorithm} model for {final_problem_type or 'ML'}",
                problem_type=final_problem_type,
                current_version="v1.0",
                latest_version="v1.0",
                total_versions=1,
                status="draft",
                best_accuracy=final_accuracy,
                best_algorithm=final_algorithm,
                source_dataset_id=dataset_id,
                source_dataset_name=dataset_name,
                training_job_id=job_id,
                tags=json.dumps(tags) if tags else None,
                created_by=created_by or "api_user",
                collection_id=collection_id,
                dataset_path=dataset_path,
            )
            db.add(registered_model)

            # --- Create first version (v1.0) ---
            version = ModelVersion(
                id=version_id,
                model_id=model_id,
                version="v1.0",
                version_number=1,
                is_current=True,
                status="draft",
                algorithm=final_algorithm,
                accuracy=final_accuracy,
                precision=final_precision,
                recall=final_recall,
                f1_score=final_f1,
                train_score=final_train_score,
                test_score=final_test_score,
                roc_auc=final_roc_auc,
                job_id=job_id,
                model_size_mb=model_size_mb,
                feature_names=json.dumps(detected.get("feature_names")) if detected.get("feature_names") else None,
                feature_importances=json.dumps(detected.get("feature_importances")) if detected.get("feature_importances") else None,
                confusion_matrix=json.dumps(eval_metrics.get("confusion_matrix")) if eval_metrics.get("confusion_matrix") else None,
                tags=json.dumps(tags) if tags else None,
                created_by=created_by or "api_user",
            )
            db.add(version)

            # --- Create artifacts ---
            for art in detected.get("artifacts", []):
                artifact = ModelArtifact(
                    id=str(uuid.uuid4()),
                    model_version_id=version_id,
                    artifact_name=art["name"],
                    artifact_type=art["type"],
                    file_path=art["path"],
                    file_size_bytes=art.get("size"),
                )
                db.add(artifact)

            db.commit()
            db.refresh(registered_model)

            logger.info(f"✅ Model registered: {model_id} (v1.0)")
            return self._model_to_dict(registered_model, include_versions=True)

        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"❌ Error registering model: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    def _add_version_to_existing(
            self, db: Session, model: RegisteredModel, detected: dict,
            algorithm, accuracy, train_score, test_score,
            precision_val, recall_val, f1_val, roc_auc,
            model_size_mb, job_id, created_by, tags
    ) -> dict:
        """Add a new version to an existing registered model"""

        new_version_number = model.total_versions + 1
        new_version_str = f"v{new_version_number}.0"
        version_id = str(uuid.uuid4())

        # Unmark old current version
        db.query(ModelVersion).filter(
            ModelVersion.model_id == model.id,
            ModelVersion.is_current == True
        ).update({"is_current": False})

        # Create new version
        version = ModelVersion(
            id=version_id,
            model_id=model.id,
            version=new_version_str,
            version_number=new_version_number,
            is_current=True,
            status="draft",
            algorithm=algorithm,
            accuracy=accuracy,
            precision=precision_val,
            recall=recall_val,
            f1_score=f1_val,
            train_score=train_score,
            test_score=test_score,
            roc_auc=roc_auc,
            job_id=job_id,
            model_size_mb=model_size_mb,
            tags=json.dumps(tags) if tags else None,
            created_by=created_by or "api_user",
        )
        db.add(version)

        # Create artifacts for new version
        for art in detected.get("artifacts", []):
            artifact = ModelArtifact(
                id=str(uuid.uuid4()),
                model_version_id=version_id,
                artifact_name=art["name"],
                artifact_type=art["type"],
                file_path=art["path"],
                file_size_bytes=art.get("size"),
            )
            db.add(artifact)

        # Update parent model
        model.total_versions = new_version_number
        model.latest_version = new_version_str
        model.current_version = new_version_str
        model.updated_at = datetime.utcnow()

        # Update best metrics if this version is better
        if accuracy is not None and (model.best_accuracy is None or accuracy > model.best_accuracy):
            model.best_accuracy = accuracy
            model.best_algorithm = algorithm

        db.commit()
        db.refresh(model)

        logger.info(f"✅ Version {new_version_str} added to model {model.id}")
        return self._model_to_dict(model, include_versions=True)

    # ========================================================================
    # LIST MODELS
    # ========================================================================

    def list_models(
            self,
            project_id: str,
            status: Optional[str] = None,
            search: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
    ) -> Dict[str, Any]:
        """List registered models for a project with filtering"""
        db = None
        try:
            db = SessionLocal()

            query = db.query(RegisteredModel).filter(
                RegisteredModel.project_id == project_id
            )

            # Filter by status
            if status and status != "all":
                query = query.filter(RegisteredModel.status == status)

            # Search in name, description, algorithm
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        RegisteredModel.name.ilike(search_term),
                        RegisteredModel.description.ilike(search_term),
                        RegisteredModel.best_algorithm.ilike(search_term),
                    )
                )

            # Get total count before pagination
            total = query.count()

            # Paginate and order
            models = query.order_by(
                desc(RegisteredModel.updated_at)
            ).offset(offset).limit(limit).all()

            return {
                "models": [self._model_to_dict(m, include_versions=False) for m in models],
                "total": total,
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            logger.error(f"❌ Error listing models: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # GET MODEL DETAIL
    # ========================================================================

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get full model details with versions and artifacts"""
        db = None
        try:
            db = SessionLocal()

            model = db.query(RegisteredModel).options(
                joinedload(RegisteredModel.versions).joinedload(ModelVersion.artifacts)
            ).filter(
                RegisteredModel.id == model_id
            ).first()

            if not model:
                return None

            return self._model_to_dict(model, include_versions=True)

        except Exception as e:
            logger.error(f"❌ Error getting model: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # GET MODEL VERSIONS
    # ========================================================================

    def get_model_versions(self, model_id: str) -> List[dict]:
        """Get all versions for a model"""
        db = None
        try:
            db = SessionLocal()

            versions = db.query(ModelVersion).options(
                joinedload(ModelVersion.artifacts)
            ).filter(
                ModelVersion.model_id == model_id
            ).order_by(
                desc(ModelVersion.version_number)
            ).all()

            return [self._version_to_dict(v) for v in versions]

        except Exception as e:
            logger.error(f"❌ Error getting versions: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # GET STATS
    # ========================================================================

    def get_stats(self, project_id: str) -> dict:
        """Get summary statistics for registry dashboard cards"""
        db = None
        try:
            db = SessionLocal()

            base_query = db.query(RegisteredModel).filter(
                RegisteredModel.project_id == project_id
            )

            total = base_query.count()
            deployed = base_query.filter(RegisteredModel.is_deployed == True).count()
            production = base_query.filter(RegisteredModel.status == "production").count()
            staging = base_query.filter(RegisteredModel.status == "staging").count()
            draft = base_query.filter(RegisteredModel.status == "draft").count()
            archived = base_query.filter(RegisteredModel.status == "archived").count()

            return {
                "total_models": total,
                "deployed": deployed,
                "production": production,
                "staging": staging,
                "draft": draft,
                "archived": archived,
            }

        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # UPDATE MODEL
    # ========================================================================

    def update_model(self, model_id: str, updates: dict) -> Optional[dict]:
        """Update model metadata (name, description, tags, labels)"""
        db = None
        try:
            db = SessionLocal()
            model = db.query(RegisteredModel).filter(RegisteredModel.id == model_id).first()

            if not model:
                return None

            if "name" in updates and updates["name"]:
                model.name = updates["name"]
            if "description" in updates and updates["description"] is not None:
                model.description = updates["description"]
            if "tags" in updates:
                model.tags = json.dumps(updates["tags"]) if updates["tags"] else None
            if "labels" in updates:
                model.labels = json.dumps(updates["labels"]) if updates["labels"] else None

            model.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(model)

            logger.info(f"✅ Model updated: {model_id}")
            return self._model_to_dict(model)

        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"❌ Error updating model: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # PROMOTE / DEPLOY / ARCHIVE
    # ========================================================================

    def promote_model(self, model_id: str, target_status: str, version: Optional[str] = None) -> Optional[dict]:
        """Promote model to staging/production"""
        db = None
        try:
            db = SessionLocal()
            model = db.query(RegisteredModel).filter(RegisteredModel.id == model_id).first()

            if not model:
                return None

            model.status = target_status
            model.updated_at = datetime.utcnow()

            # If promoting to production, also update the version status
            if version:
                ver = db.query(ModelVersion).filter(
                    ModelVersion.model_id == model_id,
                    ModelVersion.version == version
                ).first()
                if ver:
                    ver.status = target_status
            else:
                # Promote current version
                ver = db.query(ModelVersion).filter(
                    ModelVersion.model_id == model_id,
                    ModelVersion.is_current == True
                ).first()
                if ver:
                    ver.status = target_status

            db.commit()
            db.refresh(model)

            logger.info(f"✅ Model {model_id} promoted to {target_status}")
            return self._model_to_dict(model)

        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"❌ Error promoting model: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    def deploy_model(self, model_id: str, version: Optional[str] = None, environment: str = "production") -> Optional[dict]:
        """Mark model as deployed"""
        db = None
        try:
            db = SessionLocal()
            model = db.query(RegisteredModel).filter(RegisteredModel.id == model_id).first()

            if not model:
                return None

            deploy_version = version or model.current_version
            model.is_deployed = True
            model.deployed_version = deploy_version
            model.deployed_at = datetime.utcnow()
            model.status = environment  # production or staging
            model.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(model)

            logger.info(f"✅ Model {model_id} deployed (version {deploy_version})")
            return self._model_to_dict(model)

        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"❌ Error deploying model: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    def archive_model(self, model_id: str) -> Optional[dict]:
        """Archive a model"""
        db = None
        try:
            db = SessionLocal()
            model = db.query(RegisteredModel).filter(RegisteredModel.id == model_id).first()

            if not model:
                return None

            model.status = "archived"
            model.is_deployed = False
            model.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(model)

            logger.info(f"✅ Model {model_id} archived")
            return self._model_to_dict(model)

        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"❌ Error archiving model: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # DELETE MODEL
    # ========================================================================

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions/artifacts"""
        db = None
        try:
            db = SessionLocal()
            model = db.query(RegisteredModel).filter(RegisteredModel.id == model_id).first()

            if not model:
                return False

            db.delete(model)  # Cascading delete handles versions and artifacts
            db.commit()

            logger.info(f"✅ Model {model_id} deleted")
            return True

        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"❌ Error deleting model: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # GET ARTIFACTS
    # ========================================================================

    def get_artifacts(self, model_id: str, version: Optional[str] = None) -> List[dict]:
        """Get artifacts for a model (optionally filtered by version)"""
        db = None
        try:
            db = SessionLocal()

            query = db.query(ModelArtifact).join(ModelVersion).filter(
                ModelVersion.model_id == model_id
            )

            if version:
                query = query.filter(ModelVersion.version == version)
            else:
                # Default: current version
                query = query.filter(ModelVersion.is_current == True)

            artifacts = query.all()
            return [self._artifact_to_dict(a) for a in artifacts]

        except Exception as e:
            logger.error(f"❌ Error getting artifacts: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    def get_artifact_path(self, artifact_id: str) -> Optional[str]:
        """Get file path for an artifact (for download)"""
        db = None
        try:
            db = SessionLocal()
            artifact = db.query(ModelArtifact).filter(ModelArtifact.id == artifact_id).first()
            if artifact and os.path.exists(artifact.file_path):
                return artifact.file_path
            return None
        except Exception as e:
            logger.error(f"❌ Error getting artifact path: {e}", exc_info=True)
            raise
        finally:
            if db:
                db.close()

    # ========================================================================
    # SERIALIZATION HELPERS
    # ========================================================================

    def _parse_json_text(self, text: Optional[str]) -> Any:
        """Parse JSON text column, return None if invalid"""
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    def _format_dt(self, dt) -> Optional[str]:
        """Format datetime to ISO string"""
        if dt and hasattr(dt, 'isoformat'):
            return dt.isoformat()
        return str(dt) if dt else None

    def _model_to_dict(self, model: RegisteredModel, include_versions: bool = False) -> dict:
        """Convert RegisteredModel ORM to dict"""
        result = {
            "id": model.id,
            "project_id": model.project_id,
            "name": model.name,
            "description": model.description,
            "problem_type": model.problem_type,
            "current_version": model.current_version,
            "latest_version": model.latest_version,
            "total_versions": model.total_versions or 1,
            "status": model.status,
            "best_accuracy": model.best_accuracy,
            "best_algorithm": model.best_algorithm,
            "is_deployed": model.is_deployed or False,
            "deployment_url": model.deployment_url,
            "deployed_version": model.deployed_version,
            "deployed_at": self._format_dt(model.deployed_at),
            "source_dataset_id": model.source_dataset_id,
            "source_dataset_name": model.source_dataset_name,
            "training_job_id": model.training_job_id,
            "tags": self._parse_json_text(model.tags),
            "created_by": model.created_by,
            "created_at": self._format_dt(model.created_at),
            "updated_at": self._format_dt(model.updated_at),
            "collection_id": self._format_dt(model.collection_id),
            "dataset_path": self._format_dt(model.dataset_path),
        }

        if include_versions and hasattr(model, 'versions') and model.versions:
            result["versions"] = [self._version_to_dict(v) for v in model.versions]
        else:
            result["versions"] = []

        return result

    def _version_to_dict(self, version: ModelVersion) -> dict:
        """Convert ModelVersion ORM to dict"""
        result = {
            "id": version.id,
            "version": version.version,
            "version_number": version.version_number,
            "status": version.status,
            "algorithm": version.algorithm,
            "accuracy": version.accuracy,
            "precision": version.precision,
            "recall": version.recall,
            "f1_score": version.f1_score,
            "train_score": version.train_score,
            "test_score": version.test_score,
            "roc_auc": version.roc_auc,
            "is_current": version.is_current or False,
            "job_id": version.job_id,
            "model_size_mb": version.model_size_mb,
            "training_time_seconds": version.training_time_seconds,
            "created_by": version.created_by,
            "created_at": self._format_dt(version.created_at),
            "tags": self._parse_json_text(version.tags),
            "hyperparameters": self._parse_json_text(version.hyperparameters),
            "feature_names": self._parse_json_text(version.feature_names),
            "feature_importances": self._parse_json_text(version.feature_importances),
            "confusion_matrix": self._parse_json_text(version.confusion_matrix),
            "training_config": self._parse_json_text(version.training_config),
        }

        # Include artifacts if loaded
        if hasattr(version, 'artifacts') and version.artifacts:
            result["artifacts"] = [self._artifact_to_dict(a) for a in version.artifacts]
        else:
            result["artifacts"] = []

        return result

    def _artifact_to_dict(self, artifact: ModelArtifact) -> dict:
        """Convert ModelArtifact ORM to dict"""
        return {
            "id": artifact.id,
            "artifact_name": artifact.artifact_name,
            "artifact_type": artifact.artifact_type,
            "file_path": artifact.file_path,
            "file_size_bytes": artifact.file_size_bytes,
            "created_at": self._format_dt(artifact.created_at),
        }