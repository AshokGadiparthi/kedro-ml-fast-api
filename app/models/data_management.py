"""
Data Management Models - Phase 3 Extended
===========================================
Datasources and Datasets for data ingestion and management

Models:
- Datasource: Connection to any data source (DB, file, API, etc)
- Dataset: Uploaded or imported data with schema & quality metrics
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text, JSON, LargeBinary, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


# ============================================================================
# DATASOURCE MODEL
# ============================================================================

class Datasource(Base):
    """
    Represents a data source connection
    
    Supports multiple types:
    - Database: postgresql, mysql, sqlite, mongodb, etc
    - Cloud: s3, azure, gcs, snowflake, bigquery
    - API: rest, graphql, etc
    - Files: csv, parquet, json, etc
    """
    
    __tablename__ = "datasources"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relationship
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Basic Info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    type = Column(String(50), nullable=False)  # postgresql, mysql, s3, api, csv, etc
    
    # Connection Configuration (Encrypted)
    connection_config = Column(JSON, nullable=False)  # {host, port, username, password_encrypted, db, etc}
    is_encrypted = Column(Boolean, default=True)
    
    # Connection Status
    status = Column(String(50), default="disconnected")  # connected, disconnected, testing, error
    last_tested_at = Column(DateTime, nullable=True)
    test_result = Column(JSON, nullable=True)  # {status, message, latency_ms, error, etc}
    
    # Metadata
    tags = Column(JSON, nullable=True)  # ["production", "critical", etc]
    owner = Column(String(255), nullable=True)
    documentation_url = Column(String(500), nullable=True)
    sla = Column(String(50), nullable=True)  # "99.9%", "99.99%", etc
    
    # Statistics
    tables_count = Column(Integer, nullable=True)  # Number of tables accessible
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)  # Number of times accessed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="datasource", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Datasource(id={self.id}, name={self.name}, type={self.type}, status={self.status})>"


# ============================================================================
# DATASET MODEL
# ============================================================================

class Dataset(Base):
    """
    Represents a dataset (uploaded file or imported from datasource)
    
    Features:
    - Store file content as blob in database
    - Auto-detect schema (column types, names)
    - Track data quality metrics
    - Version control
    - Kedro catalog integration
    """
    
    __tablename__ = "datasets"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relationships
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    datasource_id = Column(String(36), ForeignKey("datasources.id", ondelete="SET NULL"), nullable=True)
    
    # Basic Info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Source Information
    source_type = Column(String(50), nullable=False)  # "upload" or "datasource"
    
    # File Information (for uploads)
    file_name = Column(String(255), nullable=True)
    file_format = Column(String(50), nullable=True)  # csv, json, parquet, excel, etc
    file_size_bytes = Column(Integer, nullable=True)
    file_content = Column(LargeBinary, nullable=True)  # ← Store file as blob in database
    
    # Data Information
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    
    # Schema Information (auto-detected)
    schema = Column(JSON, nullable=True)  # [{name, type, nullable, description}, ...]
    schema_inferred = Column(Boolean, default=False)
    schema_confidence = Column(Float, nullable=True)  # 0-100, how confident we are
    
    # Data Quality Metrics
    quality_score = Column(Float, nullable=True)  # 0-100 (completeness, validity, etc)
    missing_values_count = Column(Integer, nullable=True)
    missing_values_pct = Column(Float, nullable=True)
    duplicates_count = Column(Integer, nullable=True)
    anomalies_count = Column(Integer, nullable=True)
    
    # Quality Details
    quality_report = Column(JSON, nullable=True)  # {
    #   completeness: 99.5,
    #   validity: 98.2,
    #   uniqueness: 100.0,
    #   anomalies: [{type, count, severity}, ...],
    #   warnings: [...],
    #   errors: [...]
    # }
    
    # Versioning
    version = Column(Integer, default=1)
    is_latest = Column(Boolean, default=True)
    parent_version_id = Column(String(36), nullable=True)  # Link to previous version
    
    # Scheduling (for automatic refresh)
    refresh_schedule = Column(JSON, nullable=True)  # {frequency, day_of_month, time, etc}
    last_refreshed = Column(DateTime, nullable=True)
    
    # Kedro Integration
    catalog_name = Column(String(255), nullable=True)  # Name in Kedro catalog
    is_kedro_registered = Column(Boolean, default=False)
    kedro_type = Column(String(100), nullable=True)  # pandas.CSVDataset, sql.SQLTableDataset, etc
    
    # Metadata
    tags = Column(JSON, nullable=True)  # ["transactions", "monthly", etc]
    lineage_info = Column(JSON, nullable=True)  # {source_dataset_id, transformations, etc}
    
    # Status
    status = Column(String(50), default="processing")  # processing, ready, error, archived
    status_message = Column(Text, nullable=True)  # Error message if status=error
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    datasource = relationship("Datasource", back_populates="datasets")
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name={self.name}, format={self.file_format}, version={self.version})>"


# ============================================================================
# DATA PROFILE MODEL (for historical tracking)
# ============================================================================

class DataProfile(Base):
    """
    Track data profile history (optional, for analytics)
    Stores snapshots of data quality over time
    """
    
    __tablename__ = "data_profiles"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Profile Snapshot
    row_count = Column(Integer)
    quality_score = Column(Float)
    missing_values_count = Column(Integer)
    missing_values_pct = Column(Float)
    duplicates_count = Column(Integer)
    
    # Full Report
    full_report = Column(JSON)  # Complete profile data
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<DataProfile(dataset_id={self.dataset_id}, quality={self.quality_score})>"
