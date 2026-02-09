"""
Multi-Table Dataset Collection Schemas
=======================================
Pydantic models for request validation and response serialization.

Organized by wizard step:
  Step 1: Collection + File Upload
  Step 2: Primary Table + Target Column
  Step 3: Relationships
  Step 4: Aggregations
  Step 5: Review + Process
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class CollectionStatus(str, Enum):
    DRAFT = "draft"
    CONFIGURED = "configured"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class TableRole(str, Enum):
    PRIMARY = "primary"
    RELATED = "related"


class JoinType(str, Enum):
    LEFT = "left"
    INNER = "inner"
    RIGHT = "right"
    FULL = "full"


class RelationshipType(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class AggFunction(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    COUNT = "count"
    UNIQUE_COUNT = "unique_count"
    NUNIQUE = "nunique"            # Frontend alias for unique_count
    STD = "std"
    VARIANCE = "variance"
    VAR = "var"                    # Frontend alias for variance
    FIRST = "first"
    LAST = "last"
    MODE = "mode"


# ============================================================================
# COLUMN METADATA (shared across multiple responses)
# ============================================================================

class ColumnInfo(BaseModel):
    """Column metadata auto-detected from CSV"""
    name: str
    dtype: str                                  # pandas dtype: "int64", "float64", "object"
    display_type: str                           # UI-friendly: "INTEGER", "FLOAT", "VARCHAR", "DATE", "BOOLEAN"
    nullable: bool = True
    unique_count: Optional[int] = None
    null_count: Optional[int] = None
    null_percentage: Optional[float] = None
    sample_values: Optional[List[Any]] = None   # 3-5 sample values for preview
    is_potential_key: bool = False               # True if unique_count == row_count


# ============================================================================
# STEP 1: CREATE COLLECTION + UPLOAD FILES
# ============================================================================

class CreateCollectionRequest(BaseModel):
    """Sent as form fields alongside file uploads"""
    name: str = Field(..., min_length=1, max_length=255, description="Collection display name")
    description: Optional[str] = Field(None, max_length=2000)
    project_id: str = Field(..., description="Project UUID")


class TableSummary(BaseModel):
    """Lightweight table info for list views"""
    id: str
    table_name: str
    file_name: Optional[str] = None
    role: TableRole = TableRole.RELATED
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    dataset_id: Optional[str] = None


class TableDetail(TableSummary):
    """Full table info including column metadata"""
    columns: List[ColumnInfo] = []
    file_path: Optional[str] = None
    created_at: Optional[str] = None


class CollectionSummary(BaseModel):
    """Lightweight collection for list views"""
    id: str
    name: str
    description: Optional[str] = None
    project_id: str
    status: CollectionStatus
    current_step: int = 1
    total_tables: int = 0
    total_relationships: int = 0
    total_aggregations: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CollectionCreateResponse(BaseModel):
    """Response after Step 1 — collection created with uploaded files"""
    id: str
    name: str
    description: Optional[str] = None
    project_id: str
    status: CollectionStatus = CollectionStatus.DRAFT
    total_tables: int
    tables: List[TableSummary]
    created_at: str


# ============================================================================
# STEP 2: IDENTIFY PRIMARY TABLE
# ============================================================================

class SetPrimaryTableRequest(BaseModel):
    """Set which table is primary and optionally the target column"""
    primary_table_id: str = Field(..., description="CollectionTable ID to mark as primary")
    target_column: Optional[str] = Field(None, description="Target column for supervised learning")

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class PrimaryTableResponse(BaseModel):
    """Response confirming primary table selection"""
    collection_id: str
    primary_table_id: str
    primary_table_name: str
    target_column: Optional[str] = None
    tables: List[TableSummary]


# ============================================================================
# STEP 3: RELATIONSHIPS
# ============================================================================

class CreateRelationshipRequest(BaseModel):
    """Create a join between two tables"""
    left_table_id: str = Field(..., description="CollectionTable ID (usually primary)")
    right_table_id: str = Field(..., description="CollectionTable ID (detail table)")
    left_column: str = Field(..., description="Join key column in left table")
    right_column: str = Field(..., description="Join key column in right table")
    join_type: JoinType = Field(JoinType.LEFT, description="SQL join type")

    @field_validator("left_table_id")
    @classmethod
    def tables_must_differ(cls, v, info):
        if info.data.get("right_table_id") and v == info.data["right_table_id"]:
            raise ValueError("Cannot create self-join: left and right tables must be different")
        return v


class UpdateRelationshipRequest(BaseModel):
    """Update an existing relationship"""
    left_column: Optional[str] = None
    right_column: Optional[str] = None
    join_type: Optional[JoinType] = None


class RelationshipValidation(BaseModel):
    """Join quality metrics"""
    is_validated: bool = False
    relationship_type: Optional[RelationshipType] = None
    left_unique_count: Optional[int] = None
    right_unique_count: Optional[int] = None
    match_count: Optional[int] = None
    match_percentage: Optional[float] = None
    orphan_left_count: Optional[int] = None
    orphan_right_count: Optional[int] = None
    warnings: List[str] = []


class RelationshipResponse(BaseModel):
    """Full relationship detail"""
    id: str
    collection_id: str
    left_table_id: str
    right_table_id: str
    left_table_name: Optional[str] = None
    right_table_name: Optional[str] = None
    left_column: str
    right_column: str
    left_column_dtype: Optional[str] = None
    right_column_dtype: Optional[str] = None
    join_type: JoinType
    validation: Optional[RelationshipValidation] = None
    preview_sql: Optional[str] = None  # e.g. "application_train.SK_ID_CURR = bureau.SK_ID_CURR"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SuggestedRelationship(BaseModel):
    """Auto-suggested join based on column name matching"""
    left_table_id: str
    left_table_name: str
    right_table_id: str
    right_table_name: str
    left_column: str
    right_column: str
    confidence: float = Field(..., ge=0, le=1, description="0.0-1.0 confidence score")
    reason: str  # "Exact column name match", "Primary key detected", etc.


# ============================================================================
# STEP 4: AGGREGATIONS
# ============================================================================

class AggregationFeature(BaseModel):
    """One column + its aggregation functions"""
    column: str = Field(..., description="Column name to aggregate")
    functions: List[AggFunction] = Field(..., min_length=1, description="Aggregation functions to apply")


class CreateAggregationRequest(BaseModel):
    """Configure aggregation for a related table"""
    source_table_id: str = Field(..., description="CollectionTable ID of the detail table")
    group_by_column: str = Field(..., description="Column to group by (usually the join key)")
    column_prefix: str = Field(..., min_length=1, max_length=50, description="Prefix for output columns")
    features: List[AggregationFeature] = Field(..., min_length=1, description="Columns and their aggregation functions")

    @field_validator("column_prefix")
    @classmethod
    def sanitize_prefix(cls, v):
        # Ensure prefix ends with underscore
        v = v.strip().upper()
        if not v.endswith("_"):
            v += "_"
        return v


class UpdateAggregationRequest(BaseModel):
    """Update an existing aggregation"""
    group_by_column: Optional[str] = None
    column_prefix: Optional[str] = None
    features: Optional[List[AggregationFeature]] = None

    @field_validator("column_prefix")
    @classmethod
    def sanitize_prefix(cls, v):
        if v is not None:
            v = v.strip().upper()
            if not v.endswith("_"):
                v += "_"
        return v


class AggregationResponse(BaseModel):
    """Full aggregation detail"""
    id: str
    collection_id: str
    source_table_id: str
    source_table_name: Optional[str] = None
    group_by_column: str
    column_prefix: str
    features: List[AggregationFeature]
    created_columns: List[str] = []  # ["BUREAU_amount_sum", "BUREAU_amount_mean", ...]
    output_column_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ============================================================================
# STEP 5: REVIEW + PROCESS
# ============================================================================

class ReviewSummary(BaseModel):
    """Complete wizard review before processing"""
    collection: CollectionSummary
    primary_table: Optional[TableDetail] = None
    target_column: Optional[str] = None
    tables: List[TableSummary]
    relationships: List[RelationshipResponse]
    aggregations: List[AggregationResponse]

    # Computed summary
    total_input_rows: int = 0
    total_input_columns: int = 0
    estimated_output_columns: int = 0
    warnings: List[str] = []          # "bureau table has no aggregation configured"
    ready_to_process: bool = False


class ProcessRequest(BaseModel):
    """Optional parameters for processing"""
    output_name: Optional[str] = None          # Custom name for merged dataset; defaults to collection name
    save_intermediate: bool = False             # Save each aggregated table separately
    sample_rows: Optional[int] = None           # Process only N rows (for testing)
    drop_duplicates: bool = False
    handle_missing: Optional[Literal["keep", "drop_rows", "fill_zero"]] = "keep"


class ProcessStatusResponse(BaseModel):
    """Processing job status"""
    collection_id: str
    status: CollectionStatus
    processing_started_at: Optional[str] = None
    processing_completed_at: Optional[str] = None
    processing_duration_seconds: Optional[float] = None
    processing_error: Optional[str] = None

    # Output info (only if processed)
    merged_dataset_id: Optional[str] = None
    merged_file_path: Optional[str] = None
    rows_after_merge: Optional[int] = None
    columns_after_merge: Optional[int] = None


class MergedPreviewResponse(BaseModel):
    """Preview of merged dataset"""
    collection_id: str
    total_rows: int
    total_columns: int
    preview_rows: int
    columns: List[ColumnInfo]
    rows: List[Dict[str, Any]]


# ============================================================================
# FULL COLLECTION DETAIL (used by GET /collections/{id})
# ============================================================================

class CollectionDetail(BaseModel):
    """Complete collection with all nested data"""
    id: str
    name: str
    description: Optional[str] = None
    project_id: str
    status: CollectionStatus
    current_step: int

    # Step 2
    primary_table_id: Optional[str] = None
    target_column: Optional[str] = None

    # All nested data
    tables: List[TableDetail] = []
    relationships: List[RelationshipResponse] = []
    aggregations: List[AggregationResponse] = []

    # Processing output
    merged_dataset_id: Optional[str] = None
    merged_file_path: Optional[str] = None
    rows_before_merge: Optional[int] = None
    rows_after_merge: Optional[int] = None
    columns_after_merge: Optional[int] = None
    processing_duration_seconds: Optional[float] = None

    # Counts
    total_tables: int = 0
    total_relationships: int = 0
    total_aggregations: int = 0

    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ============================================================================
# UPDATE COLLECTION METADATA
# ============================================================================

class UpdateCollectionRequest(BaseModel):
    """Update collection name/description"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    current_step: Optional[int] = Field(None, ge=1, le=5)