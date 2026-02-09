"""
EDA Schemas
INTEGRATED: Pydantic models for type safety and validation
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

# ============================================================================
# HEALTH CHECK
# ============================================================================

class ComponentStatus(BaseModel):
    """Status of a single component"""
    api: str
    cache: str
    database: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response - Simple version"""
    status: str
    service: Optional[str] = None
    version: Optional[str] = None
    timestamp: datetime

# ============================================================================
# JOB MANAGEMENT
# ============================================================================

class JobStartRequest(BaseModel):
    """Request to start EDA analysis"""
    dataset_id: str

class JobStartResponse(BaseModel):
    """Response when analysis is started"""
    job_id: str
    status: str
    dataset_id: str
    created_at: datetime
    estimated_time: str
    polling_endpoint: str

class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    dataset_id: str
    status: str  # queued, processing, completed, failed
    progress: int  # 0-100
    current_phase: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    result_id: Optional[str] = None
    error: Optional[str] = None

# ============================================================================
# DATA PROFILE RESULTS
# ============================================================================

class DataProfile(BaseModel):
    """Basic data profile summary"""
    rows: int
    columns: int
    memory_mb: float
    missing_values_percent: float
    duplicate_rows: int
    data_types: Dict[str, int]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: Optional[List[str]] = None
    generated_at: datetime

class MissingValueAnalysis(BaseModel):
    """Missing value analysis"""
    count: int
    percent: float
    by_column: Dict[str, int]

class DuplicateAnalysis(BaseModel):
    """Duplicate analysis"""
    count: int
    percent: float

# ============================================================================
# STATISTICS
# ============================================================================

class NumericalStats(BaseModel):
    """Statistics for a numerical column"""
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q1: float
    q3: float
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

class CategoricalStats(BaseModel):
    """Statistics for a categorical column"""
    count: int
    unique: int
    mode: Optional[str] = None
    mode_frequency: Optional[float] = None
    entropy: Optional[float] = None

class StatisticsResponse(BaseModel):
    """Full statistics response"""
    numerical: Optional[Dict[str, NumericalStats]] = None
    categorical: Optional[Dict[str, CategoricalStats]] = None
    generated_at: datetime

# ============================================================================
# QUALITY REPORT
# ============================================================================

class QualityCheck(BaseModel):
    """Single quality check result"""
    name: str
    status: str  # pass, warning, fail
    score: float  # 0-100
    details: str

class QualityReportResponse(BaseModel):
    """Quality assessment report"""
    overall_quality_score: float
    checks: List[QualityCheck]
    recommendations: List[str]
    generated_at: datetime

# ============================================================================
# CORRELATIONS
# ============================================================================

class CorrelationPair(BaseModel):
    """Correlation between two features"""
    feature1: str
    feature2: str
    correlation: float
    p_value: Optional[float] = None
    strength: str  # strong_positive, strong_negative, weak, etc.

class CorrelationResponse(BaseModel):
    """Correlation analysis response"""
    correlation_type: str  # pearson, spearman
    pairs: List[CorrelationPair]
    high_correlation_pairs: int
    multicollinearity_detected: bool
    generated_at: datetime

# ============================================================================
# VISUALIZATIONS
# ============================================================================

class Visualization(BaseModel):
    """Single visualization metadata"""
    id: str
    type: str  # distribution, correlation, box_plot, etc.
    title: str
    column: Optional[str] = None
    url: str
    created_at: datetime

class VisualizationsResponse(BaseModel):
    """List of visualizations"""
    visualizations: List[Visualization]
    total_visualizations: int

# ============================================================================
# COLUMN ANALYSIS
# ============================================================================

class ColumnAnalysisResponse(BaseModel):
    """Detailed analysis for a single column"""
    column_name: str
    data_type: str
    statistics: Optional[Dict[str, Any]] = None
    outliers: Optional[Dict[str, Any]] = None
    distribution: Optional[str] = None
    visualization_url: Optional[str] = None

# ============================================================================
# FULL REPORT
# ============================================================================

class FullReportResponse(BaseModel):
    """Complete EDA report"""
    dataset_id: str
    profile: Optional[DataProfile] = None
    statistics: Optional[StatisticsResponse] = None
    quality: Optional[QualityReportResponse] = None
    correlations: Optional[CorrelationResponse] = None
    visualizations: Optional[VisualizationsResponse] = None
    generated_at: datetime
    format: str  # json, html, pdf

# ============================================================================
# ERROR RESPONSE
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    status_code: int
    timestamp: datetime

# ============================================================================
# SIMPLIFIED SCHEMAS FOR NEW EDA API
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request to start analysis"""
    dataset_id: str

class AnalysisResponse(BaseModel):
    """Response when analysis starts"""
    job_id: str
    dataset_id: str
    status: str
    message: str

class SummaryResponse(BaseModel):
    """Data summary response"""
    dataset_id: str
    shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    memory_usage: str

class StatisticsSimpleResponse(BaseModel):
    """Simple statistics response"""
    dataset_id: str
    basic_stats: Dict[str, Any]
    missing_values: Dict[str, Any]
    duplicates: int

class QualityResponse(BaseModel):
    """Quality report response"""
    dataset_id: str
    completeness: float
    uniqueness: float
    validity: float
    consistency: float
    duplicate_rows: int
    missing_values_count: int
    total_cells: int

class CorrelationsResponse(BaseModel):
    """Correlations response"""
    dataset_id: str
    correlations: Dict[str, float]
    threshold: float
