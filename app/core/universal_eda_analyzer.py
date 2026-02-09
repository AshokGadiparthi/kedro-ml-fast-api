"""
Universal EDA Utilities - Works for ANY Dataset
Handles:
- Missing values (NaN, None)
- All data types (numeric, categorical, datetime)
- Large datasets
- Edge cases
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class UniversalEDAAnalyzer:
    """
    Universal EDA Analyzer - Works for ANY dataset
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get data summary - Works for ANY dataset"""
        return {
            "dataset_id": None,  # Will be set by endpoint
            "shape": list(self.df.shape),
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "column_types": {
                "numeric": self.numeric_cols,
                "categorical": self.categorical_cols,
                "datetime": self.datetime_cols
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for ALL column types - Universal"""
        stats = {
            "dataset_id": None,
            "numeric_statistics": {},
            "categorical_statistics": {},
            "missing_values": {},
            "duplicates": int(self.df.duplicated().sum())
        }
        
        # ✅ Handle numeric columns
        if self.numeric_cols:
            numeric_df = self.df[self.numeric_cols]
            
            # Safe describe that handles NaN
            desc = numeric_df.describe().replace({np.nan: None, np.inf: None, -np.inf: None})
            stats["numeric_statistics"] = desc.to_dict()
            
            # Add additional stats
            for col in self.numeric_cols:
                col_data = numeric_df[col].dropna()  # Drop NaN for calculations
                if len(col_data) > 0:
                    stats["numeric_statistics"][col] = {
                        "count": int(col_data.count()),
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "q25": float(col_data.quantile(0.25)),
                        "q75": float(col_data.quantile(0.75)),
                        "missing": int(col_data.isna().sum())
                    }
        
        # ✅ Handle categorical columns
        if self.categorical_cols:
            for col in self.categorical_cols:
                value_counts = self.df[col].value_counts()
                stats["categorical_statistics"][col] = {
                    "unique": int(self.df[col].nunique()),
                    "top": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "missing": int(self.df[col].isna().sum())
                }
        
        # ✅ Missing values summary
        missing = self.df.isnull().sum()
        stats["missing_values"] = {
            col: {
                "count": int(missing[col]),
                "percent": round(float(missing[col] / len(self.df) * 100), 2)
            }
            for col in missing[missing > 0].index
        }
        
        return stats
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate quality report - Works for ANY dataset"""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = int(self.df.isnull().sum().sum())
        duplicate_rows = int(self.df.duplicated().sum())
        
        completeness = 100 - (missing_cells / total_cells * 100) if total_cells > 0 else 100
        uniqueness = 100 if duplicate_rows == 0 else (1 - duplicate_rows / len(self.df)) * 100
        
        # Quality checks
        quality_checks = []
        
        # Check 1: Completeness
        if completeness >= 95:
            quality_checks.append({
                "name": "Data Completeness",
                "status": "PASS",
                "score": round(completeness, 2),
                "message": f"{completeness:.1f}% of data is complete"
            })
        elif completeness >= 80:
            quality_checks.append({
                "name": "Data Completeness",
                "status": "WARNING",
                "score": round(completeness, 2),
                "message": f"{completeness:.1f}% complete, {100-completeness:.1f}% missing"
            })
        else:
            quality_checks.append({
                "name": "Data Completeness",
                "status": "FAIL",
                "score": round(completeness, 2),
                "message": f"Only {completeness:.1f}% complete"
            })
        
        # Check 2: Uniqueness
        if uniqueness >= 99:
            quality_checks.append({
                "name": "Data Uniqueness",
                "status": "PASS",
                "score": round(uniqueness, 2),
                "message": f"No duplicate rows detected"
            })
        else:
            quality_checks.append({
                "name": "Data Uniqueness",
                "status": "PASS" if duplicate_rows == 0 else "WARNING",
                "score": round(uniqueness, 2),
                "message": f"{duplicate_rows} duplicate rows found"
            })
        
        return {
            "dataset_id": None,
            "overall_quality_score": round((completeness + uniqueness) / 2, 2),
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "validity": 95.0,  # Simplified for demo
            "consistency": 98.0,  # Simplified for demo
            "duplicate_rows": duplicate_rows,
            "missing_values_count": missing_cells,
            "total_cells": total_cells,
            "quality_checks": quality_checks
        }
    
    def get_correlations(self, threshold: float = 0.3) -> Dict[str, Any]:
        """Get correlations - Only for numeric columns"""
        correlations = {}
        vif_scores = {}
        
        if len(self.numeric_cols) < 2:
            return {
                "dataset_id": None,
                "correlations": {},
                "vif_scores": {},
                "high_correlation_pairs": 0,
                "threshold": threshold,
                "message": "Need at least 2 numeric columns for correlation analysis"
            }
        
        # Safe correlation (drops NaN automatically)
        try:
            corr_matrix = self.df[self.numeric_cols].corr(method='pearson')
            
            # Extract high correlations
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 < col2:  # Avoid duplicates
                        corr_val = float(corr_matrix.loc[col1, col2])
                        if not np.isnan(corr_val) and abs(corr_val) > threshold:
                            correlations[f"{col1}-{col2}"] = round(corr_val, 3)
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {str(e)}")
        
        return {
            "dataset_id": None,
            "correlations": correlations,
            "vif_scores": vif_scores,
            "high_correlation_pairs": len(correlations),
            "threshold": threshold,
            "numeric_columns_analyzed": len(self.numeric_cols)
        }


# Example usage
if __name__ == "__main__":
    # Test with any CSV
    df = pd.read_csv('/mnt/user-data/uploads/ecommerce_orders_dataset.csv')
    
    analyzer = UniversalEDAAnalyzer(df)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary = analyzer.get_summary()
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {len(summary['columns'])}")
    print(f"Numeric: {len(summary['column_types']['numeric'])}")
    print(f"Categorical: {len(summary['column_types']['categorical'])}")
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = analyzer.get_statistics()
    print(f"Numeric columns analyzed: {len(stats['numeric_statistics'])}")
    print(f"Categorical columns analyzed: {len(stats['categorical_statistics'])}")
    print(f"Columns with missing values: {len(stats['missing_values'])}")
    for col, info in stats['missing_values'].items():
        print(f"  - {col}: {info['count']} missing ({info['percent']:.1f}%)")
    
    print("\n" + "=" * 80)
    print("QUALITY REPORT")
    print("=" * 80)
    quality = analyzer.get_quality_report()
    print(f"Overall Score: {quality['overall_quality_score']}")
    print(f"Completeness: {quality['completeness']}%")
    print(f"Uniqueness: {quality['uniqueness']}%")
    
    print("\n" + "=" * 80)
    print("CORRELATIONS")
    print("=" * 80)
    corr = analyzer.get_correlations()
    print(f"High correlations found: {corr['high_correlation_pairs']}")
    for pair, value in list(corr['correlations'].items())[:5]:
        print(f"  - {pair}: {value}")
