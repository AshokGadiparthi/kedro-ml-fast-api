"""
Phase 3: Advanced Correlations & Multicollinearity Analysis
Complete module for correlation heatmaps, VIF, and relationship insights
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

warnings.filterwarnings('ignore')


class AdvancedCorrelationAnalysis:
    """Advanced correlation analysis with VIF, multicollinearity detection"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_df = df[self.numeric_cols].dropna()

    def get_enhanced_correlations(self, threshold: float = 0.3) -> dict:
        """
        Get comprehensive correlation analysis

        Returns:
            dict with correlation matrix, pairs, and insights
        """
        # Calculate correlation matrix
        corr_matrix = self.numeric_df.corr()

        return {
            "correlation_matrix": self._format_correlation_matrix(corr_matrix),
            "correlation_pairs": self._extract_correlation_pairs(corr_matrix, threshold),
            "high_correlations": self._find_high_correlations(corr_matrix, threshold=0.7),
            "very_high_correlations": self._find_high_correlations(corr_matrix, threshold=0.9),
            "multicollinearity_pairs": self._detect_multicollinearity_pairs(corr_matrix),
            "correlation_strength_distribution": self._strength_distribution(corr_matrix),
            "statistics": self._correlation_statistics(corr_matrix)
        }

    def get_vif_analysis(self) -> dict:
        """
        Calculate Variance Inflation Factor (VIF) for all numeric features

        VIF > 10: High multicollinearity (problematic)
        VIF > 5: Moderate multicollinearity (caution needed)
        VIF < 5: Acceptable
        """
        from sklearn.preprocessing import StandardScaler

        vif_results = {}

        # Standardize features for VIF calculation
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.numeric_df)
        scaled_df = pd.DataFrame(scaled_data, columns=self.numeric_cols)

        # Calculate correlation matrix
        corr_matrix = scaled_df.corr()

        try:
            # Calculate VIF for each feature
            for i, col in enumerate(self.numeric_cols):
                try:
                    # Calculate R-squared from regression
                    X = scaled_df.drop(columns=[col])
                    y = scaled_df[col]

                    # Calculate R-squared using correlation
                    if len(X.columns) > 0:
                        # Multiple regression R-squared approximation
                        correlations = corr_matrix.loc[col, X.columns].values
                        r_squared = np.sum(correlations ** 2) / len(correlations) if len(correlations) > 0 else 0
                        r_squared = max(0, min(1, r_squared))  # Bound between 0 and 1

                        # VIF = 1 / (1 - R²)
                        vif = 1 / (1 - r_squared + 0.0001) if r_squared < 0.9999 else 100.0
                    else:
                        vif = 1.0

                    vif = float(vif)

                    # Determine severity
                    if vif > 10:
                        severity = "CRITICAL"
                        status = "⚠️ High multicollinearity"
                    elif vif > 5:
                        severity = "WARNING"
                        status = "⚠️ Moderate multicollinearity"
                    else:
                        severity = "OK"
                        status = "✅ Acceptable"

                    vif_results[col] = {
                        "vif_score": vif,
                        "severity": severity,
                        "status": status,
                        "interpretation": f"VIF = {vif:.2f}",
                        "recommendation": self._vif_recommendation(vif)
                    }
                except Exception as e:
                    vif_results[col] = {
                        "vif_score": 1.0,
                        "severity": "OK",
                        "status": "✅ Acceptable",
                        "interpretation": "VIF = 1.0",
                        "recommendation": "Feature has low multicollinearity"
                    }
        except Exception as e:
            pass

        return {
            "vif_scores": vif_results,
            "problematic_features": self._get_problematic_features(vif_results),
            "overall_multicollinearity_level": self._assess_multicollinearity_level(vif_results),
            "interpretation": self._vif_interpretation(vif_results)
        }

    def get_correlation_heatmap_data(self) -> dict:
        """Get data formatted for heatmap visualization"""
        corr_matrix = self.numeric_df.corr()

        # Convert to list format for JSON serialization
        heatmap_data = []
        for i, col1 in enumerate(self.numeric_cols):
            row = []
            for j, col2 in enumerate(self.numeric_cols):
                value = float(corr_matrix.loc[col1, col2])
                row.append({
                    "x": col2,
                    "y": col1,
                    "correlation": value,
                    "abs_correlation": abs(value),
                    "strength": self._correlation_strength(value)
                })
            heatmap_data.append(row)

        return {
            "heatmap": heatmap_data,
            "numeric_columns": self.numeric_cols,
            "min_value": float(corr_matrix.values.min()),
            "max_value": float(corr_matrix.values.max())
        }

    def get_correlation_clustering(self) -> dict:
        """Detect clusters of highly correlated features"""
        corr_matrix = self.numeric_df.corr()

        # Convert correlation to distance
        distance_matrix = 1 - np.abs(corr_matrix)

        try:
            # Hierarchical clustering
            from scipy.cluster.hierarchy import linkage, fcluster

            # Flatten distance matrix for linkage
            from scipy.spatial.distance import squareform
            condensed_dist = squareform(distance_matrix)

            # Perform hierarchical clustering
            Z = linkage(condensed_dist, method='ward')

            # Get clusters (cut at distance threshold)
            clusters = fcluster(Z, t=0.5, criterion='distance')

            # Group features by cluster
            cluster_groups = {}
            for feature, cluster_id in zip(self.numeric_cols, clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(feature)

            return {
                "clusters": cluster_groups,
                "cluster_count": len(cluster_groups),
                "cluster_interpretation": self._interpret_clusters(cluster_groups, corr_matrix)
            }
        except:
            return {
                "clusters": {},
                "cluster_count": 0,
                "cluster_interpretation": "Clustering not available for this dataset"
            }

    def get_relationship_insights(self) -> dict:
        """Generate insights about feature relationships"""
        corr_matrix = self.numeric_df.corr()

        insights = {
            "strongest_positive_relationships": [],
            "strongest_negative_relationships": [],
            "uncorrelated_pairs": [],
            "interesting_patterns": [],
            "feature_connectivity": {}
        }

        # Find strongest relationships
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:  # Avoid duplicates
                    corr = float(corr_matrix.loc[col1, col2])

                    if corr > 0.7:
                        insights["strongest_positive_relationships"].append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": corr,
                            "interpretation": f"{col1} and {col2} are strongly positively correlated",
                            "implication": "Consider removing one for multicollinearity"
                        })
                    elif corr < -0.7:
                        insights["strongest_negative_relationships"].append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": corr,
                            "interpretation": f"{col1} and {col2} are strongly negatively correlated",
                            "implication": "Inverse relationship detected"
                        })
                    elif abs(corr) < 0.1:
                        insights["uncorrelated_pairs"].append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": corr,
                            "interpretation": "These features are independent"
                        })

        # Calculate feature connectivity (how correlated each feature is with others)
        for col in self.numeric_cols:
            correlations = corr_matrix[col].drop(col)
            strong_corr_count = (abs(correlations) > 0.5).sum()
            insights["feature_connectivity"][col] = {
                "total_features": len(self.numeric_cols) - 1,
                "strongly_correlated": int(strong_corr_count),
                "connectivity_score": float(strong_corr_count / (len(self.numeric_cols) - 1))
            }

        # Sort by strongest correlations
        insights["strongest_positive_relationships"] = sorted(
            insights["strongest_positive_relationships"],
            key=lambda x: abs(x["correlation"]),
            reverse=True
        )[:5]  # Top 5

        insights["strongest_negative_relationships"] = sorted(
            insights["strongest_negative_relationships"],
            key=lambda x: abs(x["correlation"]),
            reverse=True
        )[:5]  # Top 5

        return insights

    def get_multicollinearity_warnings(self) -> dict:
        """Generate multicollinearity warnings and recommendations"""
        warnings_list = []

        # Get VIF results
        vif_data = self.get_vif_analysis()

        # Check for high VIF scores
        for feature, vif_info in vif_data["vif_scores"].items():
            if vif_info["severity"] != "OK":
                warnings_list.append({
                    "type": "HIGH_VIF",
                    "severity": vif_info["severity"],
                    "feature": feature,
                    "vif_score": vif_info["vif_score"],
                    "message": f"Feature '{feature}' has VIF = {vif_info['vif_score']:.2f}",
                    "recommendation": vif_info["recommendation"]
                })

        # Get correlation-based warnings
        corr_matrix = self.numeric_df.corr()
        high_corr_pairs = []

        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:
                    corr = float(corr_matrix.loc[col1, col2])
                    if abs(corr) > 0.9:
                        high_corr_pairs.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": corr,
                            "redundancy_risk": "CRITICAL"
                        })
                    elif abs(corr) > 0.8:
                        high_corr_pairs.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": corr,
                            "redundancy_risk": "HIGH"
                        })

        if high_corr_pairs:
            warnings_list.append({
                "type": "HIGH_CORRELATION_PAIRS",
                "severity": "WARNING",
                "message": f"Detected {len(high_corr_pairs)} highly correlated feature pairs",
                "pairs": high_corr_pairs[:10],  # Top 10
                "recommendation": "Consider removing redundant features"
            })

        return {
            "warning_count": len(warnings_list),
            "warnings": warnings_list,
            "overall_assessment": self._multicollinearity_assessment(warnings_list)
        }

    # Helper methods
    def _format_correlation_matrix(self, corr_matrix: pd.DataFrame) -> dict:
        """Format correlation matrix for API response"""
        return {
            col: {key: float(val) for key, val in corr_matrix[col].items()}
            for col in corr_matrix.columns
        }

    def _extract_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.3) -> list:
        """Extract correlation pairs above threshold"""
        pairs = []
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:
                    corr = float(corr_matrix.loc[col1, col2])
                    if abs(corr) >= threshold:
                        # Calculate p-value
                        data1 = self.numeric_df[col1]
                        data2 = self.numeric_df[col2]
                        _, p_value = stats.pearsonr(data1, data2)

                        pairs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr,
                            "p_value": float(p_value),
                            "significant": int(p_value < 0.05),
                            "strength": self._correlation_strength(corr)
                        })

        return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list:
        """Find correlations above threshold"""
        high_corr = []
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:
                    corr = float(corr_matrix.loc[col1, col2])
                    if abs(corr) >= threshold:
                        high_corr.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": corr,
                            "strength": self._correlation_strength(corr)
                        })

        return sorted(high_corr, key=lambda x: abs(x["correlation"]), reverse=True)

    def _detect_multicollinearity_pairs(self, corr_matrix: pd.DataFrame) -> list:
        """Detect problematic multicollinearity pairs"""
        pairs = []
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:
                    corr = float(corr_matrix.loc[col1, col2])
                    if abs(corr) > 0.8:
                        pairs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr,
                            "redundancy_level": "CRITICAL" if abs(corr) > 0.95 else "HIGH",
                            "recommendation": f"Consider removing {'column2' if abs(corr) > 0.95 else 'one of the features'}"
                        })

        return pairs

    def _strength_distribution(self, corr_matrix: pd.DataFrame) -> dict:
        """Get distribution of correlation strengths"""
        correlations = []
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:
                    correlations.append(abs(float(corr_matrix.loc[col1, col2])))

        return {
            "very_strong": int(sum(1 for c in correlations if c > 0.9)),
            "strong": int(sum(1 for c in correlations if 0.7 < c <= 0.9)),
            "moderate": int(sum(1 for c in correlations if 0.5 < c <= 0.7)),
            "weak": int(sum(1 for c in correlations if 0.3 < c <= 0.5)),
            "very_weak": int(sum(1 for c in correlations if c <= 0.3))
        }

    def _correlation_statistics(self, corr_matrix: pd.DataFrame) -> dict:
        """Calculate correlation statistics"""
        values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

        return {
            "mean_correlation": float(np.mean(np.abs(values))),
            "max_correlation": float(np.max(np.abs(values))),
            "min_correlation": float(np.min(np.abs(values))),
            "median_correlation": float(np.median(np.abs(values))),
            "total_pairs": int(len(values)),
            "std_correlation": float(np.std(np.abs(values)))
        }

    def _correlation_strength(self, corr: float) -> str:
        """Determine correlation strength label"""
        abs_corr = abs(corr)
        if abs_corr > 0.9:
            return "Very Strong"
        elif abs_corr > 0.7:
            return "Strong"
        elif abs_corr > 0.5:
            return "Moderate"
        elif abs_corr > 0.3:
            return "Weak"
        else:
            return "Very Weak"

    def _vif_recommendation(self, vif: float) -> str:
        """Get recommendation based on VIF score"""
        if vif > 10:
            return "CRITICAL: Remove this feature or its correlated counterpart"
        elif vif > 5:
            return "WARNING: Consider feature selection or dimensionality reduction"
        else:
            return "OK: Feature multicollinearity is acceptable"

    def _get_problematic_features(self, vif_results: dict) -> list:
        """Get list of features with problematic VIF"""
        problematic = []
        for feature, info in vif_results.items():
            if info["severity"] != "OK":
                problematic.append({
                    "feature": feature,
                    "vif_score": info["vif_score"],
                    "severity": info["severity"]
                })

        return sorted(problematic, key=lambda x: x["vif_score"], reverse=True)

    def _assess_multicollinearity_level(self, vif_results: dict) -> str:
        """Assess overall multicollinearity level"""
        critical_count = sum(1 for info in vif_results.values() if info["severity"] == "CRITICAL")
        warning_count = sum(1 for info in vif_results.values() if info["severity"] == "WARNING")

        if critical_count > 0:
            return "HIGH"
        elif warning_count > 2:
            return "MODERATE"
        else:
            return "LOW"

    def _vif_interpretation(self, vif_results: dict) -> str:
        """Generate interpretation of VIF results"""
        level = self._assess_multicollinearity_level(vif_results)

        if level == "HIGH":
            return "The dataset has significant multicollinearity issues that need attention"
        elif level == "MODERATE":
            return "Moderate multicollinearity detected. Consider feature selection"
        else:
            return "Multicollinearity levels are acceptable for most modeling tasks"

    def _interpret_clusters(self, cluster_groups: dict, corr_matrix: pd.DataFrame) -> str:
        """Interpret feature clusters"""
        if len(cluster_groups) <= 1:
            return "Features form a single cluster, indicating high overall correlation"
        else:
            return f"Features form {len(cluster_groups)} clusters, indicating groups of correlated features"

    def _multicollinearity_assessment(self, warnings_list: list) -> str:
        """Generate overall multicollinearity assessment"""
        if not warnings_list:
            return "✅ No significant multicollinearity issues detected"
        elif len(warnings_list) == 1:
            return "⚠️ Minor multicollinearity issues detected"
        else:
            return "🔴 Significant multicollinearity issues detected - action recommended"