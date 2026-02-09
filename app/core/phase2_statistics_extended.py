"""
Phase 2: Advanced Statistics & Visualizations
Fixed version with proper JSON serialization
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class Phase2StatisticsExtended:
    """Advanced statistical analysis for Phase 2 EDA"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize Phase 2 statistics analyzer

        Args:
            df: pandas DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def get_histograms(self, bins: int = 15) -> dict:
        """
        Generate histogram data for numeric columns

        Args:
            bins: number of histogram bins (default: 15)

        Returns:
            dict with histogram data for each numeric column
        """
        try:
            histograms = {}

            for col in self.numeric_cols:
                try:
                    data = self.df[col].dropna()

                    if len(data) == 0:
                        continue

                    # Create histogram
                    counts, bin_edges = np.histogram(data, bins=bins)

                    # Create bin labels
                    bin_labels = [
                        f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                        for i in range(len(bin_edges)-1)
                    ]

                    # Statistics
                    stats_dict = {
                        "mean": float(np.mean(data)),
                        "median": float(np.median(data)),
                        "std": float(np.std(data)),
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "q1": float(np.percentile(data, 25)),
                        "q3": float(np.percentile(data, 75))
                    }

                    histograms[col] = {
                        "column": col,
                        "bins": bin_labels,
                        "frequencies": [int(count) for count in counts],
                        "bin_edges": [float(edge) for edge in bin_edges],
                        "total_count": int(len(self.df[col])),
                        "missing_count": int(self.df[col].isna().sum()),
                        "statistics": stats_dict
                    }
                except Exception as e:
                    print(f"Warning: Could not process histogram for {col}: {str(e)}")
                    continue

            return {
                "histograms": histograms,
                "total_numeric_columns": len(self.numeric_cols),
                "successfully_generated": len(histograms)
            }

        except Exception as e:
            raise Exception(f"Error generating histograms: {str(e)}")

    def get_outliers(self) -> dict:
        """
        Detect outliers using IQR method

        Returns:
            dict with outlier information for each numeric column
        """
        try:
            outliers_dict = {}
            columns_with_outliers = 0

            for col in self.numeric_cols:
                try:
                    data = self.df[col].dropna()

                    if len(data) < 4:  # Need at least 4 values for IQR
                        continue

                    Q1 = np.percentile(data, 25)
                    Q3 = np.percentile(data, 75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outlier_mask = (data < lower_bound) | (data > upper_bound)
                    outlier_indices = data[outlier_mask].index.tolist()
                    outlier_values = data[outlier_mask].values

                    outlier_count = len(outlier_indices)
                    outlier_percentage = (outlier_count / len(data)) * 100 if len(data) > 0 else 0

                    if outlier_count > 0:
                        columns_with_outliers += 1

                    outliers_dict[col] = {
                        "column": col,
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "IQR": float(IQR),
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": float(outlier_percentage),
                        "outlier_indices": [int(idx) for idx in outlier_indices],
                        "min_outlier": float(np.min(outlier_values)) if len(outlier_values) > 0 else None,
                        "max_outlier": float(np.max(outlier_values)) if len(outlier_values) > 0 else None,
                        "statistics": {
                            "mean": float(np.mean(data)),
                            "median": float(np.median(data)),
                            "q1": float(Q1),
                            "q3": float(Q3)
                        }
                    }
                except Exception as e:
                    print(f"Warning: Could not detect outliers for {col}: {str(e)}")
                    continue

            return {
                "outliers": outliers_dict,
                "total_numeric_columns": len(self.numeric_cols),
                "columns_with_outliers": columns_with_outliers,
                "method": "IQR (Interquartile Range)"
            }

        except Exception as e:
            raise Exception(f"Error detecting outliers: {str(e)}")

    def get_normality_tests(self) -> dict:
        """
        Test normality using Shapiro-Wilk test

        Returns:
            dict with normality test results for each numeric column
        """
        try:
            normality_dict = {}
            normal_count = 0

            for col in self.numeric_cols:
                try:
                    data = self.df[col].dropna()

                    if len(data) < 3:  # Shapiro-Wilk requires at least 3 samples
                        continue

                    # Perform Shapiro-Wilk test
                    statistic, p_value = stats.shapiro(data)

                    # Calculate skewness and kurtosis
                    skewness = float(stats.skew(data))
                    kurtosis = float(stats.kurtosis(data))

                    # Determine if normal (p > 0.05)
                    is_normal = bool(p_value > 0.05)

                    if is_normal:
                        normal_count += 1

                    # Interpretation
                    if is_normal:
                        interpretation = "Approximately normal"
                    else:
                        interpretation = "Not normally distributed"

                    normality_dict[col] = {
                        "column": col,
                        "test": "Shapiro-Wilk",
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "is_normal": int(is_normal),  # Convert to int for JSON
                        "interpretation": interpretation,
                        "skewness": skewness,
                        "kurtosis": kurtosis,
                        "sample_size": int(len(data))
                    }
                except Exception as e:
                    print(f"Warning: Could not test normality for {col}: {str(e)}")
                    continue

            non_normal_count = len(normality_dict) - normal_count

            return {
                "normality_tests": normality_dict,
                "total_numeric_columns": len(self.numeric_cols),
                "normal_columns": normal_count,
                "non_normal_columns": non_normal_count
            }

        except Exception as e:
            raise Exception(f"Error running normality tests: {str(e)}")

    def get_distribution_analysis(self) -> dict:
        """
        Analyze distribution characteristics

        Returns:
            dict with distribution analysis for each numeric column
        """
        try:
            distributions = {}

            for col in self.numeric_cols:
                try:
                    data = self.df[col].dropna()

                    if len(data) < 2:
                        continue

                    # Calculate skewness and kurtosis
                    skewness = float(stats.skew(data))
                    kurtosis = float(stats.kurtosis(data))

                    # Determine skewness type
                    if abs(skewness) < 0.5:
                        skewness_type = "Approximately Symmetric"
                    elif skewness > 0:
                        skewness_type = "Right-skewed (Positive skew)"
                    else:
                        skewness_type = "Left-skewed (Negative skew)"

                    # Determine kurtosis type
                    if abs(kurtosis) < 0.5:
                        kurtosis_type = "Mesokurtic (normal-like)"
                    elif kurtosis > 0:
                        kurtosis_type = "Leptokurtic (heavy tails)"
                    else:
                        kurtosis_type = "Platykurtic (light tails)"

                    # Calculate range and coefficient of variation
                    data_range = float(np.max(data) - np.min(data))
                    mean = float(np.mean(data))
                    std = float(np.std(data))
                    cv = (std / mean * 100) if mean != 0 else 0

                    distributions[col] = {
                        "column": col,
                        "skewness": skewness,
                        "kurtosis": kurtosis,
                        "distribution_type": skewness_type,
                        "kurtosis_type": kurtosis_type,
                        "characteristics": [
                            f"Skewness: {skewness_type}",
                            f"Kurtosis: {kurtosis_type}",
                            f"Range: {data_range:.2f}",
                            f"CV: {cv:.2f}%"
                        ]
                    }
                except Exception as e:
                    print(f"Warning: Could not analyze distribution for {col}: {str(e)}")
                    continue

            return {
                "distributions": distributions,
                "total_numeric_columns": len(self.numeric_cols),
                "analyzed_columns": len(distributions)
            }

        except Exception as e:
            raise Exception(f"Error analyzing distributions: {str(e)}")

    def get_categorical_distributions(self, top_n: int = 10) -> dict:
        """
        Get categorical value distributions

        Args:
            top_n: number of top categories to return

        Returns:
            dict with category distributions
        """
        try:
            categorical_dict = {}

            for col in self.categorical_cols:
                try:
                    value_counts = self.df[col].value_counts(dropna=False)
                    total_rows = len(self.df)

                    # Get top N values
                    top_values_dict = {}
                    for value, count in value_counts.head(top_n).items():
                        percentage = (count / total_rows) * 100
                        top_values_dict[str(value)] = {
                            "count": int(count),
                            "percentage": float(percentage)
                        }

                    categorical_dict[col] = {
                        "column": col,
                        "unique_values": int(self.df[col].nunique()),
                        "total_rows": int(total_rows),
                        "missing_count": int(self.df[col].isna().sum()),
                        "top_values": top_values_dict
                    }
                except Exception as e:
                    print(f"Warning: Could not analyze categorical column {col}: {str(e)}")
                    continue

            return {
                "categorical_distributions": categorical_dict,
                "total_categorical_columns": len(self.categorical_cols),
                "analyzed_columns": len(categorical_dict)
            }

        except Exception as e:
            raise Exception(f"Error analyzing categorical data: {str(e)}")

    def get_enhanced_correlations(self, threshold: float = 0.3) -> dict:
        """
        Calculate correlations with p-values

        Args:
            threshold: minimum correlation threshold (default: 0.3)

        Returns:
            dict with correlation analysis
        """
        try:
            if len(self.numeric_cols) < 2:
                return {
                    "all_correlations": {},
                    "high_correlations": [],
                    "threshold": threshold,
                    "total_correlations": 0,
                    "high_correlation_count": 0
                }

            # Calculate correlation matrix
            corr_matrix = self.df[self.numeric_cols].corr()

            all_correlations = {}
            high_correlations = []

            # Iterate through correlation pairs
            for i, col1 in enumerate(self.numeric_cols):
                for j, col2 in enumerate(self.numeric_cols):
                    if i >= j:  # Skip duplicates and self-correlations
                        continue

                    correlation = float(corr_matrix.loc[col1, col2])

                    # Calculate p-value
                    try:
                        # Get data for both columns
                        data1 = self.df[col1].dropna()
                        data2 = self.df[col2].dropna()

                        # Find common indices
                        common_idx = data1.index.intersection(data2.index)

                        if len(common_idx) > 2:
                            _, p_value = stats.pearsonr(data1[common_idx], data2[common_idx])
                        else:
                            p_value = 1.0
                    except:
                        p_value = 1.0

                    p_value = float(p_value)

                    # Determine significance
                    is_significant = int(p_value < 0.05)  # Convert to int for JSON

                    # Determine strength
                    abs_corr = abs(correlation)
                    if abs_corr < 0.3:
                        strength = "Very Weak"
                    elif abs_corr < 0.5:
                        strength = "Weak"
                    elif abs_corr < 0.7:
                        strength = "Moderate"
                    elif abs_corr < 0.9:
                        strength = "Strong"
                    else:
                        strength = "Very Strong"

                    # Store in all correlations
                    pair_key = f"{col1}-{col2}"
                    all_correlations[pair_key] = {
                        "correlation": correlation,
                        "p_value": p_value,
                        "significant": is_significant,  # Already int
                        "strength": strength
                    }

                    # Add to high correlations if above threshold
                    if abs_corr >= threshold and is_significant:
                        high_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": correlation,
                            "p_value": p_value,
                            "strength": strength
                        })

            # Sort high correlations by absolute value
            high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

            return {
                "all_correlations": all_correlations,
                "high_correlations": high_correlations,
                "threshold": float(threshold),
                "total_correlations": len(all_correlations),
                "high_correlation_count": len(high_correlations)
            }

        except Exception as e:
            raise Exception(f"Error analyzing correlations: {str(e)}")