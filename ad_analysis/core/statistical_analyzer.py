"""
Statistical Analysis Module
===========================

Advanced statistical analysis functions for data interpretation.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from scipy import stats
import warnings


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis class for data insights.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the StatisticalAnalyzer.
        
        Args:
            data: DataFrame to analyze
        """
        self.data = data
        self.results = {}
    
    def set_data(self, data: pd.DataFrame) -> 'StatisticalAnalyzer':
        """
        Set the data for analysis.
        
        Args:
            data: DataFrame to analyze
        
        Returns:
            Self for method chaining
        """
        self.data = data
        return self
    
    def descriptive_statistics(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive descriptive statistics.
        
        Args:
            columns: Columns to analyze (None for all numeric)
        
        Returns:
            Dictionary with descriptive statistics
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        target_cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        for col in target_cols:
            if col not in self.data.columns:
                continue
                
            col_data = self.data[col].dropna()
            
            results[col] = {
                'count': len(col_data),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'mode': float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                'std': float(col_data.std()),
                'var': float(col_data.var()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'range': float(col_data.max() - col_data.min()),
                'q1': float(col_data.quantile(0.25)),
                'q3': float(col_data.quantile(0.75)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'cv': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else float('inf')
            }
        
        self.results['descriptive_stats'] = results
        return results
    
    def correlation_analysis(self, method: str = 'pearson', threshold: float = 0.5) -> Dict[str, Any]:
        """
        Perform correlation analysis.
        
        Args:
            method: 'pearson', 'spearman', or 'kendall'
            threshold: Minimum correlation threshold for highlighting
        
        Returns:
            Dictionary with correlation results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found for correlation analysis.")
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr(method=method)
        
        # Find strong correlations
        strong_correlations = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_value),
                            'strength': self._interpret_correlation_strength(abs(corr_value))
                        })
        
        results = {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'method': method,
            'threshold': threshold
        }
        
        self.results['correlation_analysis'] = results
        return results
    
    def hypothesis_testing(self, test_type: str, **kwargs) -> Dict[str, Any]:
        """
        Perform various hypothesis tests.
        
        Args:
            test_type: Type of test ('ttest_1samp', 'ttest_ind', 'chi2', 'anova')
            **kwargs: Test-specific parameters
        
        Returns:
            Dictionary with test results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        results = {'test_type': test_type}
        
        if test_type == 'ttest_1samp':
            # One-sample t-test
            column = kwargs.get('column')
            pop_mean = kwargs.get('pop_mean', 0)
            
            if not column or column not in self.data.columns:
                raise ValueError("Valid column name required for one-sample t-test.")
            
            data_col = self.data[column].dropna()
            statistic, p_value = stats.ttest_1samp(data_col, pop_mean)
            
            results.update({
                'statistic': float(statistic),
                'p_value': float(p_value),
                'pop_mean': pop_mean,
                'sample_mean': float(data_col.mean()),
                'interpretation': self._interpret_p_value(p_value)
            })
        
        elif test_type == 'ttest_ind':
            # Independent two-sample t-test
            group1_col = kwargs.get('group1_column')
            group2_col = kwargs.get('group2_column')
            
            if not group1_col or not group2_col:
                raise ValueError("Both group columns required for independent t-test.")
            
            group1_data = self.data[group1_col].dropna()
            group2_data = self.data[group2_col].dropna()
            
            statistic, p_value = stats.ttest_ind(group1_data, group2_data)
            
            results.update({
                'statistic': float(statistic),
                'p_value': float(p_value),
                'group1_mean': float(group1_data.mean()),
                'group2_mean': float(group2_data.mean()),
                'interpretation': self._interpret_p_value(p_value)
            })
        
        elif test_type == 'chi2':
            # Chi-square test of independence
            col1 = kwargs.get('column1')
            col2 = kwargs.get('column2')
            
            if not col1 or not col2:
                raise ValueError("Both columns required for chi-square test.")
            
            contingency_table = pd.crosstab(self.data[col1], self.data[col2])
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            results.update({
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'contingency_table': contingency_table.to_dict(),
                'interpretation': self._interpret_p_value(p_value)
            })
        
        elif test_type == 'anova':
            # One-way ANOVA
            groups = kwargs.get('groups', [])
            
            if len(groups) < 2:
                raise ValueError("At least 2 groups required for ANOVA.")
            
            group_data = [self.data[group].dropna() for group in groups if group in self.data.columns]
            
            if len(group_data) < 2:
                raise ValueError("At least 2 valid groups required for ANOVA.")
            
            f_stat, p_value = stats.f_oneway(*group_data)
            
            results.update({
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'groups': groups,
                'group_means': {group: float(self.data[group].mean()) for group in groups if group in self.data.columns},
                'interpretation': self._interpret_p_value(p_value)
            })
        
        else:
            raise ValueError("Unsupported test type. Choose from: ttest_1samp, ttest_ind, chi2, anova")
        
        self.results['hypothesis_testing'] = results
        return results
    
    def outlier_detection(self, columns: Optional[List[str]] = None, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.
        
        Args:
            columns: Columns to analyze (None for all numeric)
            method: 'iqr', 'zscore', or 'isolation_forest'
        
        Returns:
            Dictionary with outlier detection results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        target_cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        results = {'method': method, 'outliers_by_column': {}}
        
        for col in target_cols:
            if col not in self.data.columns:
                continue
            
            col_data = self.data[col].dropna()
            outlier_indices = []
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_indices = col_data[outlier_mask].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(col_data))
                outlier_indices = col_data[z_scores > 3].index.tolist()
            
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                outlier_indices = col_data[outlier_labels == -1].index.tolist()
            
            results['outliers_by_column'][col] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(col_data)) * 100,
                'indices': outlier_indices,
                'values': col_data.loc[outlier_indices].tolist() if outlier_indices else []
            }
        
        self.results['outlier_detection'] = results
        return results
    
    def distribution_analysis(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze the distribution of numeric columns.
        
        Args:
            columns: Columns to analyze (None for all numeric)
        
        Returns:
            Dictionary with distribution analysis results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        target_cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        results = {}
        
        for col in target_cols:
            if col not in self.data.columns:
                continue
            
            col_data = self.data[col].dropna()
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(col_data) if len(col_data) <= 5000 else (None, None)
            ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
            
            results[col] = {
                'distribution_type': self._identify_distribution(col_data),
                'normality_tests': {
                    'shapiro_wilk': {
                        'statistic': float(shapiro_stat) if shapiro_stat else None,
                        'p_value': float(shapiro_p) if shapiro_p else None,
                        'is_normal': shapiro_p > 0.05 if shapiro_p else None
                    } if len(col_data) <= 5000 else None,
                    'kolmogorov_smirnov': {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'is_normal': ks_p > 0.05
                    }
                },
                'distribution_parameters': {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
            }
        
        self.results['distribution_analysis'] = results
        return results
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all analysis results."""
        return self.results.copy()
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        if correlation >= 0.8:
            return "Very Strong"
        elif correlation >= 0.6:
            return "Strong"
        elif correlation >= 0.4:
            return "Moderate"
        elif correlation >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def _interpret_p_value(self, p_value: float, alpha: float = 0.05) -> str:
        """Interpret p-value for hypothesis testing."""
        if p_value < alpha:
            return f"Statistically significant (p < {alpha})"
        else:
            return f"Not statistically significant (p >= {alpha})"
    
    def _identify_distribution(self, data: pd.Series) -> str:
        """Identify the most likely distribution type."""
        # Simple heuristic based on skewness and kurtosis
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "Normal"
        elif skewness > 1:
            return "Right-skewed"
        elif skewness < -1:
            return "Left-skewed"
        elif kurtosis > 3:
            return "Heavy-tailed"
        elif kurtosis < -1:
            return "Light-tailed"
        else:
            return "Unknown"