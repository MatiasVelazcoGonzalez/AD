"""
Data Interpretation Module
==========================

Advanced data interpretation and insights generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime


class DataInterpreter:
    """
    Advanced data interpretation class for generating insights and recommendations.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataInterpreter.
        
        Args:
            data: DataFrame to interpret
        """
        self.data = data
        self.insights = {}
        self.recommendations = {}
    
    def set_data(self, data: pd.DataFrame) -> 'DataInterpreter':
        """
        Set the data for interpretation.
        
        Args:
            data: DataFrame to interpret
        
        Returns:
            Self for method chaining
        """
        self.data = data
        return self
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Returns:
            Dictionary with data quality insights
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        report = {
            'overview': {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
                'data_types': self.data.dtypes.value_counts().to_dict()
            },
            'missing_data': {},
            'data_types_analysis': {},
            'duplicates': {},
            'outliers_summary': {},
            'recommendations': []
        }
        
        # Missing data analysis
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100
        
        report['missing_data'] = {
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'total_missing_values': missing_counts.sum(),
            'completely_empty_columns': missing_counts[missing_counts == len(self.data)].index.tolist()
        }
        
        # Data types analysis
        for dtype in self.data.dtypes.unique():
            cols = self.data.select_dtypes(include=[dtype]).columns.tolist()
            report['data_types_analysis'][str(dtype)] = {
                'columns': cols,
                'count': len(cols)
            }
        
        # Duplicates analysis
        duplicate_rows = self.data.duplicated().sum()
        report['duplicates'] = {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': (duplicate_rows / len(self.data)) * 100,
            'unique_rows': len(self.data) - duplicate_rows
        }
        
        # Outliers summary for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            col_data = self.data[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_summary[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(col_data)) * 100
                }
        
        report['outliers_summary'] = outlier_summary
        
        # Generate recommendations
        recommendations = []
        
        # Missing data recommendations
        high_missing_cols = [col for col, pct in missing_percentages.items() if pct > 50]
        if high_missing_cols:
            recommendations.append({
                'category': 'Missing Data',
                'issue': f'Columns with >50% missing data: {high_missing_cols}',
                'recommendation': 'Consider removing these columns or investigating why data is missing.',
                'priority': 'High'
            })
        
        moderate_missing_cols = [col for col, pct in missing_percentages.items() if 10 < pct <= 50]
        if moderate_missing_cols:
            recommendations.append({
                'category': 'Missing Data',
                'issue': f'Columns with 10-50% missing data: {moderate_missing_cols}',
                'recommendation': 'Consider imputation strategies (mean, median, mode) or advanced techniques.',
                'priority': 'Medium'
            })
        
        # Duplicate data recommendations
        if duplicate_rows > 0:
            recommendations.append({
                'category': 'Data Quality',
                'issue': f'{duplicate_rows} duplicate rows found ({(duplicate_rows/len(self.data)*100):.1f}%)',
                'recommendation': 'Review and remove duplicate records if they are true duplicates.',
                'priority': 'Medium'
            })
        
        # Outliers recommendations
        high_outlier_cols = [col for col, info in outlier_summary.items() if info['percentage'] > 5]
        if high_outlier_cols:
            recommendations.append({
                'category': 'Outliers',
                'issue': f'Columns with >5% outliers: {high_outlier_cols}',
                'recommendation': 'Investigate outliers - they may be data errors or important edge cases.',
                'priority': 'Medium'
            })
        
        report['recommendations'] = recommendations
        self.insights['data_quality_report'] = report
        
        return report
    
    def analyze_relationships(self, target_column: Optional[str] = None, 
                            correlation_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze relationships between variables.
        
        Args:
            target_column: Focus analysis on relationships with this column
            correlation_threshold: Minimum correlation to highlight
        
        Returns:
            Dictionary with relationship analysis
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No numeric columns found for relationship analysis'}
        
        analysis = {
            'correlation_matrix': numeric_data.corr().to_dict(),
            'strong_relationships': [],
            'weak_relationships': [],
            'target_analysis': {},
            'insights': []
        }
        
        # Find strong and weak relationships
        corr_matrix = numeric_data.corr()
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    if abs(corr_value) >= correlation_threshold:
                        analysis['strong_relationships'].append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_value),
                            'strength': self._interpret_correlation_strength(abs(corr_value)),
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        })
                    elif abs(corr_value) < 0.1:
                        analysis['weak_relationships'].append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_value)
                        })
        
        # Target column analysis
        if target_column and target_column in numeric_data.columns:
            target_corr = corr_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
            analysis['target_analysis'] = {
                'target_column': target_column,
                'strongest_predictors': target_corr.head(5).to_dict(),
                'weakest_predictors': target_corr.tail(5).to_dict()
            }
        
        # Generate insights
        insights = []
        
        if len(analysis['strong_relationships']) == 0:
            insights.append("No strong correlations found. Variables appear to be relatively independent.")
        else:
            strongest = max(analysis['strong_relationships'], key=lambda x: abs(x['correlation']))
            insights.append(f"Strongest relationship: {strongest['variable1']} and {strongest['variable2']} "
                          f"(r={strongest['correlation']:.3f})")
        
        if target_column and target_column in analysis['target_analysis']:
            target_predictors = analysis['target_analysis']['strongest_predictors']
            if target_predictors:
                best_predictor = max(target_predictors.items(), key=lambda x: abs(x[1]))
                insights.append(f"Best predictor of {target_column}: {best_predictor[0]} "
                              f"(r={best_predictor[1]:.3f})")
        
        analysis['insights'] = insights
        self.insights['relationship_analysis'] = analysis
        
        return analysis
    
    def detect_patterns_and_trends(self, date_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect patterns and trends in the data.
        
        Args:
            date_column: Column with datetime data for time series analysis
        
        Returns:
            Dictionary with pattern analysis
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        patterns = {
            'categorical_patterns': {},
            'numeric_patterns': {},
            'temporal_patterns': {},
            'insights': []
        }
        
        # Categorical patterns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = self.data[col].value_counts()
            patterns['categorical_patterns'][col] = {
                'unique_values': int(self.data[col].nunique()),
                'most_common': value_counts.head(3).to_dict(),
                'distribution': 'uniform' if value_counts.std() < value_counts.mean() * 0.5 else 'skewed'
            }
        
        # Numeric patterns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = self.data[col].dropna()
            if len(col_data) > 0:
                patterns['numeric_patterns'][col] = {
                    'distribution_type': self._identify_distribution_pattern(col_data),
                    'has_outliers': self._has_significant_outliers(col_data),
                    'trend': self._detect_trend(col_data) if date_column else 'unknown',
                    'seasonality': self._detect_seasonality(col_data) if date_column else 'unknown'
                }
        
        # Temporal patterns (if date column provided)
        if date_column and date_column in self.data.columns:
            try:
                date_data = pd.to_datetime(self.data[date_column])
                patterns['temporal_patterns'] = self._analyze_temporal_patterns(date_data, numeric_cols)
            except:
                patterns['temporal_patterns'] = {'error': 'Could not parse date column'}
        
        # Generate insights
        insights = []
        
        # Categorical insights
        for col, info in patterns['categorical_patterns'].items():
            if info['unique_values'] == 1:
                insights.append(f"Column '{col}' has only one unique value - consider removing it.")
            elif info['unique_values'] == len(self.data):
                insights.append(f"Column '{col}' has all unique values - might be an identifier.")
            elif info['distribution'] == 'skewed':
                most_common = list(info['most_common'].keys())[0]
                insights.append(f"Column '{col}' is dominated by '{most_common}' category.")
        
        # Numeric insights
        for col, info in patterns['numeric_patterns'].items():
            if info['has_outliers']:
                insights.append(f"Column '{col}' contains significant outliers that may need attention.")
            if info['distribution_type'] == 'highly_skewed':
                insights.append(f"Column '{col}' is highly skewed - consider transformation.")
        
        patterns['insights'] = insights
        self.insights['patterns_and_trends'] = patterns
        
        return patterns
    
    def generate_business_insights(self, context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate business-oriented insights from the data.
        
        Args:
            context: Dictionary with business context (e.g., {'domain': 'e-commerce', 'goal': 'increase_sales'})
        
        Returns:
            Dictionary with business insights
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        insights = {
            'key_metrics': {},
            'opportunities': [],
            'risks': [],
            'recommendations': [],
            'actionable_insights': []
        }
        
        # Calculate key metrics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = self.data[col].dropna()
            if len(col_data) > 0:
                insights['key_metrics'][col] = {
                    'average': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'total': float(col_data.sum()),
                    'range': float(col_data.max() - col_data.min()),
                    'coefficient_of_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else float('inf')
                }
        
        # Identify opportunities
        for col, metrics in insights['key_metrics'].items():
            cv = metrics['coefficient_of_variation']
            if cv > 1:  # High variability
                insights['opportunities'].append({
                    'metric': col,
                    'opportunity': 'High variability suggests potential for optimization',
                    'description': f'{col} shows high variability (CV={cv:.2f}), indicating inconsistent performance'
                })
            
            if metrics['average'] != metrics['median']:
                skew_direction = 'right' if metrics['average'] > metrics['median'] else 'left'
                insights['opportunities'].append({
                    'metric': col,
                    'opportunity': f'Distribution is {skew_direction}-skewed',
                    'description': f'Mean ({metrics["average"]:.2f}) differs from median ({metrics["median"]:.2f})'
                })
        
        # Identify risks
        missing_percentages = (self.data.isnull().sum() / len(self.data)) * 100
        for col, pct in missing_percentages.items():
            if pct > 20:
                insights['risks'].append({
                    'risk_type': 'Data Quality',
                    'description': f'Column {col} has {pct:.1f}% missing data',
                    'impact': 'High' if pct > 50 else 'Medium',
                    'mitigation': 'Implement data collection improvements or imputation strategies'
                })
        
        # Generate recommendations
        recommendations = []
        
        # Data completeness recommendations
        high_missing = [col for col, pct in missing_percentages.items() if pct > 30]
        if high_missing:
            recommendations.append({
                'category': 'Data Quality',
                'recommendation': f'Improve data collection for columns: {high_missing}',
                'expected_impact': 'Improved analysis reliability and model performance',
                'effort': 'Medium'
            })
        
        # Variability recommendations
        high_variability_cols = [col for col, metrics in insights['key_metrics'].items() 
                               if metrics['coefficient_of_variation'] > 1]
        if high_variability_cols:
            recommendations.append({
                'category': 'Performance Optimization',
                'recommendation': f'Investigate high variability in: {high_variability_cols}',
                'expected_impact': 'Reduced variability could improve consistency',
                'effort': 'Low to Medium'
            })
        
        insights['recommendations'] = recommendations
        
        # Context-specific insights
        if context:
            domain = context.get('domain', 'general')
            goal = context.get('goal', 'analysis')
            
            context_insights = self._generate_domain_specific_insights(domain, goal, insights)
            insights['actionable_insights'] = context_insights
        
        self.insights['business_insights'] = insights
        return insights
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all insights.
        
        Returns:
            Dictionary with complete analysis summary
        """
        if not self.insights:
            return {'error': 'No insights generated yet. Run analysis methods first.'}
        
        summary = {
            'report_timestamp': datetime.now().isoformat(),
            'data_overview': {
                'shape': self.data.shape if self.data is not None else 'No data',
                'columns': list(self.data.columns) if self.data is not None else []
            },
            'key_findings': [],
            'critical_issues': [],
            'recommendations': [],
            'next_steps': []
        }
        
        # Extract key findings from all insights
        for analysis_type, results in self.insights.items():
            if 'insights' in results:
                summary['key_findings'].extend([
                    {'analysis': analysis_type, 'finding': insight} 
                    for insight in results['insights']
                ])
        
        # Extract critical issues
        if 'data_quality_report' in self.insights:
            quality_report = self.insights['data_quality_report']
            for rec in quality_report.get('recommendations', []):
                if rec.get('priority') == 'High':
                    summary['critical_issues'].append(rec)
        
        # Consolidate recommendations
        all_recommendations = []
        for analysis_type, results in self.insights.items():
            if 'recommendations' in results:
                recs = results['recommendations']
                if isinstance(recs, list):
                    all_recommendations.extend(recs)
        
        summary['recommendations'] = all_recommendations[:10]  # Top 10 recommendations
        
        # Generate next steps
        next_steps = [
            "Review and validate critical data quality issues",
            "Implement recommended data cleaning procedures",
            "Investigate strong correlations for business impact",
            "Address high variability metrics for consistency improvement",
            "Consider advanced analytics based on discovered patterns"
        ]
        
        summary['next_steps'] = next_steps
        
        return summary
    
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
    
    def _identify_distribution_pattern(self, data: pd.Series) -> str:
        """Identify distribution pattern."""
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "normal"
        elif abs(skewness) > 2:
            return "highly_skewed"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        else:
            return "unknown"
    
    def _has_significant_outliers(self, data: pd.Series, threshold: float = 0.05) -> bool:
        """Check if data has significant outliers."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return len(outliers) / len(data) > threshold
    
    def _detect_trend(self, data: pd.Series) -> str:
        """Detect trend in time series data."""
        if len(data) < 3:
            return "insufficient_data"
        
        # Simple trend detection using linear regression slope
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        if abs(slope) < data.std() * 0.01:
            return "no_trend"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _detect_seasonality(self, data: pd.Series) -> str:
        """Detect seasonality in time series data."""
        # Simplified seasonality detection
        if len(data) < 24:  # Need sufficient data
            return "insufficient_data"
        
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        return "unknown"
    
    def _analyze_temporal_patterns(self, date_data: pd.Series, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze temporal patterns in data."""
        patterns = {
            'date_range': {
                'start': date_data.min().isoformat(),
                'end': date_data.max().isoformat(),
                'span_days': (date_data.max() - date_data.min()).days
            },
            'frequency': 'unknown',
            'gaps': []
        }
        
        # Detect frequency
        if len(date_data) > 1:
            diff = date_data.diff().dropna()
            mode_diff = diff.mode()
            if not mode_diff.empty:
                common_freq = mode_diff.iloc[0]
                if common_freq == pd.Timedelta(days=1):
                    patterns['frequency'] = 'daily'
                elif common_freq == pd.Timedelta(days=7):
                    patterns['frequency'] = 'weekly'
                elif common_freq == pd.Timedelta(days=30):
                    patterns['frequency'] = 'monthly'
        
        return patterns
    
    def _generate_domain_specific_insights(self, domain: str, goal: str, insights: Dict[str, Any]) -> List[str]:
        """Generate domain-specific actionable insights."""
        actionable_insights = []
        
        if domain.lower() == 'e-commerce':
            actionable_insights.extend([
                "Focus on reducing cart abandonment by analyzing user behavior patterns",
                "Optimize pricing strategy based on demand variability",
                "Implement personalization based on customer segmentation"
            ])
        elif domain.lower() == 'finance':
            actionable_insights.extend([
                "Monitor risk indicators and implement early warning systems",
                "Optimize portfolio allocation based on correlation analysis",
                "Implement fraud detection for outlier transactions"
            ])
        elif domain.lower() == 'marketing':
            actionable_insights.extend([
                "Optimize campaign targeting based on demographic patterns",
                "Improve conversion rates by addressing data quality issues",
                "Implement A/B testing for high-variability metrics"
            ])
        else:
            actionable_insights.extend([
                "Prioritize data quality improvements for better decision making",
                "Investigate strong correlations for operational improvements",
                "Address high variability to improve consistency"
            ])
        
        return actionable_insights[:5]  # Return top 5 insights