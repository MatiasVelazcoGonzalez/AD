"""
Data Validation Utilities
=========================

Utilities for validating data quality and integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
from datetime import datetime


class DataValidator:
    """
    Comprehensive data validation class for ensuring data quality.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataValidator.
        
        Args:
            data: DataFrame to validate
        """
        self.data = data
        self.validation_results = {}
        self.validation_rules = {}
    
    def set_data(self, data: pd.DataFrame) -> 'DataValidator':
        """
        Set the data for validation.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Self for method chaining
        """
        self.data = data
        return self
    
    def add_validation_rule(self, rule_name: str, column: str, rule_type: str, 
                          parameters: Dict[str, Any]) -> 'DataValidator':
        """
        Add a custom validation rule.
        
        Args:
            rule_name: Name for the validation rule
            column: Column to validate
            rule_type: Type of validation ('range', 'values', 'pattern', 'custom')
            parameters: Parameters for the validation rule
        
        Returns:
            Self for method chaining
        """
        self.validation_rules[rule_name] = {
            'column': column,
            'rule_type': rule_type,
            'parameters': parameters
        }
        return self
    
    def validate_data_types(self, expected_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Validate data types of columns.
        
        Args:
            expected_types: Dictionary mapping column names to expected types
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        results = {
            'passed': True,
            'issues': [],
            'summary': {},
            'column_types': {}
        }
        
        # Get current data types
        current_types = self.data.dtypes.to_dict()
        results['column_types'] = {col: str(dtype) for col, dtype in current_types.items()}
        
        if expected_types:
            for col, expected_type in expected_types.items():
                if col not in self.data.columns:
                    results['issues'].append({
                        'column': col,
                        'issue': 'Column not found',
                        'severity': 'High'
                    })
                    results['passed'] = False
                    continue
                
                current_type = str(current_types[col])
                if expected_type not in current_type:
                    results['issues'].append({
                        'column': col,
                        'issue': f'Expected {expected_type}, found {current_type}',
                        'severity': 'Medium'
                    })
                    results['passed'] = False
        
        # Auto-detect potential type issues
        for col in self.data.columns:
            col_data = self.data[col]
            
            # Check for mixed types in object columns
            if col_data.dtype == 'object':
                sample_types = set(type(val).__name__ for val in col_data.dropna().head(100))
                if len(sample_types) > 1:
                    results['issues'].append({
                        'column': col,
                        'issue': f'Mixed data types detected: {sample_types}',
                        'severity': 'Medium'
                    })
        
        results['summary'] = {
            'total_columns': len(self.data.columns),
            'issues_found': len(results['issues']),
            'validation_passed': results['passed']
        }
        
        self.validation_results['data_types'] = results
        return results
    
    def validate_missing_values(self, max_missing_percent: float = 50.0, 
                              critical_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate missing value constraints.
        
        Args:
            max_missing_percent: Maximum allowed missing value percentage
            critical_columns: Columns that cannot have missing values
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        results = {
            'passed': True,
            'issues': [],
            'summary': {},
            'missing_stats': {}
        }
        
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100
        
        results['missing_stats'] = {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict()
        }
        
        # Check maximum missing percentage
        for col, percentage in missing_percentages.items():
            if percentage > max_missing_percent:
                results['issues'].append({
                    'column': col,
                    'issue': f'Missing values: {percentage:.1f}% (max allowed: {max_missing_percent}%)',
                    'severity': 'High' if percentage > 80 else 'Medium'
                })
                results['passed'] = False
        
        # Check critical columns
        if critical_columns:
            for col in critical_columns:
                if col not in self.data.columns:
                    results['issues'].append({
                        'column': col,
                        'issue': 'Critical column not found',
                        'severity': 'High'
                    })
                    results['passed'] = False
                elif missing_counts[col] > 0:
                    results['issues'].append({
                        'column': col,
                        'issue': f'Critical column has {missing_counts[col]} missing values',
                        'severity': 'High'
                    })
                    results['passed'] = False
        
        results['summary'] = {
            'total_missing_values': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'max_missing_percentage': missing_percentages.max(),
            'validation_passed': results['passed']
        }
        
        self.validation_results['missing_values'] = results
        return results
    
    def validate_duplicates(self, subset: Optional[List[str]] = None, 
                          max_duplicate_percent: float = 10.0) -> Dict[str, Any]:
        """
        Validate duplicate record constraints.
        
        Args:
            subset: Columns to consider for duplicate detection
            max_duplicate_percent: Maximum allowed duplicate percentage
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        results = {
            'passed': True,
            'issues': [],
            'summary': {},
            'duplicate_stats': {}
        }
        
        # Check for duplicate rows
        duplicate_mask = self.data.duplicated(subset=subset, keep=False)
        duplicate_count = duplicate_mask.sum()
        duplicate_percentage = (duplicate_count / len(self.data)) * 100
        
        results['duplicate_stats'] = {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': float(duplicate_percentage),
            'unique_count': len(self.data) - int(duplicate_count)
        }
        
        if duplicate_percentage > max_duplicate_percent:
            results['issues'].append({
                'issue': f'Duplicate records: {duplicate_percentage:.1f}% (max allowed: {max_duplicate_percent}%)',
                'severity': 'Medium' if duplicate_percentage < 25 else 'High',
                'affected_rows': int(duplicate_count)
            })
            results['passed'] = False
        
        # Check for duplicate values in potential ID columns
        potential_id_cols = [col for col in self.data.columns 
                           if 'id' in col.lower() or col.lower().endswith('_key')]
        
        for col in potential_id_cols:
            if col in self.data.columns:
                unique_count = self.data[col].nunique()
                total_count = len(self.data.dropna(subset=[col]))
                
                if unique_count < total_count:
                    duplicate_id_count = total_count - unique_count
                    results['issues'].append({
                        'column': col,
                        'issue': f'Potential ID column has {duplicate_id_count} duplicate values',
                        'severity': 'Medium'
                    })
        
        results['summary'] = {
            'total_rows': len(self.data),
            'duplicate_rows': int(duplicate_count),
            'validation_passed': results['passed']
        }
        
        self.validation_results['duplicates'] = results
        return results
    
    def validate_ranges(self, range_rules: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Validate numeric ranges.
        
        Args:
            range_rules: Dictionary mapping column names to {'min': value, 'max': value}
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        results = {
            'passed': True,
            'issues': [],
            'summary': {},
            'range_stats': {}
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Auto-detect potential range issues
        for col in numeric_cols:
            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue
            
            min_val, max_val = col_data.min(), col_data.max()
            results['range_stats'][col] = {
                'min': float(min_val),
                'max': float(max_val),
                'range': float(max_val - min_val)
            }
            
            # Check for negative values in columns that might not expect them
            if col.lower() in ['age', 'price', 'amount', 'quantity', 'count'] and min_val < 0:
                negative_count = (col_data < 0).sum()
                results['issues'].append({
                    'column': col,
                    'issue': f'Negative values found: {negative_count} rows',
                    'severity': 'Medium'
                })
                results['passed'] = False
            
            # Check for extreme outliers
            if len(col_data) > 10:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                extreme_lower = Q1 - 3 * IQR
                extreme_upper = Q3 + 3 * IQR
                
                extreme_outliers = ((col_data < extreme_lower) | (col_data > extreme_upper)).sum()
                if extreme_outliers > len(col_data) * 0.01:  # More than 1% extreme outliers
                    results['issues'].append({
                        'column': col,
                        'issue': f'Extreme outliers detected: {extreme_outliers} rows',
                        'severity': 'Low'
                    })
        
        # Validate custom range rules
        if range_rules:
            for col, rules in range_rules.items():
                if col not in self.data.columns:
                    results['issues'].append({
                        'column': col,
                        'issue': 'Column not found for range validation',
                        'severity': 'High'
                    })
                    results['passed'] = False
                    continue
                
                col_data = self.data[col].dropna()
                
                if 'min' in rules:
                    below_min = (col_data < rules['min']).sum()
                    if below_min > 0:
                        results['issues'].append({
                            'column': col,
                            'issue': f'Values below minimum ({rules["min"]}): {below_min} rows',
                            'severity': 'High'
                        })
                        results['passed'] = False
                
                if 'max' in rules:
                    above_max = (col_data > rules['max']).sum()
                    if above_max > 0:
                        results['issues'].append({
                            'column': col,
                            'issue': f'Values above maximum ({rules["max"]}): {above_max} rows',
                            'severity': 'High'
                        })
                        results['passed'] = False
        
        results['summary'] = {
            'numeric_columns_checked': len(numeric_cols),
            'range_violations': len([issue for issue in results['issues'] if 'range' in issue.get('issue', '').lower()]),
            'validation_passed': results['passed']
        }
        
        self.validation_results['ranges'] = results
        return results
    
    def validate_categorical_values(self, allowed_values: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Validate categorical value constraints.
        
        Args:
            allowed_values: Dictionary mapping column names to lists of allowed values
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        results = {
            'passed': True,
            'issues': [],
            'summary': {},
            'categorical_stats': {}
        }
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        # Analyze categorical columns
        for col in categorical_cols:
            col_data = self.data[col].dropna()
            unique_values = col_data.unique()
            value_counts = col_data.value_counts()
            
            results['categorical_stats'][col] = {
                'unique_count': len(unique_values),
                'most_common': value_counts.head(5).to_dict(),
                'least_common': value_counts.tail(5).to_dict()
            }
            
            # Check for potential data quality issues
            # 1. Too many unique values (might indicate data entry errors)
            if len(unique_values) > len(col_data) * 0.8:
                results['issues'].append({
                    'column': col,
                    'issue': f'High cardinality: {len(unique_values)} unique values in {len(col_data)} records',
                    'severity': 'Low'
                })
            
            # 2. Single character values (might indicate codes that need expansion)
            single_char_values = [val for val in unique_values if isinstance(val, str) and len(val) == 1]
            if len(single_char_values) > 1:
                results['issues'].append({
                    'column': col,
                    'issue': f'Single character values found: {single_char_values}',
                    'severity': 'Low'
                })
            
            # 3. Mixed case variations
            if len(unique_values) > 1:
                lower_case_values = {val.lower() if isinstance(val, str) else val for val in unique_values}
                if len(lower_case_values) < len(unique_values):
                    results['issues'].append({
                        'column': col,
                        'issue': 'Mixed case variations detected (e.g., "Yes" vs "yes")',
                        'severity': 'Medium'
                    })
        
        # Validate against allowed values
        if allowed_values:
            for col, allowed in allowed_values.items():
                if col not in self.data.columns:
                    results['issues'].append({
                        'column': col,
                        'issue': 'Column not found for categorical validation',
                        'severity': 'High'
                    })
                    results['passed'] = False
                    continue
                
                col_data = self.data[col].dropna()
                invalid_values = set(col_data.unique()) - set(allowed)
                
                if invalid_values:
                    invalid_count = col_data.isin(invalid_values).sum()
                    results['issues'].append({
                        'column': col,
                        'issue': f'Invalid values found: {list(invalid_values)} ({invalid_count} rows)',
                        'severity': 'High'
                    })
                    results['passed'] = False
        
        results['summary'] = {
            'categorical_columns_checked': len(categorical_cols),
            'total_unique_values': sum(len(self.data[col].unique()) for col in categorical_cols),
            'validation_passed': results['passed']
        }
        
        self.validation_results['categorical_values'] = results
        return results
    
    def validate_custom_rules(self) -> Dict[str, Any]:
        """
        Validate custom rules defined by add_validation_rule().
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        results = {
            'passed': True,
            'issues': [],
            'summary': {},
            'rule_results': {}
        }
        
        for rule_name, rule_config in self.validation_rules.items():
            column = rule_config['column']
            rule_type = rule_config['rule_type']
            parameters = rule_config['parameters']
            
            rule_result = {'rule_name': rule_name, 'passed': True, 'violations': 0}
            
            if column not in self.data.columns:
                results['issues'].append({
                    'rule': rule_name,
                    'column': column,
                    'issue': 'Column not found',
                    'severity': 'High'
                })
                results['passed'] = False
                rule_result['passed'] = False
                continue
            
            col_data = self.data[column].dropna()
            
            if rule_type == 'range':
                min_val = parameters.get('min')
                max_val = parameters.get('max')
                
                violations = 0
                if min_val is not None:
                    violations += (col_data < min_val).sum()
                if max_val is not None:
                    violations += (col_data > max_val).sum()
                
                if violations > 0:
                    results['issues'].append({
                        'rule': rule_name,
                        'column': column,
                        'issue': f'Range violations: {violations} rows',
                        'severity': parameters.get('severity', 'Medium')
                    })
                    results['passed'] = False
                    rule_result['passed'] = False
                    rule_result['violations'] = violations
            
            elif rule_type == 'values':
                allowed_values = parameters.get('allowed_values', [])
                violations = (~col_data.isin(allowed_values)).sum()
                
                if violations > 0:
                    results['issues'].append({
                        'rule': rule_name,
                        'column': column,
                        'issue': f'Invalid values: {violations} rows',
                        'severity': parameters.get('severity', 'Medium')
                    })
                    results['passed'] = False
                    rule_result['passed'] = False
                    rule_result['violations'] = violations
            
            elif rule_type == 'pattern':
                import re
                pattern = parameters.get('pattern', '')
                violations = 0
                
                for value in col_data:
                    if isinstance(value, str) and not re.match(pattern, value):
                        violations += 1
                
                if violations > 0:
                    results['issues'].append({
                        'rule': rule_name,
                        'column': column,
                        'issue': f'Pattern violations: {violations} rows',
                        'severity': parameters.get('severity', 'Medium')
                    })
                    results['passed'] = False
                    rule_result['passed'] = False
                    rule_result['violations'] = violations
            
            results['rule_results'][rule_name] = rule_result
        
        results['summary'] = {
            'custom_rules_checked': len(self.validation_rules),
            'rules_passed': sum(1 for result in results['rule_results'].values() if result['passed']),
            'validation_passed': results['passed']
        }
        
        self.validation_results['custom_rules'] = results
        return results
    
    def run_all_validations(self, **kwargs) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Args:
            **kwargs: Parameters for specific validation methods
        
        Returns:
            Dictionary with comprehensive validation results
        """
        all_results = {
            'overall_passed': True,
            'validation_timestamp': datetime.now().isoformat(),
            'data_shape': self.data.shape if self.data is not None else None,
            'validations': {},
            'summary': {}
        }
        
        # Run all validation methods
        validations = [
            ('data_types', self.validate_data_types),
            ('missing_values', self.validate_missing_values),
            ('duplicates', self.validate_duplicates),
            ('ranges', self.validate_ranges),
            ('categorical_values', self.validate_categorical_values),
            ('custom_rules', self.validate_custom_rules)
        ]
        
        total_issues = 0
        
        for validation_name, validation_method in validations:
            try:
                method_kwargs = kwargs.get(validation_name, {})
                result = validation_method(**method_kwargs)
                all_results['validations'][validation_name] = result
                
                if not result.get('passed', True):
                    all_results['overall_passed'] = False
                
                total_issues += len(result.get('issues', []))
                
            except Exception as e:
                all_results['validations'][validation_name] = {
                    'error': str(e),
                    'passed': False
                }
                all_results['overall_passed'] = False
        
        # Generate summary
        all_results['summary'] = {
            'total_validations': len(validations),
            'passed_validations': sum(1 for v in all_results['validations'].values() 
                                    if v.get('passed', False)),
            'total_issues': total_issues,
            'high_severity_issues': sum(1 for v in all_results['validations'].values() 
                                      for issue in v.get('issues', []) 
                                      if issue.get('severity') == 'High'),
            'overall_data_quality': 'Good' if all_results['overall_passed'] else 'Issues Found'
        }
        
        return all_results
    
    def get_validation_report(self) -> str:
        """
        Generate a human-readable validation report.
        
        Returns:
            Formatted validation report as string
        """
        if not self.validation_results:
            return "No validation results available. Run validations first."
        
        report = ["Data Validation Report", "=" * 50, ""]
        
        for validation_name, results in self.validation_results.items():
            report.append(f"{validation_name.title().replace('_', ' ')}:")
            report.append("-" * (len(validation_name) + 1))
            
            if results.get('passed', False):
                report.append("✓ Validation PASSED")
            else:
                report.append("✗ Validation FAILED")
            
            # Add summary
            if 'summary' in results:
                for key, value in results['summary'].items():
                    report.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Add issues
            if results.get('issues'):
                report.append("  Issues found:")
                for issue in results['issues'][:5]:  # Show first 5 issues
                    severity = issue.get('severity', 'Unknown')
                    column = issue.get('column', '')
                    description = issue.get('issue', '')
                    report.append(f"    [{severity}] {column}: {description}")
                
                if len(results['issues']) > 5:
                    report.append(f"    ... and {len(results['issues']) - 5} more issues")
            
            report.append("")
        
        return "\n".join(report)