"""
Basic functionality tests for the AD Data Analysis Framework
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

from ad_analysis import DataProcessor, StatisticalAnalyzer, DataVisualizer, DataInterpreter, DataLoader, DataValidator


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality."""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_sample_data_generation(self):
        """Test sample data generation."""
        config = {
            'type': 'sample',
            'sample_type': 'mixed',
            'n_rows': 100
        }
        
        data = self.loader.load_data(config)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertGreater(len(data.columns), 0)
    
    def test_different_sample_types(self):
        """Test different sample data types."""
        sample_types = ['mixed', 'timeseries', 'ecommerce', 'financial']
        
        for sample_type in sample_types:
            config = {
                'type': 'sample',
                'sample_type': sample_type,
                'n_rows': 50
            }
            
            data = self.loader.load_data(config)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 50)


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor functionality."""
    
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        self.data = pd.DataFrame({
            'numeric1': np.random.normal(100, 15, 100),
            'numeric2': np.random.uniform(0, 100, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'with_missing': np.random.normal(50, 10, 100)
        })
        
        # Add some missing values
        self.data.loc[::10, 'with_missing'] = np.nan
        
        self.processor = DataProcessor(self.data)
    
    def test_data_loading(self):
        """Test data loading."""
        self.assertIsNotNone(self.processor.data)
        self.assertEqual(len(self.processor.data), 100)
        self.assertEqual(len(self.processor.processing_log), 1)
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        original_missing = self.processor.data['with_missing'].isna().sum()
        self.assertGreater(original_missing, 0)
        
        self.processor.clean_missing_data(strategy='mean')
        new_missing = self.processor.data['with_missing'].isna().sum()
        self.assertEqual(new_missing, 0)
    
    def test_outlier_removal(self):
        """Test outlier removal."""
        original_shape = self.processor.data.shape
        self.processor.remove_outliers(method='iqr')
        new_shape = self.processor.data.shape
        
        # Should have same number of columns, possibly fewer rows
        self.assertEqual(new_shape[1], original_shape[1])
        self.assertLessEqual(new_shape[0], original_shape[0])
    
    def test_normalization(self):
        """Test data normalization."""
        original_mean = self.processor.data['numeric1'].mean()
        self.processor.normalize_data(method='standard')
        new_mean = self.processor.data['numeric1'].mean()
        
        # Standard normalization should make mean close to 0
        self.assertAlmostEqual(new_mean, 0, places=10)
    
    def test_method_chaining(self):
        """Test method chaining."""
        result = (self.processor
                 .clean_missing_data(strategy='mean')
                 .normalize_data(method='minmax'))
        
        self.assertIsInstance(result, DataProcessor)
        self.assertGreater(len(self.processor.processing_log), 2)


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test StatisticalAnalyzer functionality."""
    
    def setUp(self):
        np.random.seed(42)
        self.data = pd.DataFrame({
            'numeric1': np.random.normal(100, 15, 100),
            'numeric2': np.random.uniform(0, 100, 100),
            'numeric3': np.random.exponential(2, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        self.analyzer = StatisticalAnalyzer(self.data)
    
    def test_descriptive_statistics(self):
        """Test descriptive statistics calculation."""
        stats = self.analyzer.descriptive_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('numeric1', stats)
        self.assertIn('mean', stats['numeric1'])
        self.assertIn('std', stats['numeric1'])
        self.assertIn('skewness', stats['numeric1'])
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        corr_result = self.analyzer.correlation_analysis(threshold=0.1)
        
        self.assertIsInstance(corr_result, dict)
        self.assertIn('correlation_matrix', corr_result)
        self.assertIn('strong_correlations', corr_result)
        self.assertIn('method', corr_result)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        outlier_result = self.analyzer.outlier_detection(method='iqr')
        
        self.assertIsInstance(outlier_result, dict)
        self.assertIn('outliers_by_column', outlier_result)
        self.assertIn('method', outlier_result)
    
    def test_distribution_analysis(self):
        """Test distribution analysis."""
        dist_result = self.analyzer.distribution_analysis()
        
        self.assertIsInstance(dist_result, dict)
        self.assertIn('numeric1', dist_result)
        self.assertIn('distribution_type', dist_result['numeric1'])
    
    def test_hypothesis_testing(self):
        """Test hypothesis testing."""
        # One-sample t-test
        ttest_result = self.analyzer.hypothesis_testing(
            test_type='ttest_1samp',
            column='numeric1',
            pop_mean=100
        )
        
        self.assertIsInstance(ttest_result, dict)
        self.assertIn('statistic', ttest_result)
        self.assertIn('p_value', ttest_result)
        self.assertIn('interpretation', ttest_result)


class TestDataValidator(unittest.TestCase):
    """Test DataValidator functionality."""
    
    def setUp(self):
        np.random.seed(42)
        self.data = pd.DataFrame({
            'id': range(1, 101),
            'numeric': np.random.normal(50, 10, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'with_missing': np.random.normal(25, 5, 100)
        })
        
        # Add some issues for testing
        self.data.loc[::20, 'with_missing'] = np.nan  # Missing values
        self.data.loc[95:99, :] = self.data.loc[90:94, :].values  # Duplicates
        
        self.validator = DataValidator(self.data)
    
    def test_data_type_validation(self):
        """Test data type validation."""
        expected_types = {
            'id': 'int',
            'numeric': 'float',
            'category': 'object'
        }
        
        result = self.validator.validate_data_types(expected_types)
        
        self.assertIsInstance(result, dict)
        self.assertIn('passed', result)
        self.assertIn('issues', result)
        self.assertIn('column_types', result)
    
    def test_missing_value_validation(self):
        """Test missing value validation."""
        result = self.validator.validate_missing_values(max_missing_percent=10.0)
        
        self.assertIsInstance(result, dict)
        self.assertIn('missing_stats', result)
        self.assertIn('summary', result)
    
    def test_duplicate_validation(self):
        """Test duplicate validation."""
        result = self.validator.validate_duplicates(max_duplicate_percent=3.0)
        
        self.assertIsInstance(result, dict)
        self.assertIn('duplicate_stats', result)
        # Should detect the duplicates we added
        self.assertGreater(result['duplicate_stats']['duplicate_count'], 0)
    
    def test_range_validation(self):
        """Test range validation."""
        range_rules = {
            'numeric': {'min': 0, 'max': 100}
        }
        
        result = self.validator.validate_ranges(range_rules)
        
        self.assertIsInstance(result, dict)
        self.assertIn('range_stats', result)
        self.assertIn('summary', result)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation."""
        result = self.validator.run_all_validations()
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_passed', result)
        self.assertIn('validations', result)
        self.assertIn('summary', result)


class TestDataInterpreter(unittest.TestCase):
    """Test DataInterpreter functionality."""
    
    def setUp(self):
        np.random.seed(42)
        self.data = pd.DataFrame({
            'sales': np.random.lognormal(4, 0.5, 100),
            'customers': np.random.randint(1, 100, 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'product_type': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        self.interpreter = DataInterpreter(self.data)
    
    def test_data_quality_report(self):
        """Test data quality report generation."""
        report = self.interpreter.generate_data_quality_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('overview', report)
        self.assertIn('missing_data', report)
        self.assertIn('recommendations', report)
    
    def test_relationship_analysis(self):
        """Test relationship analysis."""
        analysis = self.interpreter.analyze_relationships(target_column='sales')
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('correlation_matrix', analysis)
        self.assertIn('target_analysis', analysis)
    
    def test_pattern_detection(self):
        """Test pattern detection."""
        patterns = self.interpreter.detect_patterns_and_trends()
        
        self.assertIsInstance(patterns, dict)
        self.assertIn('categorical_patterns', patterns)
        self.assertIn('numeric_patterns', patterns)
        self.assertIn('insights', patterns)
    
    def test_business_insights(self):
        """Test business insights generation."""
        context = {'domain': 'sales', 'goal': 'increase_revenue'}
        insights = self.interpreter.generate_business_insights(context=context)
        
        self.assertIsInstance(insights, dict)
        self.assertIn('key_metrics', insights)
        self.assertIn('recommendations', insights)
    
    def test_summary_report(self):
        """Test summary report generation."""
        # Generate some insights first
        self.interpreter.generate_data_quality_report()
        self.interpreter.analyze_relationships()
        
        summary = self.interpreter.generate_summary_report()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('report_timestamp', summary)
        self.assertIn('key_findings', summary)
        self.assertIn('next_steps', summary)


if __name__ == '__main__':
    print("Running AD Data Analysis Framework Tests")
    print("=" * 50)
    
    # Run all tests
    unittest.main(verbosity=2)