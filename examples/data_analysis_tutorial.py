"""
Data Analysis Tutorial
=====================

Complete tutorial demonstrating the AD Data Analysis Framework capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from ad_analysis import DataProcessor, StatisticalAnalyzer, DataVisualizer, DataInterpreter, DataLoader, DataValidator

print("AD Data Analysis Framework Tutorial")
print("=" * 50)

# Step 1: Load Sample Data
print("\n1. Loading Sample Data")
print("-" * 30)

loader = DataLoader()

# Generate different types of sample data
print("Generating e-commerce sample data...")
ecommerce_config = {
    'type': 'sample',
    'sample_type': 'ecommerce',
    'n_rows': 1000
}
data = loader.load_data(ecommerce_config)

print(f"✓ Data loaded: {data.shape}")
print("\nFirst few rows:")
print(data.head())

print("\nData types:")
print(data.dtypes)

# Step 2: Data Validation
print("\n\n2. Data Validation")
print("-" * 30)

validator = DataValidator(data)

# Run comprehensive validation
validation_results = validator.run_all_validations(
    missing_values={'max_missing_percent': 10.0},
    duplicates={'max_duplicate_percent': 5.0}
)

print(f"Overall validation passed: {validation_results['overall_passed']}")
print(f"Total issues found: {validation_results['summary']['total_issues']}")

if validation_results['summary']['total_issues'] > 0:
    print("\nValidation issues:")
    for validation_name, results in validation_results['validations'].items():
        if results.get('issues'):
            print(f"\n{validation_name}:")
            for issue in results['issues'][:2]:  # Show first 2 issues
                print(f"  - {issue.get('issue', 'Unknown issue')}")

# Step 3: Data Processing
print("\n\n3. Data Processing")
print("-" * 30)

processor = DataProcessor(data)

# Get data summary before processing
summary_before = processor.get_data_summary()
print(f"Before processing: {summary_before['shape']}")

# Clean missing values
processor.clean_missing_data(strategy='mean')

# Handle outliers in numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    processor.remove_outliers(columns=numeric_cols[:2])  # Process first 2 numeric columns

# Get summary after processing
summary_after = processor.get_data_summary()
print(f"After processing: {summary_after['shape']}")

print("\nProcessing log:")
for step in processor.processing_log:
    print(f"  - {step}")

# Step 4: Statistical Analysis
print("\n\n4. Statistical Analysis")
print("-" * 30)

analyzer = StatisticalAnalyzer(processor.data)

# Descriptive statistics
desc_stats = analyzer.descriptive_statistics()
print(f"✓ Descriptive statistics computed for {len(desc_stats)} columns")

# Show some key statistics
if 'order_value' in desc_stats:
    order_stats = desc_stats['order_value']
    print(f"\nOrder Value Statistics:")
    print(f"  Mean: ${order_stats['mean']:.2f}")
    print(f"  Median: ${order_stats['median']:.2f}")
    print(f"  Std: ${order_stats['std']:.2f}")
    print(f"  Range: ${order_stats['range']:.2f}")

# Correlation analysis
if len(numeric_cols) > 1:
    corr_analysis = analyzer.correlation_analysis(threshold=0.3)
    strong_correlations = corr_analysis['strong_correlations']
    print(f"\n✓ Found {len(strong_correlations)} strong correlations")
    
    if strong_correlations:
        print("Strongest correlations:")
        for corr in strong_correlations[:3]:
            print(f"  - {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f} ({corr['strength']})")

# Distribution analysis
dist_analysis = analyzer.distribution_analysis()
print(f"\n✓ Distribution analysis completed for {len(dist_analysis)} columns")

# Step 5: Data Visualization
print("\n\n5. Data Visualization")
print("-" * 30)

visualizer = DataVisualizer(processor.data)

try:
    # Distribution plots
    if numeric_cols:
        dist_plot = visualizer.distribution_plots(columns=numeric_cols[:3], plot_type='histogram')
        print("✓ Distribution plots created")
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        corr_plot = visualizer.correlation_heatmap()
        print("✓ Correlation heatmap created")
    
    # Categorical plots
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols and numeric_cols:
        cat_plot = visualizer.categorical_plots(
            categorical_column=categorical_cols[0],
            numeric_column=numeric_cols[0] if numeric_cols else None,
            plot_type='box'
        )
        print("✓ Categorical plots created")
    
    print(f"Available figures: {visualizer.get_figure_list()}")

except Exception as e:
    print(f"Note: Visualization requires display capability: {e}")

# Step 6: Data Interpretation
print("\n\n6. Data Interpretation")
print("-" * 30)

interpreter = DataInterpreter(processor.data)

# Data quality report
quality_report = interpreter.generate_data_quality_report()
print("✓ Data quality report generated")

print(f"\nData Quality Summary:")
print(f"  Total rows: {quality_report['overview']['total_rows']}")
print(f"  Total columns: {quality_report['overview']['total_columns']}")
print(f"  Missing values: {quality_report['missing_data']['total_missing_values']}")
print(f"  Duplicate rows: {quality_report['duplicates']['duplicate_rows']}")

# Relationship analysis
relationship_analysis = interpreter.analyze_relationships(
    target_column='order_value' if 'order_value' in processor.data.columns else None
)
print("\n✓ Relationship analysis completed")

insights = relationship_analysis.get('insights', [])
if insights:
    print("Key relationship insights:")
    for insight in insights[:3]:
        print(f"  - {insight}")

# Pattern detection
pattern_analysis = interpreter.detect_patterns_and_trends()
print("\n✓ Pattern analysis completed")

pattern_insights = pattern_analysis.get('insights', [])
if pattern_insights:
    print("Pattern insights:")
    for insight in pattern_insights[:3]:
        print(f"  - {insight}")

# Business insights
business_context = {
    'domain': 'e-commerce',
    'goal': 'increase_sales'
}

business_insights = interpreter.generate_business_insights(context=business_context)
print("\n✓ Business insights generated")

recommendations = business_insights.get('recommendations', [])
if recommendations:
    print("Business recommendations:")
    for rec in recommendations[:3]:
        print(f"  - {rec['recommendation']} (Expected impact: {rec['expected_impact']})")

# Summary report
summary_report = interpreter.generate_summary_report()
print("\n✓ Summary report created")

print(f"\nReport Summary:")
print(f"  Key findings: {len(summary_report['key_findings'])}")
print(f"  Critical issues: {len(summary_report['critical_issues'])}")
print(f"  Recommendations: {len(summary_report['recommendations'])}")

# Step 7: Advanced Analysis Example
print("\n\n7. Advanced Analysis Example")
print("-" * 30)

try:
    # Hypothesis testing example
    if 'order_value' in processor.data.columns and len(processor.data) > 30:
        # Test if average order value is different from $100
        ttest_result = analyzer.hypothesis_testing(
            test_type='ttest_1samp',
            column='order_value',
            pop_mean=100
        )
        
        print("Hypothesis Test Results:")
        print(f"  Test: One-sample t-test")
        print(f"  H0: Mean order value = $100")
        print(f"  Sample mean: ${ttest_result['sample_mean']:.2f}")
        print(f"  p-value: {ttest_result['p_value']:.4f}")
        print(f"  Interpretation: {ttest_result['interpretation']}")

except Exception as e:
    print(f"Advanced analysis note: {e}")

# Final Summary
print("\n\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)

print("\nFramework Components Demonstrated:")
print("✓ DataLoader - Load data from various sources")
print("✓ DataValidator - Validate data quality")
print("✓ DataProcessor - Clean and transform data")
print("✓ StatisticalAnalyzer - Perform statistical analysis")
print("✓ DataVisualizer - Create visualizations")
print("✓ DataInterpreter - Generate insights and recommendations")

print(f"\nFinal dataset: {processor.data.shape}")
print(f"Analysis completed successfully!")

print("\n" + "=" * 50)
print("Try running this with different sample data types:")
print("- 'mixed': General mixed data types")
print("- 'timeseries': Time series data")
print("- 'financial': Financial transaction data")
print("- 'ecommerce': E-commerce data (used in this example)")
print("=" * 50)