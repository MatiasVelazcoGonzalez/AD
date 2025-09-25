#!/usr/bin/env python3
"""
AD Data Analysis Framework - Quick Demo
=======================================

This demo shows the core functionality of the framework.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ad_analysis import DataProcessor, StatisticalAnalyzer, DataInterpreter, DataLoader, DataValidator

def main():
    print("🚀 AD Data Analysis Framework - Quick Demo")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating Sample Data")
    loader = DataLoader()
    data = loader.load_data({
        'type': 'sample',
        'sample_type': 'ecommerce',
        'n_rows': 500
    })
    print(f"✓ Generated e-commerce dataset: {data.shape}")
    print(f"  Columns: {', '.join(data.columns)}")
    
    # Step 2: Validate data quality
    print("\n🔍 Step 2: Data Quality Validation")
    validator = DataValidator(data)
    validation_results = validator.run_all_validations()
    print(f"✓ Validation completed: {validation_results['summary']['overall_data_quality']}")
    print(f"  Issues found: {validation_results['summary']['total_issues']}")
    
    # Step 3: Process and clean data
    print("\n🧹 Step 3: Data Processing")
    processor = DataProcessor(data)
    processor.clean_missing_data(strategy='mean')
    processor.remove_outliers(method='iqr')
    print(f"✓ Data processed: {processor.data.shape}")
    print(f"  Processing steps: {len(processor.processing_log)}")
    
    # Step 4: Statistical Analysis
    print("\n📈 Step 4: Statistical Analysis")
    analyzer = StatisticalAnalyzer(processor.data)
    
    # Descriptive statistics
    stats = analyzer.descriptive_statistics()
    print(f"✓ Descriptive statistics for {len(stats)} numeric columns")
    
    # Correlation analysis
    correlations = analyzer.correlation_analysis(threshold=0.3)
    strong_corr = len(correlations['strong_correlations'])
    print(f"✓ Found {strong_corr} strong correlations")
    
    # Distribution analysis
    distributions = analyzer.distribution_analysis()
    print(f"✓ Distribution analysis completed")
    
    # Step 5: Generate Insights
    print("\n💡 Step 5: Automated Insights Generation")
    interpreter = DataInterpreter(processor.data)
    
    # Data quality report
    quality_report = interpreter.generate_data_quality_report()
    recommendations = len(quality_report['recommendations'])
    print(f"✓ Data quality report: {recommendations} recommendations")
    
    # Business insights
    business_insights = interpreter.generate_business_insights(context={
        'domain': 'e-commerce',
        'goal': 'increase_sales'
    })
    business_recs = len(business_insights['recommendations'])
    print(f"✓ Business insights: {business_recs} strategic recommendations")
    
    # Summary report
    summary = interpreter.generate_summary_report()
    key_findings = len(summary['key_findings'])
    print(f"✓ Summary report: {key_findings} key findings")
    
    # Step 6: Show Key Results
    print("\n📋 Step 6: Key Results Summary")
    print("-" * 40)
    
    # Show some key statistics
    if 'order_value' in stats:
        order_stats = stats['order_value']
        print(f"📊 Order Value Analysis:")
        print(f"   Average: ${order_stats['mean']:.2f}")
        print(f"   Median:  ${order_stats['median']:.2f}")
        print(f"   Range:   ${order_stats['range']:.2f}")
    
    # Show correlations if any
    if strong_corr > 0:
        print(f"\n🔗 Strong Correlations:")
        for corr in correlations['strong_correlations'][:3]:
            print(f"   {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f}")
    
    # Show top recommendations
    if quality_report['recommendations']:
        print(f"\n💡 Top Data Quality Recommendations:")
        for rec in quality_report['recommendations'][:2]:
            print(f"   • {rec['recommendation']}")
    
    if business_insights['recommendations']:
        print(f"\n🎯 Top Business Recommendations:")
        for rec in business_insights['recommendations'][:2]:
            print(f"   • {rec['recommendation']}")
    
    print("\n" + "=" * 60)
    print("✨ Demo completed successfully!")
    print("The AD Data Analysis Framework is ready for production use.")
    print("=" * 60)

if __name__ == "__main__":
    main()
