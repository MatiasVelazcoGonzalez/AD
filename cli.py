#!/usr/bin/env python3
"""
AD Data Analysis CLI
===================

Command-line interface for the AD data analysis framework.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd

from ad_analysis import DataProcessor, StatisticalAnalyzer, DataVisualizer, DataInterpreter, DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="AD Data Analysis Framework - Comprehensive data analysis and interpretation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load and preview data')
    load_parser.add_argument('file', help='Path to data file')
    load_parser.add_argument('--preview', '-p', action='store_true', help='Show data preview')
    load_parser.add_argument('--info', '-i', action='store_true', help='Show file information')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process and clean data')
    process_parser.add_argument('file', help='Path to data file')
    process_parser.add_argument('--output', '-o', help='Output file path')
    process_parser.add_argument('--missing', choices=['drop', 'mean', 'median', 'mode'], 
                               default='drop', help='Missing value strategy')
    process_parser.add_argument('--outliers', action='store_true', help='Remove outliers')
    process_parser.add_argument('--normalize', choices=['minmax', 'standard', 'robust'], 
                               help='Normalization method')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Perform statistical analysis')
    analyze_parser.add_argument('file', help='Path to data file')
    analyze_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    analyze_parser.add_argument('--correlations', action='store_true', help='Include correlation analysis')
    analyze_parser.add_argument('--distributions', action='store_true', help='Include distribution analysis')
    analyze_parser.add_argument('--outliers', action='store_true', help='Include outlier detection')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Create data visualizations')
    visualize_parser.add_argument('file', help='Path to data file')
    visualize_parser.add_argument('--output-dir', '-o', default='./plots', help='Output directory for plots')
    visualize_parser.add_argument('--plot-types', nargs='+', 
                                 choices=['histogram', 'correlation', 'scatter', 'box'],
                                 default=['histogram', 'correlation'], help='Types of plots to create')
    
    # Interpret command
    interpret_parser = subparsers.add_parser('interpret', help='Generate data insights')
    interpret_parser.add_argument('file', help='Path to data file')
    interpret_parser.add_argument('--output', '-o', help='Output file for insights (JSON)')
    interpret_parser.add_argument('--context', help='Business context (JSON file)')
    interpret_parser.add_argument('--target', help='Target column for focused analysis')
    
    # Generate sample data command
    sample_parser = subparsers.add_parser('sample', help='Generate sample data')
    sample_parser.add_argument('--type', choices=['mixed', 'timeseries', 'ecommerce', 'financial'],
                              default='mixed', help='Type of sample data')
    sample_parser.add_argument('--rows', '-n', type=int, default=1000, help='Number of rows')
    sample_parser.add_argument('--output', '-o', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'load':
            handle_load_command(args)
        elif args.command == 'process':
            handle_process_command(args)
        elif args.command == 'analyze':
            handle_analyze_command(args)
        elif args.command == 'visualize':
            handle_visualize_command(args)
        elif args.command == 'interpret':
            handle_interpret_command(args)
        elif args.command == 'sample':
            handle_sample_command(args)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def handle_load_command(args):
    """Handle the load command."""
    loader = DataLoader()
    
    if args.info:
        info = loader.get_data_info(args.file)
        print("File Information:")
        print("-" * 20)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    if args.preview:
        print("\nData Preview:")
        print("-" * 20)
        preview = loader.preview_data(args.file)
        print(preview.to_string())
        print(f"\nShape: {preview.shape}")


def handle_process_command(args):
    """Handle the process command."""
    # Load data
    loader = DataLoader()
    data = loader.load_data(args.file)
    print(f"Loaded data with shape: {data.shape}")
    
    # Process data
    processor = DataProcessor(data)
    
    # Apply cleaning steps
    processor.clean_missing_data(strategy=args.missing)
    print(f"After missing value handling: {processor.data.shape}")
    
    if args.outliers:
        processor.remove_outliers()
        print(f"After outlier removal: {processor.data.shape}")
    
    if args.normalize:
        processor.normalize_data(method=args.normalize)
        print("Data normalized")
    
    # Save processed data
    output_file = args.output or args.file.replace('.csv', '_processed.csv')
    processor.data.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")
    
    # Print processing log
    print("\nProcessing Log:")
    for step in processor.processing_log:
        print(f"- {step}")


def handle_analyze_command(args):
    """Handle the analyze command."""
    # Load data
    loader = DataLoader()
    data = loader.load_data(args.file)
    print(f"Analyzing data with shape: {data.shape}")
    
    analyzer = StatisticalAnalyzer(data)
    results = {}
    
    # Basic descriptive statistics
    desc_stats = analyzer.descriptive_statistics()
    results['descriptive_statistics'] = desc_stats
    print("✓ Descriptive statistics computed")
    
    if args.correlations:
        corr_analysis = analyzer.correlation_analysis()
        results['correlation_analysis'] = corr_analysis
        print("✓ Correlation analysis completed")
    
    if args.distributions:
        dist_analysis = analyzer.distribution_analysis()
        results['distribution_analysis'] = dist_analysis
        print("✓ Distribution analysis completed")
    
    if args.outliers:
        outlier_analysis = analyzer.outlier_detection()
        results['outlier_detection'] = outlier_analysis
        print("✓ Outlier detection completed")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")
    else:
        print("\nAnalysis Results Summary:")
        print(f"- Numeric columns analyzed: {len(desc_stats)}")
        if args.correlations and 'strong_correlations' in results.get('correlation_analysis', {}):
            strong_corr = len(results['correlation_analysis']['strong_correlations'])
            print(f"- Strong correlations found: {strong_corr}")


def handle_visualize_command(args):
    """Handle the visualize command."""
    # Load data
    loader = DataLoader()
    data = loader.load_data(args.file)
    print(f"Creating visualizations for data with shape: {data.shape}")
    
    visualizer = DataVisualizer(data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for plot_type in args.plot_types:
        try:
            if plot_type == 'histogram':
                result = visualizer.distribution_plots(plot_type='histogram')
                output_file = output_dir / 'histograms.png'
                visualizer.save_figure('distribution_histogram', str(output_file))
                print(f"✓ Histogram saved to: {output_file}")
            
            elif plot_type == 'correlation':
                result = visualizer.correlation_heatmap()
                output_file = output_dir / 'correlation_heatmap.png'
                visualizer.save_figure('correlation_heatmap', str(output_file))
                print(f"✓ Correlation heatmap saved to: {output_file}")
            
            elif plot_type == 'scatter':
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2:
                    result = visualizer.scatter_plot_matrix(columns=numeric_cols[:4])  # Limit to 4 columns
                    output_file = output_dir / 'scatter_matrix.png'
                    visualizer.save_figure('scatter_matrix', str(output_file))
                    print(f"✓ Scatter plot matrix saved to: {output_file}")
            
            elif plot_type == 'box':
                result = visualizer.distribution_plots(plot_type='box')
                output_file = output_dir / 'box_plots.png'
                visualizer.save_figure('distribution_box', str(output_file))
                print(f"✓ Box plots saved to: {output_file}")
        
        except Exception as e:
            print(f"Warning: Could not create {plot_type} plot: {e}")


def handle_interpret_command(args):
    """Handle the interpret command."""
    # Load data
    loader = DataLoader()
    data = loader.load_data(args.file)
    print(f"Interpreting data with shape: {data.shape}")
    
    interpreter = DataInterpreter(data)
    
    # Load business context if provided
    context = None
    if args.context:
        with open(args.context, 'r') as f:
            context = json.load(f)
    
    # Generate insights
    quality_report = interpreter.generate_data_quality_report()
    print("✓ Data quality report generated")
    
    relationship_analysis = interpreter.analyze_relationships(target_column=args.target)
    print("✓ Relationship analysis completed")
    
    pattern_analysis = interpreter.detect_patterns_and_trends()
    print("✓ Pattern analysis completed")
    
    business_insights = interpreter.generate_business_insights(context=context)
    print("✓ Business insights generated")
    
    summary_report = interpreter.generate_summary_report()
    print("✓ Summary report created")
    
    # Compile all results
    all_insights = {
        'data_quality_report': quality_report,
        'relationship_analysis': relationship_analysis,
        'pattern_analysis': pattern_analysis,
        'business_insights': business_insights,
        'summary_report': summary_report
    }
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_insights, f, indent=2, default=str)
        print(f"Insights saved to: {args.output}")
    else:
        # Print key insights
        print("\nKey Insights:")
        print("-" * 30)
        for finding in summary_report.get('key_findings', [])[:5]:
            print(f"- {finding.get('finding', '')}")
        
        critical_issues = summary_report.get('critical_issues', [])
        if critical_issues:
            print(f"\nCritical Issues ({len(critical_issues)}):")
            for issue in critical_issues[:3]:
                print(f"- {issue.get('issue', '')}")


def handle_sample_command(args):
    """Handle the sample data generation command."""
    loader = DataLoader()
    
    config = {
        'type': 'sample',
        'sample_type': args.type,
        'n_rows': args.rows
    }
    
    data = loader.load_data(config)
    data.to_csv(args.output, index=False)
    
    print(f"Generated {args.type} sample data:")
    print(f"- Rows: {len(data)}")
    print(f"- Columns: {len(data.columns)}")
    print(f"- Saved to: {args.output}")
    
    print("\nColumn Summary:")
    for col in data.columns:
        dtype = data[col].dtype
        print(f"- {col}: {dtype}")


if __name__ == '__main__':
    sys.exit(main())