# AD - Advanced Data Analysis Framework

A comprehensive Python framework for data analysis and interpretation, providing powerful tools for data processing, statistical analysis, visualization, and automated insight generation.

## 🚀 Features

### Core Components

- **DataLoader**: Load data from various sources (CSV, Excel, JSON, Parquet) and generate sample datasets
- **DataProcessor**: Clean, transform, and prepare data with advanced preprocessing capabilities
- **StatisticalAnalyzer**: Perform comprehensive statistical analysis including hypothesis testing
- **DataVisualizer**: Create publication-ready static and interactive visualizations
- **DataInterpreter**: Generate automated insights and business recommendations
- **DataValidator**: Validate data quality and integrity with customizable rules

### Key Capabilities

✅ **Data Processing**
- Automated missing value handling (drop, imputation with mean/median/mode)
- Advanced outlier detection and removal (IQR, Z-score, Isolation Forest)
- Data normalization and standardization
- Categorical encoding (one-hot, label encoding)
- Method chaining for streamlined workflows

✅ **Statistical Analysis**
- Comprehensive descriptive statistics
- Correlation analysis with multiple methods (Pearson, Spearman, Kendall)
- Distribution analysis and normality testing
- Hypothesis testing (t-tests, chi-square, ANOVA)
- Outlier detection with multiple algorithms

✅ **Data Visualization**
- Distribution plots (histograms, density, box, violin, Q-Q plots)
- Correlation heatmaps with customizable styling
- Scatter plot matrices and pair plots
- Time series visualizations with resampling
- Categorical analysis plots
- Interactive visualizations with Plotly

✅ **Automated Insights**
- Data quality reports with actionable recommendations
- Relationship analysis and pattern detection
- Business intelligence insights with domain-specific recommendations
- Comprehensive summary reports
- Automated trend detection

✅ **Data Validation**
- Data type validation
- Missing value constraints
- Duplicate detection
- Range validation for numeric data
- Categorical value validation
- Custom validation rules

## 📦 Installation

### Option 1: Direct Installation
```bash
# Clone the repository
git clone https://github.com/MatiasVelazcoGonzalez/AD.git
cd AD

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Requirements Only
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn scipy jupyter statsmodels
```

## 🎯 Quick Start

### Basic Usage

```python
from ad_analysis import DataProcessor, StatisticalAnalyzer, DataVisualizer, DataInterpreter, DataLoader

# 1. Load or generate sample data
loader = DataLoader()
data = loader.load_data({
    'type': 'sample',
    'sample_type': 'ecommerce',
    'n_rows': 1000
})

# 2. Process and clean data
processor = DataProcessor(data)
processor.clean_missing_data(strategy='mean').remove_outliers().normalize_data()

# 3. Perform statistical analysis
analyzer = StatisticalAnalyzer(processor.data)
stats = analyzer.descriptive_statistics()
correlations = analyzer.correlation_analysis()

# 4. Create visualizations
visualizer = DataVisualizer(processor.data)
visualizer.distribution_plots()
visualizer.correlation_heatmap()

# 5. Generate insights
interpreter = DataInterpreter(processor.data)
quality_report = interpreter.generate_data_quality_report()
insights = interpreter.generate_business_insights()
```

### Command Line Interface

The framework includes a powerful CLI for common tasks:

```bash
# Generate sample data
python cli.py sample --type ecommerce --rows 1000 --output sample_data.csv

# Load and preview data
python cli.py load sample_data.csv --preview --info

# Process data
python cli.py process sample_data.csv --missing mean --outliers --normalize standard

# Perform analysis
python cli.py analyze sample_data.csv --correlations --distributions --output results.json

# Create visualizations
python cli.py visualize sample_data.csv --output-dir plots --plot-types histogram correlation scatter

# Generate insights
python cli.py interpret sample_data.csv --context business_context.json --output insights.json
```

## 📊 Example Use Cases

### E-commerce Analysis
```python
# Load e-commerce data
data = loader.load_data({'type': 'sample', 'sample_type': 'ecommerce', 'n_rows': 5000})

# Analyze customer behavior
interpreter = DataInterpreter(data)
insights = interpreter.generate_business_insights(context={
    'domain': 'e-commerce',
    'goal': 'increase_sales'
})

# Key insights might include:
# - Customer segmentation opportunities
# - Seasonal trends in sales
# - Product performance analysis
# - Price optimization recommendations
```

### Financial Data Analysis
```python
# Load financial data
data = loader.load_data({'type': 'sample', 'sample_type': 'financial', 'n_rows': 10000})

# Detect fraud patterns
analyzer = StatisticalAnalyzer(data)
outliers = analyzer.outlier_detection(method='isolation_forest')

# Risk analysis
interpreter = DataInterpreter(data)
risk_report = interpreter.generate_business_insights(context={
    'domain': 'finance',
    'goal': 'risk_management'
})
```

### Time Series Analysis
```python
# Load time series data
data = loader.load_data({'type': 'sample', 'sample_type': 'timeseries', 'n_rows': 365})

# Visualize trends
visualizer = DataVisualizer(data)
visualizer.time_series_plot(
    date_column='date',
    value_columns=['value', 'moving_avg_7'],
    resample_freq='W'
)

# Detect patterns
interpreter = DataInterpreter(data)
patterns = interpreter.detect_patterns_and_trends(date_column='date')
```

## 🔧 Advanced Features

### Custom Validation Rules
```python
validator = DataValidator(data)
validator.add_validation_rule(
    rule_name='price_range',
    column='price',
    rule_type='range',
    parameters={'min': 0, 'max': 1000, 'severity': 'High'}
)

validator.add_validation_rule(
    rule_name='email_format',
    column='email',
    rule_type='pattern',
    parameters={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
)

results = validator.run_all_validations()
```

### Interactive Visualizations
```python
visualizer = DataVisualizer(data)

# Create interactive scatter plot
interactive_plot = visualizer.interactive_scatter(
    x_column='sales',
    y_column='profit',
    color_column='region',
    size_column='customers'
)

# Save as HTML
visualizer.save_figure('interactive_scatter', 'sales_analysis.html', format='html')
```

### Hypothesis Testing
```python
analyzer = StatisticalAnalyzer(data)

# Test if average sales differ between regions
anova_result = analyzer.hypothesis_testing(
    test_type='anova',
    groups=['north_sales', 'south_sales', 'east_sales', 'west_sales']
)

# Chi-square test for independence
chi2_result = analyzer.hypothesis_testing(
    test_type='chi2',
    column1='product_category',
    column2='customer_segment'
)
```

## 📁 Project Structure

```
AD/
├── ad_analysis/              # Main package
│   ├── core/                 # Core analysis modules
│   │   ├── data_processor.py
│   │   └── statistical_analyzer.py
│   ├── visualization/        # Visualization modules
│   │   └── data_visualizer.py
│   ├── interpretation/       # Insight generation
│   │   └── data_interpreter.py
│   └── utils/               # Utility modules
│       ├── data_loader.py
│       └── data_validator.py
├── examples/                # Example scripts and tutorials
│   └── data_analysis_tutorial.py
├── tests/                   # Unit tests
│   └── test_basic_functionality.py
├── cli.py                   # Command-line interface
├── requirements.txt         # Dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_basic_functionality.py

# Run tutorial example
python examples/data_analysis_tutorial.py
```

## 📚 Documentation

### Key Classes and Methods

#### DataProcessor
- `load_data(data)`: Load data from various sources
- `clean_missing_data(strategy)`: Handle missing values
- `remove_outliers(method)`: Remove statistical outliers
- `normalize_data(method)`: Normalize numeric data
- `encode_categorical(method)`: Encode categorical variables

#### StatisticalAnalyzer
- `descriptive_statistics()`: Comprehensive descriptive stats
- `correlation_analysis()`: Correlation matrix and analysis
- `hypothesis_testing()`: Various statistical tests
- `outlier_detection()`: Multiple outlier detection methods
- `distribution_analysis()`: Distribution fitting and testing

#### DataVisualizer
- `distribution_plots()`: Histograms, box plots, etc.
- `correlation_heatmap()`: Correlation visualization
- `scatter_plot_matrix()`: Pair plots
- `time_series_plot()`: Time series visualization
- `categorical_plots()`: Categorical data visualization

#### DataInterpreter
- `generate_data_quality_report()`: Automated quality assessment
- `analyze_relationships()`: Relationship analysis
- `detect_patterns_and_trends()`: Pattern detection
- `generate_business_insights()`: Business intelligence
- `generate_summary_report()`: Comprehensive summary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with powerful Python data science libraries: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn
- Inspired by the need for automated, comprehensive data analysis workflows
- Designed for both analysts and data scientists

## 🆘 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the examples/ directory for tutorials
- Run the test suite for validation

---

**Happy Analyzing! 📈✨**
