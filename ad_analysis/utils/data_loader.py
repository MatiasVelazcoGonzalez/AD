"""
Data Loading Utilities
======================

Utilities for loading data from various sources and formats.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional, List
import os
import warnings
import json


class DataLoader:
    """
    Utility class for loading data from various sources and formats.
    """
    
    def __init__(self):
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.txt': self._load_text,
            '.tsv': self._load_tsv
        }
    
    def load_data(self, source: Union[str, Dict[str, Any]], **kwargs) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            source: File path, URL, or configuration dictionary
            **kwargs: Additional parameters for specific loaders
        
        Returns:
            Loaded DataFrame
        """
        if isinstance(source, str):
            return self._load_from_file(source, **kwargs)
        elif isinstance(source, dict):
            return self._load_from_config(source, **kwargs)
        else:
            raise ValueError("Source must be a file path (string) or configuration dictionary")
    
    def _load_from_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from a file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return self.supported_formats[file_ext](file_path, **kwargs)
    
    def _load_from_config(self, config: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Load data from configuration dictionary."""
        source_type = config.get('type', '').lower()
        
        if source_type == 'file':
            return self._load_from_file(config['path'], **kwargs)
        elif source_type == 'sample':
            return self._generate_sample_data(config, **kwargs)
        elif source_type == 'database':
            return self._load_from_database(config, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        default_params = {
            'encoding': 'utf-8',
            'low_memory': False,
            'parse_dates': True,
            'infer_datetime_format': True
        }
        default_params.update(kwargs)
        
        try:
            return pd.read_csv(file_path, **default_params)
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    default_params['encoding'] = encoding
                    return pd.read_csv(file_path, **default_params)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode file with common encodings")
    
    def _load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        default_params = {
            'engine': 'openpyxl' if file_path.endswith('.xlsx') else 'xlrd'
        }
        default_params.update(kwargs)
        
        return pd.read_excel(file_path, **default_params)
    
    def _load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        default_params = {
            'orient': 'records',
            'lines': False
        }
        default_params.update(kwargs)
        
        try:
            return pd.read_json(file_path, **default_params)
        except ValueError:
            # Try loading as JSONL (lines=True)
            default_params['lines'] = True
            return pd.read_json(file_path, **default_params)
    
    def _load_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(file_path, **kwargs)
    
    def _load_text(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load text file with custom delimiter."""
        default_params = {
            'sep': '\t',
            'encoding': 'utf-8'
        }
        default_params.update(kwargs)
        
        return pd.read_csv(file_path, **default_params)
    
    def _load_tsv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load TSV (Tab-Separated Values) file."""
        default_params = {
            'sep': '\t',
            'encoding': 'utf-8'
        }
        default_params.update(kwargs)
        
        return pd.read_csv(file_path, **default_params)
    
    def _load_from_database(self, config: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Load data from database (placeholder for future implementation)."""
        # This would require database-specific libraries
        raise NotImplementedError("Database loading not yet implemented")
    
    def _generate_sample_data(self, config: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Generate sample data for testing and demonstration."""
        sample_type = config.get('sample_type', 'mixed')
        n_rows = config.get('n_rows', 1000)
        
        np.random.seed(config.get('seed', 42))
        
        if sample_type == 'mixed':
            return self._generate_mixed_sample_data(n_rows)
        elif sample_type == 'timeseries':
            return self._generate_timeseries_sample_data(n_rows)
        elif sample_type == 'ecommerce':
            return self._generate_ecommerce_sample_data(n_rows)
        elif sample_type == 'financial':
            return self._generate_financial_sample_data(n_rows)
        else:
            return self._generate_mixed_sample_data(n_rows)
    
    def _generate_mixed_sample_data(self, n_rows: int) -> pd.DataFrame:
        """Generate mixed sample data with various data types."""
        data = {
            'id': range(1, n_rows + 1),
            'numeric_normal': np.random.normal(100, 15, n_rows),
            'numeric_uniform': np.random.uniform(0, 100, n_rows),
            'numeric_skewed': np.random.exponential(2, n_rows),
            'category_a': np.random.choice(['A', 'B', 'C', 'D'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
            'category_b': np.random.choice(['Type1', 'Type2', 'Type3'], n_rows, p=[0.5, 0.3, 0.2]),
            'boolean_flag': np.random.choice([True, False], n_rows, p=[0.7, 0.3]),
            'date': pd.date_range(start='2020-01-01', periods=n_rows, freq='D'),
            'text_length': np.random.randint(5, 50, n_rows)
        }
        
        # Add some missing values
        df = pd.DataFrame(data)
        df.loc[np.random.choice(df.index, size=int(n_rows * 0.05), replace=False), 'numeric_normal'] = np.nan
        df.loc[np.random.choice(df.index, size=int(n_rows * 0.03), replace=False), 'category_a'] = np.nan
        
        return df
    
    def _generate_timeseries_sample_data(self, n_rows: int) -> pd.DataFrame:
        """Generate time series sample data."""
        dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='D')
        
        # Generate trend + seasonality + noise
        trend = np.linspace(100, 200, n_rows)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(n_rows) / 365.25)
        noise = np.random.normal(0, 5, n_rows)
        
        values = trend + seasonality + noise
        
        data = {
            'date': dates,
            'value': values,
            'cumulative': np.cumsum(values),
            'moving_avg_7': pd.Series(values).rolling(7).mean(),
            'volatility': pd.Series(values).rolling(30).std()
        }
        
        return pd.DataFrame(data)
    
    def _generate_ecommerce_sample_data(self, n_rows: int) -> pd.DataFrame:
        """Generate e-commerce sample data."""
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
        customer_types = ['Regular', 'Premium', 'VIP']
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        data = {
            'order_id': range(1, n_rows + 1),
            'customer_id': np.random.randint(1, n_rows//3, n_rows),
            'product_category': np.random.choice(categories, n_rows),
            'customer_type': np.random.choice(customer_types, n_rows, p=[0.6, 0.3, 0.1]),
            'region': np.random.choice(regions, n_rows),
            'order_value': np.random.lognormal(4, 0.5, n_rows),
            'quantity': np.random.randint(1, 10, n_rows),
            'discount_percent': np.random.beta(2, 8, n_rows) * 30,
            'order_date': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
            'is_returned': np.random.choice([True, False], n_rows, p=[0.1, 0.9])
        }
        
        return pd.DataFrame(data)
    
    def _generate_financial_sample_data(self, n_rows: int) -> pd.DataFrame:
        """Generate financial sample data."""
        data = {
            'transaction_id': range(1, n_rows + 1),
            'account_id': np.random.randint(1000, 9999, n_rows),
            'transaction_type': np.random.choice(['Credit', 'Debit'], n_rows, p=[0.4, 0.6]),
            'amount': np.random.lognormal(5, 1, n_rows),
            'balance_before': np.random.uniform(100, 10000, n_rows),
            'transaction_date': pd.date_range(start='2023-01-01', periods=n_rows, freq='T'),
            'merchant_category': np.random.choice(['ATM', 'Grocery', 'Gas', 'Restaurant', 'Online'], n_rows),
            'is_fraud': np.random.choice([True, False], n_rows, p=[0.001, 0.999])
        }
        
        df = pd.DataFrame(data)
        # Calculate balance after
        df['balance_after'] = df['balance_before'] + np.where(df['transaction_type'] == 'Credit', 
                                                              df['amount'], -df['amount'])
        
        return df
    
    def get_data_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a data file without loading it completely.
        
        Args:
            file_path: Path to the data file
        
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        info = {
            'file_path': file_path,
            'file_extension': file_ext,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024)
        }
        
        if file_ext == '.csv':
            # Get CSV info without loading all data
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                info['columns'] = first_line.strip().split(',')
                info['estimated_columns'] = len(info['columns'])
            
            # Estimate rows (rough estimate)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
                info['estimated_rows'] = line_count - 1  # Subtract header
        
        elif file_ext in ['.xlsx', '.xls']:
            # Get Excel info
            try:
                excel_file = pd.ExcelFile(file_path)
                info['sheet_names'] = excel_file.sheet_names
                # Get info for first sheet
                df_sample = pd.read_excel(file_path, nrows=0)
                info['columns'] = list(df_sample.columns)
                info['estimated_columns'] = len(info['columns'])
            except Exception as e:
                info['error'] = str(e)
        
        return info
    
    def preview_data(self, file_path: str, n_rows: int = 5) -> pd.DataFrame:
        """
        Preview the first few rows of a data file.
        
        Args:
            file_path: Path to the data file
            n_rows: Number of rows to preview
        
        Returns:
            DataFrame with preview data
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path, nrows=n_rows)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, nrows=n_rows)
        elif file_ext == '.json':
            # For JSON, load all and take first n_rows
            df = pd.read_json(file_path)
            return df.head(n_rows)
        else:
            raise ValueError(f"Preview not supported for {file_ext} files")
    
    def list_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.supported_formats.keys())