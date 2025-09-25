"""
Data Processing Module
=====================

Core data processing functionality for cleaning, transforming, and preparing data for analysis.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
import warnings


class DataProcessor:
    """
    A comprehensive data processing class for cleaning and transforming datasets.
    """
    
    def __init__(self, data: Union[pd.DataFrame, str, None] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            data: DataFrame, file path, or None
        """
        self.data = None
        self.original_data = None
        self.processing_log = []
        
        if data is not None:
            self.load_data(data)
    
    def load_data(self, data: Union[pd.DataFrame, str]) -> 'DataProcessor':
        """
        Load data from DataFrame or file path.
        
        Args:
            data: DataFrame or file path (CSV, Excel, JSON)
        
        Returns:
            Self for method chaining
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str):
            if data.endswith('.csv'):
                self.data = pd.read_csv(data)
            elif data.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(data)
            elif data.endswith('.json'):
                self.data = pd.read_json(data)
            else:
                raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")
        else:
            raise ValueError("Data must be a DataFrame or file path string.")
        
        self.original_data = self.data.copy()
        self._log_operation(f"Loaded data with shape {self.data.shape}")
        return self
    
    def clean_missing_data(self, strategy: str = 'drop', columns: Optional[List[str]] = None) -> 'DataProcessor':
        """
        Handle missing data using various strategies.
        
        Args:
            strategy: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
            columns: Specific columns to process (None for all)
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        target_cols = columns or self.data.columns.tolist()
        
        if strategy == 'drop':
            original_shape = self.data.shape
            self.data = self.data.dropna(subset=target_cols)
            self._log_operation(f"Dropped missing values: {original_shape} -> {self.data.shape}")
        
        elif strategy == 'mean':
            numeric_cols = self.data[target_cols].select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
            self._log_operation(f"Filled missing values with mean for columns: {list(numeric_cols)}")
        
        elif strategy == 'median':
            numeric_cols = self.data[target_cols].select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
            self._log_operation(f"Filled missing values with median for columns: {list(numeric_cols)}")
        
        elif strategy == 'mode':
            for col in target_cols:
                if col in self.data.columns:
                    mode_value = self.data[col].mode()
                    if not mode_value.empty:
                        self.data[col] = self.data[col].fillna(mode_value[0])
            self._log_operation(f"Filled missing values with mode for columns: {target_cols}")
        
        elif strategy == 'forward_fill':
            self.data[target_cols] = self.data[target_cols].fillna(method='ffill')
            self._log_operation(f"Forward filled missing values for columns: {target_cols}")
        
        elif strategy == 'backward_fill':
            self.data[target_cols] = self.data[target_cols].fillna(method='bfill')
            self._log_operation(f"Backward filled missing values for columns: {target_cols}")
        
        else:
            raise ValueError("Invalid strategy. Choose from: drop, mean, median, mode, forward_fill, backward_fill")
        
        return self
    
    def remove_outliers(self, columns: Optional[List[str]] = None, method: str = 'iqr', threshold: float = 1.5) -> 'DataProcessor':
        """
        Remove outliers from numeric columns.
        
        Args:
            columns: Columns to process (None for all numeric)
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        numeric_cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        original_shape = self.data.shape
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                self.data = self.data[z_scores < threshold]
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        self._log_operation(f"Removed outliers using {method}: {original_shape} -> {self.data.shape}")
        return self
    
    def normalize_data(self, columns: Optional[List[str]] = None, method: str = 'minmax') -> 'DataProcessor':
        """
        Normalize numeric data.
        
        Args:
            columns: Columns to normalize (None for all numeric)
            method: 'minmax', 'standard', or 'robust'
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        numeric_cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'minmax', 'standard', or 'robust'")
        
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        self._log_operation(f"Normalized columns using {method}: {numeric_cols}")
        return self
    
    def encode_categorical(self, columns: Optional[List[str]] = None, method: str = 'onehot') -> 'DataProcessor':
        """
        Encode categorical variables.
        
        Args:
            columns: Columns to encode (None for all categorical)
            method: 'onehot', 'label', or 'target'
        
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        categorical_cols = columns or self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'onehot':
            self.data = pd.get_dummies(self.data, columns=categorical_cols, prefix=categorical_cols)
            self._log_operation(f"One-hot encoded columns: {categorical_cols}")
        
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
            self._log_operation(f"Label encoded columns: {categorical_cols}")
        
        else:
            raise ValueError("Method must be 'onehot' or 'label'")
        
        return self
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Returns:
            Dictionary with data summary information
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'processing_log': self.processing_log.copy()
        }
        
        # Add numeric summary
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = self.data[numeric_cols].describe().to_dict()
        
        return summary
    
    def _log_operation(self, operation: str):
        """Log processing operations."""
        self.processing_log.append(operation)
    
    def reset_data(self) -> 'DataProcessor':
        """Reset data to original state."""
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self.processing_log = []
            self._log_operation("Reset to original data")
        return self