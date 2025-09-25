"""
Data Visualization Module
=========================

Comprehensive data visualization capabilities for analysis and interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings


class DataVisualizer:
    """
    Comprehensive data visualization class for creating insights through charts.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, style: str = 'seaborn'):
        """
        Initialize the DataVisualizer.
        
        Args:
            data: DataFrame to visualize
            style: Matplotlib style ('seaborn', 'ggplot', 'classic', etc.)
        """
        self.data = data
        self.figures = {}
        
        # Set plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Configure matplotlib for better display
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def set_data(self, data: pd.DataFrame) -> 'DataVisualizer':
        """
        Set the data for visualization.
        
        Args:
            data: DataFrame to visualize
        
        Returns:
            Self for method chaining
        """
        self.data = data
        return self
    
    def distribution_plots(self, columns: Optional[List[str]] = None, plot_type: str = 'histogram') -> Dict[str, Any]:
        """
        Create distribution plots for numeric columns.
        
        Args:
            columns: Columns to plot (None for all numeric)
            plot_type: 'histogram', 'density', 'box', 'violin', or 'qq'
        
        Returns:
            Dictionary with plot information
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        numeric_cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            raise ValueError("No numeric columns found for distribution plots.")
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if len(numeric_cols) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
            
            col_data = self.data[col].dropna()
            
            if plot_type == 'histogram':
                ax.hist(col_data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'Histogram of {col}')
            elif plot_type == 'density':
                col_data.plot(kind='density', ax=ax)
                ax.set_title(f'Density Plot of {col}')
            elif plot_type == 'box':
                ax.boxplot(col_data)
                ax.set_title(f'Box Plot of {col}')
            elif plot_type == 'violin':
                sns.violinplot(y=col_data, ax=ax)
                ax.set_title(f'Violin Plot of {col}')
            elif plot_type == 'qq':
                from scipy import stats
                stats.probplot(col_data, dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot of {col}')
            
            ax.set_xlabel(col)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_key = f'distribution_{plot_type}'
        self.figures[plot_key] = fig
        
        return {
            'plot_type': plot_type,
            'columns': numeric_cols,
            'figure': fig
        }
    
    def correlation_heatmap(self, method: str = 'pearson', annot: bool = True) -> Dict[str, Any]:
        """
        Create a correlation heatmap.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate correlation values
        
        Returns:
            Dictionary with plot information
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found for correlation heatmap.")
        
        corr_matrix = numeric_data.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=annot, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        
        ax.set_title(f'Correlation Heatmap ({method.capitalize()})')
        plt.tight_layout()
        
        self.figures['correlation_heatmap'] = fig
        
        return {
            'plot_type': 'correlation_heatmap',
            'method': method,
            'correlation_matrix': corr_matrix,
            'figure': fig
        }
    
    def scatter_plot_matrix(self, columns: Optional[List[str]] = None, color_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a scatter plot matrix (pair plot).
        
        Args:
            columns: Columns to include (None for all numeric)
            color_column: Column to use for coloring points
        
        Returns:
            Dictionary with plot information
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        numeric_cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("At least 2 numeric columns required for scatter plot matrix.")
        
        plot_data = self.data[numeric_cols + ([color_column] if color_column else [])]
        
        if color_column and color_column in self.data.columns:
            g = sns.pairplot(plot_data, hue=color_column, diag_kind='hist')
            title = f'Scatter Plot Matrix (colored by {color_column})'
        else:
            g = sns.pairplot(plot_data, diag_kind='hist')
            title = 'Scatter Plot Matrix'
        
        g.fig.suptitle(title, y=1.02)
        
        self.figures['scatter_matrix'] = g.fig
        
        return {
            'plot_type': 'scatter_matrix',
            'columns': numeric_cols,
            'color_column': color_column,
            'figure': g.fig
        }
    
    def time_series_plot(self, date_column: str, value_columns: List[str], 
                        resample_freq: Optional[str] = None) -> Dict[str, Any]:
        """
        Create time series plots.
        
        Args:
            date_column: Column with datetime data
            value_columns: Columns with numeric values to plot
            resample_freq: Resampling frequency ('D', 'W', 'M', etc.)
        
        Returns:
            Dictionary with plot information
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        if date_column not in self.data.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            try:
                date_data = pd.to_datetime(self.data[date_column])
            except:
                raise ValueError(f"Cannot convert '{date_column}' to datetime.")
        else:
            date_data = self.data[date_column]
        
        plot_data = self.data.copy()
        plot_data[date_column] = date_data
        plot_data = plot_data.set_index(date_column)
        
        # Resample if requested
        if resample_freq:
            plot_data = plot_data.resample(resample_freq).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in value_columns:
            if col in plot_data.columns:
                ax.plot(plot_data.index, plot_data[col], label=col, linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Time Series Plot')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self.figures['time_series'] = fig
        
        return {
            'plot_type': 'time_series',
            'date_column': date_column,
            'value_columns': value_columns,
            'resample_freq': resample_freq,
            'figure': fig
        }
    
    def categorical_plots(self, categorical_column: str, numeric_column: Optional[str] = None, 
                         plot_type: str = 'count') -> Dict[str, Any]:
        """
        Create plots for categorical data.
        
        Args:
            categorical_column: Column with categorical data
            numeric_column: Column with numeric data (for some plot types)
            plot_type: 'count', 'box', 'violin', 'bar', 'pie'
        
        Returns:
            Dictionary with plot information
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        if categorical_column not in self.data.columns:
            raise ValueError(f"Categorical column '{categorical_column}' not found.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'count':
            sns.countplot(data=self.data, x=categorical_column, ax=ax)
            ax.set_title(f'Count Plot of {categorical_column}')
            plt.xticks(rotation=45)
        
        elif plot_type == 'box' and numeric_column:
            sns.boxplot(data=self.data, x=categorical_column, y=numeric_column, ax=ax)
            ax.set_title(f'Box Plot: {numeric_column} by {categorical_column}')
            plt.xticks(rotation=45)
        
        elif plot_type == 'violin' and numeric_column:
            sns.violinplot(data=self.data, x=categorical_column, y=numeric_column, ax=ax)
            ax.set_title(f'Violin Plot: {numeric_column} by {categorical_column}')
            plt.xticks(rotation=45)
        
        elif plot_type == 'bar' and numeric_column:
            grouped_data = self.data.groupby(categorical_column)[numeric_column].mean()
            ax.bar(grouped_data.index, grouped_data.values)
            ax.set_title(f'Bar Plot: Average {numeric_column} by {categorical_column}')
            ax.set_xlabel(categorical_column)
            ax.set_ylabel(f'Average {numeric_column}')
            plt.xticks(rotation=45)
        
        elif plot_type == 'pie':
            value_counts = self.data[categorical_column].value_counts()
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax.set_title(f'Pie Chart of {categorical_column}')
        
        else:
            raise ValueError("Invalid plot_type or missing numeric_column for selected plot type.")
        
        plt.tight_layout()
        
        plot_key = f'categorical_{plot_type}'
        self.figures[plot_key] = fig
        
        return {
            'plot_type': plot_type,
            'categorical_column': categorical_column,
            'numeric_column': numeric_column,
            'figure': fig
        }
    
    def interactive_scatter(self, x_column: str, y_column: str, 
                           color_column: Optional[str] = None, 
                           size_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an interactive scatter plot using Plotly.
        
        Args:
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_column: Column for coloring points
            size_column: Column for sizing points
        
        Returns:
            Dictionary with plot information
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")
        
        fig = px.scatter(self.data, x=x_column, y=y_column,
                        color=color_column, size=size_column,
                        title=f'Interactive Scatter Plot: {y_column} vs {x_column}',
                        hover_data=self.data.columns.tolist())
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            hovermode='closest'
        )
        
        self.figures['interactive_scatter'] = fig
        
        return {
            'plot_type': 'interactive_scatter',
            'x_column': x_column,
            'y_column': y_column,
            'color_column': color_column,
            'size_column': size_column,
            'figure': fig
        }
    
    def save_figure(self, figure_key: str, filename: str, format: str = 'png', dpi: int = 300):
        """
        Save a figure to file.
        
        Args:
            figure_key: Key of the figure to save
            filename: Output filename
            format: File format ('png', 'pdf', 'svg', 'html' for plotly)
            dpi: Resolution for raster formats
        """
        if figure_key not in self.figures:
            raise ValueError(f"Figure '{figure_key}' not found. Available: {list(self.figures.keys())}")
        
        fig = self.figures[figure_key]
        
        # Handle Plotly figures
        if hasattr(fig, 'write_html'):
            if format == 'html':
                fig.write_html(filename)
            elif format == 'png':
                fig.write_image(filename, format='png', width=1200, height=800)
            elif format == 'pdf':
                fig.write_image(filename, format='pdf', width=1200, height=800)
            else:
                raise ValueError("For Plotly figures, use 'html', 'png', or 'pdf' format.")
        else:
            # Handle matplotlib/seaborn figures
            fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
    
    def show_figure(self, figure_key: str):
        """
        Display a figure.
        
        Args:
            figure_key: Key of the figure to display
        """
        if figure_key not in self.figures:
            raise ValueError(f"Figure '{figure_key}' not found. Available: {list(self.figures.keys())}")
        
        fig = self.figures[figure_key]
        
        if hasattr(fig, 'show'):
            fig.show()
        else:
            plt.figure(fig.number)
            plt.show()
    
    def get_figure_list(self) -> List[str]:
        """Get list of available figures."""
        return list(self.figures.keys())
    
    def close_all_figures(self):
        """Close all matplotlib figures."""
        plt.close('all')
        self.figures.clear()