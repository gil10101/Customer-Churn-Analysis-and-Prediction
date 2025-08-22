"""
Visualization Theme Module for Customer Churn Analysis.

This module provides comprehensive visualization styling capabilities including
professional matplotlib/seaborn styling framework, plotly template system,
and publication-ready figure export utilities with proper formatting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ColorPalette:
    """
    Color palette configuration for consistent visualization styling.
    """
    # Primary colors
    primary: str = "#1f77b4"
    secondary: str = "#ff7f0e"
    success: str = "#2ca02c"
    danger: str = "#d62728"
    warning: str = "#ff7f0e"
    info: str = "#17becf"
    
    # Extended palette for categorical data
    categorical: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    # Sequential palettes for continuous data
    sequential_blue: List[str] = field(default_factory=lambda: [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b"
    ])
    
    sequential_red: List[str] = field(default_factory=lambda: [
        "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a",
        "#ef3b2c", "#cb181d", "#a50f15", "#67000d"
    ])
    
    # Diverging palette
    diverging: List[str] = field(default_factory=lambda: [
        "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
        "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"
    ])
    
    # Business-specific colors
    churn_colors: Dict[str, str] = field(default_factory=lambda: {
        "churned": "#d62728",
        "retained": "#2ca02c",
        "at_risk": "#ff7f0e",
        "safe": "#1f77b4"
    })


@dataclass
class FontConfiguration:
    """Font configuration for consistent typography."""
    family: str = "DejaVu Sans"
    title_size: int = 16
    subtitle_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 11
    annotation_size: int = 9
    
    # Font weights
    title_weight: str = "bold"
    subtitle_weight: str = "normal"
    label_weight: str = "normal"


@dataclass
class FigureConfiguration:
    """Figure size and layout configuration."""
    # Default figure sizes (width, height) in inches
    default_size: Tuple[float, float] = (12, 8)
    small_size: Tuple[float, float] = (8, 6)
    large_size: Tuple[float, float] = (16, 10)
    wide_size: Tuple[float, float] = (15, 6)
    square_size: Tuple[float, float] = (10, 10)
    
    # DPI settings
    screen_dpi: int = 100
    print_dpi: int = 300
    
    # Layout parameters
    tight_layout: bool = True
    constrained_layout: bool = False
    
    # Margins and spacing
    left_margin: float = 0.1
    right_margin: float = 0.9
    bottom_margin: float = 0.1
    top_margin: float = 0.9
    wspace: float = 0.2
    hspace: float = 0.2


class VisualizationTheme:
    """
    Comprehensive visualization theme manager.
    
    Provides consistent styling across matplotlib, seaborn, and plotly
    with professional themes and publication-ready export capabilities.
    """
    
    def __init__(
        self,
        theme_name: str = "professional",
        color_palette: Optional[ColorPalette] = None,
        font_config: Optional[FontConfiguration] = None,
        figure_config: Optional[FigureConfiguration] = None
    ):
        """
        Initialize visualization theme.
        
        Args:
            theme_name: Name of the theme ('professional', 'minimal', 'dark', 'publication')
            color_palette: Custom color palette configuration
            font_config: Custom font configuration
            figure_config: Custom figure configuration
        """
        self.theme_name = theme_name
        self.color_palette = color_palette or ColorPalette()
        self.font_config = font_config or FontConfiguration()
        self.figure_config = figure_config or FigureConfiguration()
        
        # Initialize theme configurations
        self._setup_matplotlib_theme()
        self._setup_seaborn_theme()
        self._setup_plotly_theme()
        
        logger.info(f"Initialized VisualizationTheme with '{theme_name}' theme")
    
    def _setup_matplotlib_theme(self) -> None:
        """Setup matplotlib theme configuration."""
        
        # Set matplotlib parameters
        plt.rcParams.update({
            # Figure settings
            'figure.figsize': self.figure_config.default_size,
            'figure.dpi': self.figure_config.screen_dpi,
            'savefig.dpi': self.figure_config.print_dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # Font settings
            'font.family': self.font_config.family,
            'font.size': self.font_config.label_size,
            'axes.titlesize': self.font_config.title_size,
            'axes.labelsize': self.font_config.label_size,
            'xtick.labelsize': self.font_config.tick_size,
            'ytick.labelsize': self.font_config.tick_size,
            'legend.fontsize': self.font_config.legend_size,
            'axes.titleweight': self.font_config.title_weight,
            
            # Color settings
            'axes.prop_cycle': plt.cycler('color', self.color_palette.categorical),
            
            # Grid and spines
            'axes.grid': True,
            'axes.grid.axis': 'both',
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.0,
            
            # Layout
            'figure.constrained_layout.use': self.figure_config.constrained_layout,
            'figure.autolayout': self.figure_config.tight_layout,
            
            # Other settings
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'axes.edgecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black'
        })
        
        # Apply theme-specific modifications
        if self.theme_name == 'minimal':
            plt.rcParams.update({
                'axes.grid': False,
                'axes.spines.left': False,
                'axes.spines.bottom': False,
                'xtick.bottom': False,
                'ytick.left': False
            })
        elif self.theme_name == 'dark':
            plt.rcParams.update({
                'axes.facecolor': '#2e2e2e',
                'figure.facecolor': '#2e2e2e',
                'savefig.facecolor': '#2e2e2e',
                'axes.edgecolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'text.color': 'white',
                'axes.labelcolor': 'white'
            })
        elif self.theme_name == 'publication':
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'mathtext.fontset': 'stix',
                'axes.linewidth': 1.5,
                'grid.linewidth': 0.8,
                'lines.linewidth': 2.0,
                'patch.linewidth': 0.5,
                'xtick.major.width': 1.5,
                'ytick.major.width': 1.5,
                'xtick.minor.width': 1.0,
                'ytick.minor.width': 1.0
            })
    
    def _setup_seaborn_theme(self) -> None:
        """Setup seaborn theme configuration."""
        
        # Set seaborn style based on theme
        if self.theme_name == 'minimal':
            sns.set_style("white")
        elif self.theme_name == 'dark':
            sns.set_style("dark")
        elif self.theme_name == 'publication':
            sns.set_style("whitegrid")
        else:  # professional
            sns.set_style("whitegrid")
        
        # Set color palette
        sns.set_palette(self.color_palette.categorical)
        
        # Set context for scaling
        sns.set_context("notebook", font_scale=1.0)
    
    def _setup_plotly_theme(self) -> None:
        """Setup plotly theme configuration."""
        
        # Create custom plotly template
        template_name = f"custom_{self.theme_name}"
        
        # Base template selection
        if self.theme_name == 'dark':
            base_template = "plotly_dark"
        else:
            base_template = "plotly_white"
        
        # Create custom template
        custom_template = go.layout.Template()
        
        # Layout configuration
        custom_template.layout = go.Layout(
            font=dict(
                family=self.font_config.family,
                size=self.font_config.label_size,
                color='black' if self.theme_name != 'dark' else 'white'
            ),
            title=dict(
                font=dict(
                    size=self.font_config.title_size,
                    family=self.font_config.family
                ),
                x=0.5,  # Center title
                xanchor='center'
            ),
            colorway=self.color_palette.categorical,
            plot_bgcolor='white' if self.theme_name != 'dark' else '#2e2e2e',
            paper_bgcolor='white' if self.theme_name != 'dark' else '#2e2e2e',
            
            # Grid configuration
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray' if self.theme_name != 'dark' else 'gray',
                showline=True,
                linewidth=1,
                linecolor='black' if self.theme_name != 'dark' else 'white'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray' if self.theme_name != 'dark' else 'gray',
                showline=True,
                linewidth=1,
                linecolor='black' if self.theme_name != 'dark' else 'white'
            ),
            
            # Legend configuration
            legend=dict(
                font=dict(size=self.font_config.legend_size),
                bgcolor='rgba(255,255,255,0.8)' if self.theme_name != 'dark' else 'rgba(46,46,46,0.8)',
                bordercolor='black' if self.theme_name != 'dark' else 'white',
                borderwidth=1
            )
        )
        
        # Register template
        pio.templates[template_name] = custom_template
        pio.templates.default = template_name
        
        self.plotly_template = template_name
    
    @contextmanager
    def apply_style(self):
        """Context manager to temporarily apply theme styles."""
        # Store current settings
        original_rcParams = plt.rcParams.copy()
        original_sns_style = sns.axes_style()
        original_sns_palette = sns.color_palette()
        original_plotly_template = pio.templates.default
        
        try:
            # Apply theme
            self._setup_matplotlib_theme()
            self._setup_seaborn_theme()
            pio.templates.default = self.plotly_template
            yield self
        finally:
            # Restore original settings
            plt.rcParams.update(original_rcParams)
            sns.set_style(original_sns_style)
            sns.set_palette(original_sns_palette)
            pio.templates.default = original_plotly_template
    
    def create_figure(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        subplot_config: Optional[Dict] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a matplotlib figure with theme styling.
        
        Args:
            figsize: Figure size (width, height) in inches
            subplot_config: Subplot configuration dictionary
            
        Returns:
            Tuple of (figure, axes)
        """
        figsize = figsize or self.figure_config.default_size
        
        if subplot_config:
            fig, axes = plt.subplots(figsize=figsize, **subplot_config)
        else:
            fig, axes = plt.subplots(figsize=figsize)
        
        return fig, axes
    
    def create_plotly_figure(
        self,
        subplot_config: Optional[Dict] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create a plotly figure with theme styling.
        
        Args:
            subplot_config: Subplot configuration for make_subplots
            **kwargs: Additional arguments for go.Figure()
            
        Returns:
            Plotly Figure object
        """
        if subplot_config:
            fig = make_subplots(**subplot_config)
        else:
            fig = go.Figure(**kwargs)
        
        # Apply template
        fig.update_layout(template=self.plotly_template)
        
        return fig
    
    def save_figure(
        self,
        fig: Union[plt.Figure, go.Figure],
        filepath: Path,
        format: str = 'png',
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight',
        transparent: bool = False,
        **kwargs
    ) -> None:
        """
        Save figure with publication-ready formatting.
        
        Args:
            fig: Figure object (matplotlib or plotly)
            filepath: Output file path
            format: Output format ('png', 'pdf', 'svg', 'eps', 'html')
            dpi: Resolution for raster formats
            bbox_inches: Bounding box for matplotlib figures
            transparent: Whether to use transparent background
            **kwargs: Additional arguments for save function
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        dpi = dpi or self.figure_config.print_dpi
        
        if isinstance(fig, plt.Figure):
            # Matplotlib figure
            save_kwargs = {
                'dpi': dpi,
                'bbox_inches': bbox_inches,
                'transparent': transparent,
                'facecolor': 'white' if not transparent else 'none',
                **kwargs
            }
            fig.savefig(filepath, format=format, **save_kwargs)
            
        elif isinstance(fig, go.Figure):
            # Plotly figure
            if format.lower() == 'html':
                fig.write_html(str(filepath), **kwargs)
            elif format.lower() in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps']:
                # Requires kaleido package
                try:
                    save_kwargs = {
                        'width': 1200,
                        'height': 800,
                        **kwargs
                    }
                    if format.lower() in ['png', 'jpg', 'jpeg', 'webp']:
                        save_kwargs['scale'] = dpi / 100  # Convert DPI to scale
                    
                    fig.write_image(str(filepath), format=format, **save_kwargs)
                except Exception as e:
                    logger.warning(f"Could not save plotly figure as {format}: {e}")
                    # Fallback to HTML
                    html_path = filepath.with_suffix('.html')
                    fig.write_html(str(html_path))
                    logger.info(f"Saved as HTML instead: {html_path}")
            else:
                raise ValueError(f"Unsupported format for plotly figure: {format}")
        
        logger.info(f"Figure saved to {filepath}")
    
    def apply_business_colors(self, ax: plt.Axes, data_type: str = "churn") -> None:
        """
        Apply business-specific color scheme to matplotlib axes.
        
        Args:
            ax: Matplotlib axes object
            data_type: Type of business data ('churn', 'revenue', 'segments')
        """
        if data_type == "churn":
            colors = [
                self.color_palette.churn_colors["retained"],
                self.color_palette.churn_colors["churned"]
            ]
        elif data_type == "risk":
            colors = [
                self.color_palette.churn_colors["safe"],
                self.color_palette.churn_colors["at_risk"],
                self.color_palette.churn_colors["churned"]
            ]
        else:
            colors = self.color_palette.categorical
        
        # Apply colors to current plot elements
        for i, patch in enumerate(ax.patches):
            if i < len(colors):
                patch.set_facecolor(colors[i % len(colors)])
    
    def create_correlation_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        figsize: Optional[Tuple[float, float]] = None,
        annot: bool = True,
        cmap: str = "RdBu_r"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a styled correlation heatmap.
        
        Args:
            data: DataFrame with numerical data
            title: Plot title
            figsize: Figure size
            annot: Whether to annotate cells
            cmap: Colormap name
            
        Returns:
            Tuple of (figure, axes)
        """
        figsize = figsize or self.figure_config.large_size
        
        fig, ax = self.create_figure(figsize=figsize)
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=annot,
            cmap=cmap,
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        
        ax.set_title(title, fontsize=self.font_config.title_size, pad=20)
        plt.tight_layout()
        
        return fig, ax
    
    def create_distribution_plot(
        self,
        data: pd.Series,
        title: str = "Distribution Plot",
        figsize: Optional[Tuple[float, float]] = None,
        kde: bool = True,
        hist: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a styled distribution plot.
        
        Args:
            data: Series with numerical data
            title: Plot title
            figsize: Figure size
            kde: Whether to show KDE
            hist: Whether to show histogram
            
        Returns:
            Tuple of (figure, axes)
        """
        figsize = figsize or self.figure_config.default_size
        
        fig, ax = self.create_figure(figsize=figsize)
        
        # Create distribution plot
        sns.histplot(
            data=data,
            kde=kde,
            stat='density' if kde else 'count',
            alpha=0.7,
            color=self.color_palette.primary,
            ax=ax
        )
        
        ax.set_title(title, fontsize=self.font_config.title_size, pad=20)
        ax.set_xlabel(data.name or 'Value', fontsize=self.font_config.label_size)
        ax.set_ylabel('Density' if kde else 'Count', fontsize=self.font_config.label_size)
        
        plt.tight_layout()
        
        return fig, ax
    
    def create_interactive_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: str = "Interactive Scatter Plot"
    ) -> go.Figure:
        """
        Create an interactive scatter plot with plotly.
        
        Args:
            data: DataFrame with plot data
            x: Column name for x-axis
            y: Column name for y-axis
            color: Column name for color encoding
            size: Column name for size encoding
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title,
            template=self.plotly_template,
            color_discrete_sequence=self.color_palette.categorical
        )
        
        fig.update_layout(
            title_x=0.5,
            font=dict(size=self.font_config.label_size)
        )
        
        return fig
    
    def get_color_palette(self, palette_type: str = "categorical", n_colors: Optional[int] = None) -> List[str]:
        """
        Get color palette for plotting.
        
        Args:
            palette_type: Type of palette ('categorical', 'sequential_blue', 'sequential_red', 'diverging', 'churn')
            n_colors: Number of colors to return
            
        Returns:
            List of color hex codes
        """
        if palette_type == "categorical":
            colors = self.color_palette.categorical
        elif palette_type == "sequential_blue":
            colors = self.color_palette.sequential_blue
        elif palette_type == "sequential_red":
            colors = self.color_palette.sequential_red
        elif palette_type == "diverging":
            colors = self.color_palette.diverging
        elif palette_type == "churn":
            colors = list(self.color_palette.churn_colors.values())
        else:
            colors = self.color_palette.categorical
        
        if n_colors:
            if n_colors <= len(colors):
                return colors[:n_colors]
            else:
                # Repeat colors if more needed
                return (colors * ((n_colors // len(colors)) + 1))[:n_colors]
        
        return colors
    
    def create_theme_showcase(self) -> go.Figure:
        """Create a showcase of the current theme."""
        
        # Sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'value': np.random.exponential(2, 100)
        })
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Scatter Plot', 'Bar Chart', 'Histogram', 'Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scatter plot
        for i, cat in enumerate(['A', 'B', 'C']):
            cat_data = sample_data[sample_data['category'] == cat]
            fig.add_trace(
                go.Scatter(
                    x=cat_data['x'],
                    y=cat_data['y'],
                    mode='markers',
                    name=f'Category {cat}',
                    marker=dict(color=self.color_palette.categorical[i])
                ),
                row=1, col=1
            )
        
        # Bar chart
        category_counts = sample_data['category'].value_counts()
        fig.add_trace(
            go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                name='Counts',
                marker=dict(color=self.color_palette.categorical[:len(category_counts)])
            ),
            row=1, col=2
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=sample_data['value'],
                name='Distribution',
                marker=dict(color=self.color_palette.primary)
            ),
            row=2, col=1
        )
        
        # Box plot
        for i, cat in enumerate(['A', 'B', 'C']):
            cat_data = sample_data[sample_data['category'] == cat]
            fig.add_trace(
                go.Box(
                    y=cat_data['value'],
                    name=f'Category {cat}',
                    marker=dict(color=self.color_palette.categorical[i])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Theme Showcase: {self.theme_name.title()}",
            template=self.plotly_template,
            showlegend=True,
            height=800
        )
        
        return fig


# Predefined themes
def get_professional_theme() -> VisualizationTheme:
    """Get professional business theme."""
    return VisualizationTheme(theme_name="professional")

def get_minimal_theme() -> VisualizationTheme:
    """Get minimal clean theme."""
    return VisualizationTheme(theme_name="minimal")

def get_dark_theme() -> VisualizationTheme:
    """Get dark theme for presentations."""
    return VisualizationTheme(theme_name="dark")

def get_publication_theme() -> VisualizationTheme:
    """Get publication-ready theme."""
    return VisualizationTheme(theme_name="publication")

# Global theme instance
DEFAULT_THEME = get_professional_theme()

# Convenience functions
def apply_theme(theme_name: str = "professional") -> VisualizationTheme:
    """Apply a visualization theme globally."""
    global DEFAULT_THEME
    
    if theme_name == "professional":
        DEFAULT_THEME = get_professional_theme()
    elif theme_name == "minimal":
        DEFAULT_THEME = get_minimal_theme()
    elif theme_name == "dark":
        DEFAULT_THEME = get_dark_theme()
    elif theme_name == "publication":
        DEFAULT_THEME = get_publication_theme()
    else:
        raise ValueError(f"Unknown theme: {theme_name}")
    
    return DEFAULT_THEME

def get_current_theme() -> VisualizationTheme:
    """Get the currently active theme."""
    return DEFAULT_THEME

def save_publication_figure(
    fig: Union[plt.Figure, go.Figure],
    filepath: Path,
    formats: List[str] = ['png', 'pdf'],
    **kwargs
) -> None:
    """
    Save figure in multiple publication-ready formats.
    
    Args:
        fig: Figure object
        filepath: Base output path (without extension)
        formats: List of formats to save
        **kwargs: Additional arguments for save function
    """
    theme = get_current_theme()
    
    for fmt in formats:
        output_path = Path(str(filepath)).with_suffix(f'.{fmt}')
        theme.save_figure(fig, output_path, format=fmt, **kwargs)