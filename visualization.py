"""
Visualization module for urban transport analysis.
Creates charts, plots, and visual insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def setup_figure(figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
    """Create a new figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def save_figure(fig, filename: str, output_dir: str = 'output'):
    """Save figure to file."""
    Path(output_dir).mkdir(exist_ok=True)
    filepath = Path(output_dir) / filename
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return str(filepath)


def plot_traffic_distribution(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Plot distribution of Traffic Index across all data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    sns.histplot(data=df, x='Traffic_Index', hue='City', kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Traffic Index by City')
    axes[0].set_xlabel('Traffic Index')
    axes[0].set_ylabel('Frequency')
    
    # Box plot
    sns.boxplot(data=df, x='City', y='Traffic_Index', ax=axes[1])
    axes[1].set_title('Traffic Index Box Plot by City')
    axes[1].set_xlabel('City')
    axes[1].set_ylabel('Traffic Index')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'traffic_distribution.png', output_dir)
    
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Plot correlation heatmap for numeric variables.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in columns if col != 'Year']
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title('Correlation Matrix: Transport & Congestion Variables', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'correlation_heatmap.png', output_dir)
    
    return fig


def plot_vehicle_vs_congestion(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Scatter plot of Vehicle Count vs Traffic Index.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.scatterplot(
        data=df,
        x='Vehicle_Count',
        y='Traffic_Index',
        hue='City',
        size='Population',
        sizes=(100, 500),
        alpha=0.7,
        ax=ax
    )
    
    # Add trend line
    z = np.polyfit(df['Vehicle_Count'], df['Traffic_Index'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Vehicle_Count'].min(), df['Vehicle_Count'].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.8, label='Trend Line')
    
    ax.set_title('Vehicle Count vs Traffic Congestion', fontsize=14)
    ax.set_xlabel('Number of Registered Vehicles')
    ax.set_ylabel('Traffic Index')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'vehicle_vs_congestion.png', output_dir)
    
    return fig


def plot_transport_vs_congestion(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Scatter plot of Public Transport Capacity vs Traffic Index.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.scatterplot(
        data=df,
        x='Public_Transport_Capacity',
        y='Traffic_Index',
        hue='City',
        size='Population',
        sizes=(100, 500),
        alpha=0.7,
        ax=ax
    )
    
    # Add trend line
    z = np.polyfit(df['Public_Transport_Capacity'], df['Traffic_Index'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Public_Transport_Capacity'].min(), df['Public_Transport_Capacity'].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.8, label='Trend Line')
    
    ax.set_title('Public Transport Capacity vs Traffic Congestion', fontsize=14)
    ax.set_xlabel('Public Transport Capacity (Daily Passengers)')
    ax.set_ylabel('Traffic Index')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'transport_vs_congestion.png', output_dir)
    
    return fig


def plot_city_comparison(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Multi-panel comparison of cities across key metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    latest_year = df['Year'].max()
    latest_df = df[df['Year'] == latest_year]
    
    # Panel 1: Traffic Index by City
    sns.barplot(data=latest_df, x='City', y='Traffic_Index', ax=axes[0, 0], palette='Reds_r')
    axes[0, 0].set_title(f'Traffic Index by City ({latest_year})')
    axes[0, 0].set_ylabel('Traffic Index')
    
    # Panel 2: Transport Capacity per Capita
    if 'Transport_Capacity_Per_Capita' in latest_df.columns:
        sns.barplot(data=latest_df, x='City', y='Transport_Capacity_Per_Capita', 
                   ax=axes[0, 1], palette='Greens')
        axes[0, 1].set_title(f'Transport Capacity per Capita ({latest_year})')
        axes[0, 1].set_ylabel('Capacity / Population')
    
    # Panel 3: Vehicles per Capita
    if 'Vehicles_Per_Capita' in latest_df.columns:
        sns.barplot(data=latest_df, x='City', y='Vehicles_Per_Capita', 
                   ax=axes[1, 0], palette='Oranges')
        axes[1, 0].set_title(f'Vehicles per Capita ({latest_year})')
        axes[1, 0].set_ylabel('Vehicles / Population')
    
    # Panel 4: Average Travel Time
    if 'Average_Travel_Time' in latest_df.columns:
        sns.barplot(data=latest_df, x='City', y='Average_Travel_Time', 
                   ax=axes[1, 1], palette='Blues')
        axes[1, 1].set_title(f'Average Travel Time ({latest_year})')
        axes[1, 1].set_ylabel('Minutes')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'city_comparison.png', output_dir)
    
    return fig


def plot_trends_over_time(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Plot trends of key metrics over time by city.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Traffic Index trend
    sns.lineplot(data=df, x='Year', y='Traffic_Index', hue='City', 
                marker='o', ax=axes[0, 0])
    axes[0, 0].set_title('Traffic Index Trend')
    axes[0, 0].set_ylabel('Traffic Index')
    
    # Vehicle Count trend
    sns.lineplot(data=df, x='Year', y='Vehicle_Count', hue='City', 
                marker='o', ax=axes[0, 1])
    axes[0, 1].set_title('Vehicle Count Trend')
    axes[0, 1].set_ylabel('Number of Vehicles')
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Transport Capacity trend
    sns.lineplot(data=df, x='Year', y='Public_Transport_Capacity', hue='City', 
                marker='o', ax=axes[1, 0])
    axes[1, 0].set_title('Public Transport Capacity Trend')
    axes[1, 0].set_ylabel('Daily Capacity')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Transport Capacity per Capita trend
    if 'Transport_Capacity_Per_Capita' in df.columns:
        sns.lineplot(data=df, x='Year', y='Transport_Capacity_Per_Capita', hue='City', 
                    marker='o', ax=axes[1, 1])
        axes[1, 1].set_title('Transport Capacity per Capita Trend')
        axes[1, 1].set_ylabel('Capacity / Population')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'trends_over_time.png', output_dir)
    
    return fig


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Plot feature importance from regression models.
    """
    # Get importance from Random Forest or best available model
    if 'Random Forest' in feature_importance['Model'].values:
        model_imp = feature_importance[feature_importance['Model'] == 'Random Forest']
    else:
        model_imp = feature_importance.groupby('Feature')['Importance'].mean().reset_index()
    
    model_imp = model_imp.sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(model_imp['Feature'], model_imp['Importance'], color='steelblue')
    
    ax.set_title('Feature Importance for Traffic Congestion Prediction', fontsize=14)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Add value labels
    for bar, val in zip(bars, model_imp['Importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'feature_importance.png', output_dir)
    
    return fig


def plot_regression_results(
    df: pd.DataFrame,
    predictions: np.ndarray,
    save: bool = True,
    output_dir: str = 'output'
) -> plt.Figure:
    """
    Plot actual vs predicted values from regression model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    actual = df['Traffic_Index'].values[:len(predictions)]
    
    # Actual vs Predicted scatter
    axes[0].scatter(actual, predictions, alpha=0.7, c='steelblue')
    
    # Perfect prediction line
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    axes[0].set_title('Actual vs Predicted Traffic Index')
    axes[0].set_xlabel('Actual Traffic Index')
    axes[0].set_ylabel('Predicted Traffic Index')
    axes[0].legend()
    
    # Residuals
    residuals = actual - predictions
    axes[1].scatter(predictions, residuals, alpha=0.7, c='steelblue')
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_title('Residual Plot')
    axes[1].set_xlabel('Predicted Traffic Index')
    axes[1].set_ylabel('Residuals')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'regression_results.png', output_dir)
    
    return fig


def create_all_visualizations(
    df: pd.DataFrame,
    feature_importance: pd.DataFrame = None,
    predictions: np.ndarray = None,
    output_dir: str = 'output'
) -> List[str]:
    """
    Generate all visualizations and return list of saved file paths.
    """
    saved_files = []
    
    print("\n=== Generating Visualizations ===")
    
    # Traffic distribution
    plot_traffic_distribution(df, save=True, output_dir=output_dir)
    saved_files.append('traffic_distribution.png')
    print("✓ Traffic distribution plot")
    
    # Correlation heatmap
    plot_correlation_heatmap(df, save=True, output_dir=output_dir)
    saved_files.append('correlation_heatmap.png')
    print("✓ Correlation heatmap")
    
    # Vehicle vs congestion
    plot_vehicle_vs_congestion(df, save=True, output_dir=output_dir)
    saved_files.append('vehicle_vs_congestion.png')
    print("✓ Vehicle vs congestion scatter plot")
    
    # Transport vs congestion
    plot_transport_vs_congestion(df, save=True, output_dir=output_dir)
    saved_files.append('transport_vs_congestion.png')
    print("✓ Transport vs congestion scatter plot")
    
    # City comparison
    plot_city_comparison(df, save=True, output_dir=output_dir)
    saved_files.append('city_comparison.png')
    print("✓ City comparison charts")
    
    # Trends over time
    plot_trends_over_time(df, save=True, output_dir=output_dir)
    saved_files.append('trends_over_time.png')
    print("✓ Trends over time")
    
    # Feature importance (if available)
    if feature_importance is not None and not feature_importance.empty:
        plot_feature_importance(feature_importance, save=True, output_dir=output_dir)
        saved_files.append('feature_importance.png')
        print("✓ Feature importance plot")
    
    # Regression results (if available)
    if predictions is not None:
        plot_regression_results(df, predictions, save=True, output_dir=output_dir)
        saved_files.append('regression_results.png')
        print("✓ Regression results plot")
    
    print(f"\nAll visualizations saved to '{output_dir}/'")
    
    return saved_files
