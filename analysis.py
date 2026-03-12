"""
Statistical analysis module for urban transport analysis.
Handles exploratory analysis, correlation, and statistical tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics for the dataset.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = df[numeric_cols].describe()
    
    # Add additional statistics
    summary.loc['median'] = df[numeric_cols].median()
    summary.loc['skewness'] = df[numeric_cols].skew()
    summary.loc['kurtosis'] = df[numeric_cols].kurtosis()
    summary.loc['missing'] = df[numeric_cols].isnull().sum()
    
    return summary.round(2)


def analyze_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate city-level aggregated statistics.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_agg = [col for col in numeric_cols if col != 'Year']
    
    city_stats = df.groupby('City')[cols_to_agg].agg(['mean', 'std', 'min', 'max'])
    city_stats.columns = ['_'.join(col).strip() for col in city_stats.columns.values]
    
    return city_stats.round(2)


def analyze_trends(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze year-over-year trends for each city.
    """
    trends = {}
    
    for city in df['City'].unique():
        city_data = df[df['City'] == city].sort_values('Year')
        
        # Calculate year-over-year changes
        numeric_cols = city_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Year']
        
        changes = city_data[numeric_cols].pct_change() * 100
        changes['Year'] = city_data['Year'].values
        changes = changes.dropna()
        
        trends[city] = changes.round(2)
    
    return trends


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to include (None for all numeric)
        method: 'pearson', 'spearman', or 'kendall'
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude Year from correlation
        columns = [col for col in columns if col != 'Year']
    
    return df[columns].corr(method=method).round(3)


def compute_key_correlations(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute correlations for key research questions.
    """
    results = {}
    
    # Research Question 1: Public transport capacity vs congestion
    if 'Public_Transport_Capacity' in df.columns and 'Traffic_Index' in df.columns:
        corr, p_value = stats.pearsonr(
            df['Public_Transport_Capacity'].dropna(),
            df['Traffic_Index'].dropna()
        )
        results['transport_capacity_vs_congestion'] = {
            'correlation': round(corr, 3),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'interpretation': 'negative' if corr < 0 else 'positive'
        }
    
    # Research Question 2: Vehicle count vs congestion
    if 'Vehicle_Count' in df.columns and 'Traffic_Index' in df.columns:
        corr, p_value = stats.pearsonr(
            df['Vehicle_Count'].dropna(),
            df['Traffic_Index'].dropna()
        )
        results['vehicle_count_vs_congestion'] = {
            'correlation': round(corr, 3),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'interpretation': 'negative' if corr < 0 else 'positive'
        }
    
    # Additional: Vehicles per capita vs congestion
    if 'Vehicles_Per_Capita' in df.columns and 'Traffic_Index' in df.columns:
        corr, p_value = stats.pearsonr(
            df['Vehicles_Per_Capita'].dropna(),
            df['Traffic_Index'].dropna()
        )
        results['vehicles_per_capita_vs_congestion'] = {
            'correlation': round(corr, 3),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'interpretation': 'negative' if corr < 0 else 'positive'
        }
    
    # Transport capacity per capita vs congestion
    if 'Transport_Capacity_Per_Capita' in df.columns and 'Traffic_Index' in df.columns:
        corr, p_value = stats.pearsonr(
            df['Transport_Capacity_Per_Capita'].dropna(),
            df['Traffic_Index'].dropna()
        )
        results['transport_per_capita_vs_congestion'] = {
            'correlation': round(corr, 3),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'interpretation': 'negative' if corr < 0 else 'positive'
        }
    
    return results


def rank_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank cities by transport infrastructure efficiency.
    Research Question 3: Best balance between infrastructure and traffic flow.
    """
    # Get latest year data for each city
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].copy()
    
    # Calculate efficiency score
    # Lower traffic index + higher transport capacity per capita = better
    if 'Transport_Capacity_Per_Capita' in latest_data.columns:
        # Normalize metrics to 0-1 scale
        latest_data['traffic_score'] = 1 - (
            (latest_data['Traffic_Index'] - latest_data['Traffic_Index'].min()) /
            (latest_data['Traffic_Index'].max() - latest_data['Traffic_Index'].min())
        )
        latest_data['transport_score'] = (
            (latest_data['Transport_Capacity_Per_Capita'] - latest_data['Transport_Capacity_Per_Capita'].min()) /
            (latest_data['Transport_Capacity_Per_Capita'].max() - latest_data['Transport_Capacity_Per_Capita'].min())
        )
        latest_data['efficiency_score'] = (latest_data['traffic_score'] + latest_data['transport_score']) / 2
    else:
        # Simpler scoring without transport per capita
        latest_data['efficiency_score'] = 1 - (
            (latest_data['Traffic_Index'] - latest_data['Traffic_Index'].min()) /
            (latest_data['Traffic_Index'].max() - latest_data['Traffic_Index'].min())
        )
    
    rankings = latest_data[['City', 'Traffic_Index', 'efficiency_score']].copy()
    
    if 'Transport_Capacity_Per_Capita' in latest_data.columns:
        rankings['Transport_Capacity_Per_Capita'] = latest_data['Transport_Capacity_Per_Capita']
    
    rankings = rankings.sort_values('efficiency_score', ascending=False)
    rankings['Rank'] = range(1, len(rankings) + 1)
    
    return rankings.round(3)


def perform_anova(df: pd.DataFrame, group_col: str = 'City', value_col: str = 'Traffic_Index') -> Dict:
    """
    Perform one-way ANOVA to test if congestion differs significantly between cities.
    """
    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'f_statistic': round(f_stat, 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'interpretation': 'Cities have significantly different congestion levels' if p_value < 0.05 
                         else 'No significant difference in congestion between cities'
    }


def generate_insights(
    df: pd.DataFrame,
    correlations: Dict,
    rankings: pd.DataFrame
) -> List[str]:
    """
    Generate human-readable insights from the analysis.
    """
    insights = []
    
    # Insight 1: Transport capacity and congestion relationship
    if 'transport_capacity_vs_congestion' in correlations:
        corr_data = correlations['transport_capacity_vs_congestion']
        if corr_data['significant']:
            direction = "lower" if corr_data['correlation'] < 0 else "higher"
            insights.append(
                f"Cities with higher public transport capacity tend to have {direction} "
                f"congestion levels (r={corr_data['correlation']}, p={corr_data['p_value']})."
            )
    
    # Insight 2: Vehicle count and congestion
    if 'vehicle_count_vs_congestion' in correlations:
        corr_data = correlations['vehicle_count_vs_congestion']
        strength = "strongly" if abs(corr_data['correlation']) > 0.7 else "moderately"
        insights.append(
            f"Vehicle count is {strength} correlated with congestion levels "
            f"(r={corr_data['correlation']})."
        )
    
    # Insight 3: Best performing city
    if not rankings.empty:
        best_city = rankings.iloc[0]['City']
        worst_city = rankings.iloc[-1]['City']
        insights.append(
            f"{best_city} has the best balance between transport infrastructure and traffic flow, "
            f"while {worst_city} has the most room for improvement."
        )
    
    # Insight 4: Per capita analysis
    if 'transport_per_capita_vs_congestion' in correlations:
        corr_data = correlations['transport_per_capita_vs_congestion']
        if corr_data['significant'] and corr_data['correlation'] < 0:
            insights.append(
                "Transport capacity per capita is a stronger predictor of low congestion "
                "than absolute transport capacity, suggesting cities should scale public "
                "transport with population growth."
            )
    
    # Insight 5: City comparison
    city_means = df.groupby('City')['Traffic_Index'].mean()
    highest = city_means.idxmax()
    lowest = city_means.idxmin()
    diff = city_means[highest] - city_means[lowest]
    insights.append(
        f"{highest} experiences {diff:.1f} points higher average traffic index than {lowest}, "
        f"indicating significant variation in urban congestion management approaches."
    )
    
    return insights
