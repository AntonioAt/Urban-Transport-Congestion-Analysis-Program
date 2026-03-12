"""
Data cleaning module for urban transport analysis.
Handles missing values, normalization, type conversion, and deduplication.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def normalize_city_names(df: pd.DataFrame, city_column: str = 'City') -> pd.DataFrame:
    """
    Normalize city names to consistent format.
    Handles common variations and typos.
    """
    df = df.copy()
    
    city_mapping = {
        'jakarta': 'Jakarta',
        'jkt': 'Jakarta',
        'dki jakarta': 'Jakarta',
        'surabaya': 'Surabaya',
        'sby': 'Surabaya',
        'bandung': 'Bandung',
        'bdg': 'Bandung',
        'singapore': 'Singapore',
        'sg': 'Singapore',
        'singapura': 'Singapore'
    }
    
    df[city_column] = df[city_column].str.strip().str.lower()
    df[city_column] = df[city_column].map(lambda x: city_mapping.get(x, x.title()))
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mixed',
    numeric_fill: str = 'median',
    categorical_fill: str = 'mode'
) -> Tuple[pd.DataFrame, dict]:
    """
    Handle missing values with configurable strategies.
    
    Args:
        df: Input DataFrame
        strategy: 'drop', 'fill', or 'mixed' (fill numerics, drop if categorical missing)
        numeric_fill: 'mean', 'median', or 'zero' for numeric columns
        categorical_fill: 'mode' or 'unknown' for categorical columns
        
    Returns:
        Tuple of (cleaned DataFrame, report dict)
    """
    df = df.copy()
    report = {
        'original_rows': len(df),
        'missing_before': df.isnull().sum().to_dict(),
        'columns_affected': []
    }
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy in ['fill', 'mixed']:
        for column in df.columns:
            if df[column].isnull().any():
                report['columns_affected'].append(column)
                
                if pd.api.types.is_numeric_dtype(df[column]):
                    if numeric_fill == 'mean':
                        fill_value = df[column].mean()
                    elif numeric_fill == 'median':
                        fill_value = df[column].median()
                    else:
                        fill_value = 0
                    df[column] = df[column].fillna(fill_value)
                else:
                    if strategy == 'mixed':
                        df = df.dropna(subset=[column])
                    elif categorical_fill == 'mode':
                        mode_val = df[column].mode()
                        if len(mode_val) > 0:
                            df[column] = df[column].fillna(mode_val[0])
                    else:
                        df[column] = df[column].fillna('Unknown')
    
    report['final_rows'] = len(df)
    report['rows_removed'] = report['original_rows'] - report['final_rows']
    report['missing_after'] = df.isnull().sum().to_dict()
    
    return df, report


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert specified columns to numeric types.
    Auto-detects numeric columns if none specified.
    """
    df = df.copy()
    
    if columns is None:
        # Auto-detect columns that should be numeric
        numeric_patterns = [
            'count', 'population', 'fleet', 'passengers', 
            'capacity', 'index', 'time', 'rate'
        ]
        columns = [
            col for col in df.columns 
            if any(pattern in col.lower() for pattern in numeric_patterns)
        ]
    
    for col in columns:
        if col in df.columns:
            # Remove commas and convert
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'last'
) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows.
    
    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed)
    """
    original_len = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    duplicates_removed = original_len - len(df)
    
    return df, duplicates_removed


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated features useful for analysis.
    """
    df = df.copy()
    
    # Vehicles per capita
    if 'Vehicle_Count' in df.columns and 'Population' in df.columns:
        df['Vehicles_Per_Capita'] = df['Vehicle_Count'] / df['Population']
    
    # Transport capacity per capita
    if 'Public_Transport_Capacity' in df.columns and 'Population' in df.columns:
        df['Transport_Capacity_Per_Capita'] = df['Public_Transport_Capacity'] / df['Population']
    
    # Transport utilization rate
    if 'Public_Transport_Passengers' in df.columns and 'Public_Transport_Capacity' in df.columns:
        df['Transport_Utilization'] = df['Public_Transport_Passengers'] / df['Public_Transport_Capacity']
    
    # Vehicle to transport ratio
    if 'Vehicle_Count' in df.columns and 'Public_Transport_Fleet' in df.columns:
        df['Vehicle_Transport_Ratio'] = df['Vehicle_Count'] / df['Public_Transport_Fleet']
    
    return df


def clean_dataset(
    vehicle_df: pd.DataFrame,
    transport_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Main cleaning pipeline for all datasets.
    
    Returns:
        Tuple of (cleaned_vehicle, cleaned_transport, cleaned_traffic, cleaning_report)
    """
    report = {}
    
    # Clean vehicle data
    vehicle_df = normalize_city_names(vehicle_df)
    vehicle_df = convert_numeric_columns(vehicle_df)
    vehicle_df, missing_report = handle_missing_values(vehicle_df, strategy='mixed')
    vehicle_df, dups = remove_duplicates(vehicle_df, subset=['City', 'Year'])
    report['vehicle'] = {'missing': missing_report, 'duplicates_removed': dups}
    
    # Clean transport data
    transport_df = normalize_city_names(transport_df)
    transport_df = convert_numeric_columns(transport_df)
    transport_df, missing_report = handle_missing_values(transport_df, strategy='fill')
    transport_df, dups = remove_duplicates(transport_df, subset=['City', 'Year'])
    report['transport'] = {'missing': missing_report, 'duplicates_removed': dups}
    
    # Clean traffic data
    traffic_df = normalize_city_names(traffic_df)
    traffic_df = convert_numeric_columns(traffic_df)
    traffic_df, missing_report = handle_missing_values(traffic_df, strategy='mixed')
    traffic_df, dups = remove_duplicates(traffic_df, subset=['City', 'Year'])
    report['traffic'] = {'missing': missing_report, 'duplicates_removed': dups}
    
    if verbose:
        print("\n=== Cleaning Report ===")
        for dataset_name, dataset_report in report.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Rows removed for missing data: {dataset_report['missing']['rows_removed']}")
            print(f"  Duplicates removed: {dataset_report['duplicates_removed']}")
    
    return vehicle_df, transport_df, traffic_df, report


def integrate_datasets(
    vehicle_df: pd.DataFrame,
    transport_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    merge_on: List[str] = ['City', 'Year']
) -> pd.DataFrame:
    """
    Merge all datasets into a single analysis-ready DataFrame.
    """
    # Merge vehicle and transport data
    merged = vehicle_df.merge(transport_df, on=merge_on, how='inner')
    
    # Merge with traffic data
    merged = merged.merge(traffic_df, on=merge_on, how='inner')
    
    # Add derived features
    merged = add_derived_features(merged)
    
    print(f"\nIntegrated dataset: {len(merged)} records, {len(merged.columns)} columns")
    print(f"Cities: {merged['City'].unique().tolist()}")
    print(f"Years: {sorted(merged['Year'].unique().tolist())}")
    
    return merged
  
