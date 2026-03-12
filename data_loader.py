"""
Data loading module for urban transport analysis.
Handles loading data from CSV files or generating sample data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def generate_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic sample data for analysis when real data isn't available.
    Based on approximate real-world figures for Indonesian cities and Singapore.
    """
    np.random.seed(42)
    
    cities = ['Jakarta', 'Surabaya', 'Bandung', 'Singapore']
    years = [2020, 2021, 2022, 2023]
    
    # Vehicle data with realistic estimates
    vehicle_records = []
    base_vehicles = {
        'Jakarta': 22_000_000,
        'Surabaya': 7_500_000,
        'Bandung': 3_200_000,
        'Singapore': 980_000
    }
    populations = {
        'Jakarta': 10_560_000,
        'Surabaya': 2_870_000,
        'Bandung': 2_440_000,
        'Singapore': 5_450_000
    }
    
    for city in cities:
        for i, year in enumerate(years):
            growth_rate = np.random.uniform(0.02, 0.06) if city != 'Singapore' else np.random.uniform(-0.01, 0.02)
            vehicle_count = int(base_vehicles[city] * (1 + growth_rate) ** i)
            pop_growth = np.random.uniform(0.01, 0.02)
            population = int(populations[city] * (1 + pop_growth) ** i)
            
            vehicle_records.append({
                'City': city,
                'Year': year,
                'Vehicle_Count': vehicle_count,
                'Population': population,
                'Vehicle_Growth_Rate': round(growth_rate * 100, 2)
            })
    
    vehicle_df = pd.DataFrame(vehicle_records)
    
    # Public transport data
    transport_records = []
    base_fleet = {
        'Jakarta': 10_500,
        'Surabaya': 2_800,
        'Bandung': 1_500,
        'Singapore': 6_200
    }
    base_capacity = {
        'Jakarta': 3_500_000,
        'Surabaya': 850_000,
        'Bandung': 420_000,
        'Singapore': 4_200_000
    }
    
    for city in cities:
        for i, year in enumerate(years):
            fleet_growth = np.random.uniform(0.03, 0.08)
            fleet = int(base_fleet[city] * (1 + fleet_growth) ** i)
            capacity = int(base_capacity[city] * (1 + fleet_growth * 1.2) ** i)
            passengers = int(capacity * np.random.uniform(0.6, 0.85))
            
            transport_records.append({
                'City': city,
                'Year': year,
                'Public_Transport_Fleet': fleet,
                'Public_Transport_Passengers': passengers,
                'Public_Transport_Capacity': capacity
            })
    
    transport_df = pd.DataFrame(transport_records)
    
    # Traffic congestion data (TomTom-style index)
    traffic_records = []
    base_congestion = {
        'Jakarta': 53,
        'Surabaya': 38,
        'Bandung': 35,
        'Singapore': 27
    }
    
    for city in cities:
        for i, year in enumerate(years):
            # Congestion affected by vehicle growth and transport capacity
            vehicle_effect = (vehicle_df[(vehicle_df['City'] == city) & 
                            (vehicle_df['Year'] == year)]['Vehicle_Count'].values[0] / 
                            base_vehicles[city] - 1) * 10
            transport_effect = (transport_df[(transport_df['City'] == city) & 
                               (transport_df['Year'] == year)]['Public_Transport_Capacity'].values[0] / 
                               base_capacity[city] - 1) * -8
            
            traffic_index = base_congestion[city] + vehicle_effect + transport_effect + np.random.uniform(-3, 3)
            traffic_index = max(15, min(70, traffic_index))
            
            if traffic_index > 45:
                congestion_level = 'High'
            elif traffic_index > 30:
                congestion_level = 'Medium'
            else:
                congestion_level = 'Low'
            
            avg_travel_time = 30 + traffic_index * 0.8 + np.random.uniform(-5, 5)
            
            traffic_records.append({
                'City': city,
                'Year': year,
                'Traffic_Index': round(traffic_index, 1),
                'Congestion_Level': congestion_level,
                'Average_Travel_Time': round(avg_travel_time, 1)
            })
    
    traffic_df = pd.DataFrame(traffic_records)
    
    return vehicle_df, transport_df, traffic_df


def load_from_csv(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from CSV files.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Tuple of (vehicle_df, transport_df, traffic_df)
    """
    data_path = Path(data_dir)
    
    vehicle_file = data_path / 'vehicle_data.csv'
    transport_file = data_path / 'transport_data.csv'
    traffic_file = data_path / 'traffic_data.csv'
    
    if all(f.exists() for f in [vehicle_file, transport_file, traffic_file]):
        vehicle_df = pd.read_csv(vehicle_file)
        transport_df = pd.read_csv(transport_file)
        traffic_df = pd.read_csv(traffic_file)
        print(f"Loaded data from {data_dir}/")
        return vehicle_df, transport_df, traffic_df
    else:
        print("CSV files not found. Generating sample data...")
        return generate_sample_data()


def save_sample_data(data_dir: str = 'data') -> None:
    """Generate and save sample data to CSV files."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    vehicle_df, transport_df, traffic_df = generate_sample_data()
    
    vehicle_df.to_csv(data_path / 'vehicle_data.csv', index=False)
    transport_df.to_csv(data_path / 'transport_data.csv', index=False)
    traffic_df.to_csv(data_path / 'traffic_data.csv', index=False)
    
    print(f"Sample data saved to {data_dir}/")


def load_data(source: str = 'auto', data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main data loading function.
    
    Args:
        source: 'csv', 'sample', or 'auto' (tries CSV first, falls back to sample)
        data_dir: Directory for CSV files
        
    Returns:
        Tuple of (vehicle_df, transport_df, traffic_df)
    """
    if source == 'sample':
        print("Generating sample data...")
        return generate_sample_data()
    elif source == 'csv':
        return load_from_csv(data_dir)
    else:  # auto
        return load_from_csv(data_dir)
