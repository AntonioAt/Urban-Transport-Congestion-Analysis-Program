"""
Main execution module for Urban Traffic Congestion Analysis.
Orchestrates the complete analysis pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Import project modules
from data_loader import load_data, save_sample_data
from data_cleaning import clean_dataset, integrate_datasets
from analysis import (
    generate_summary_statistics,
    analyze_by_city,
    compute_correlation_matrix,
    compute_key_correlations,
    rank_cities,
    perform_anova,
    generate_insights
)
from regression_model import build_congestion_model, interpret_coefficients
from visualization import create_all_visualizations


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def generate_report(
    df: pd.DataFrame,
    summary_stats: pd.DataFrame,
    city_stats: pd.DataFrame,
    correlations: dict,
    rankings: pd.DataFrame,
    model_results: dict,
    insights: list,
    output_dir: str = 'output'
) -> str:
    """
    Generate a comprehensive analysis report.
    """
    report_lines = [
        "=" * 70,
        "URBAN TRAFFIC CONGESTION ANALYSIS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
    ]
    
    # Add insights
    for i, insight in enumerate(insights, 1):
        report_lines.append(f"{i}. {insight}")
    
    report_lines.extend([
        "",
        "DATA OVERVIEW",
        "-" * 40,
        f"Total Records: {len(df)}",
        f"Cities Analyzed: {', '.join(df['City'].unique())}",
        f"Time Period: {df['Year'].min()} - {df['Year'].max()}",
        "",
        "SUMMARY STATISTICS",
        "-" * 40,
    ])
    report_lines.append(summary_stats.to_string())
    
    report_lines.extend([
        "",
        "KEY CORRELATIONS",
        "-" * 40,
    ])
    
    for key, value in correlations.items():
        report_lines.append(f"{key.replace('_', ' ').title()}:")
        report_lines.append(f"  Correlation: {value['correlation']}")
        report_lines.append(f"  P-value: {value['p_value']}")
        report_lines.append(f"  Significant: {'Yes' if value['significant'] else 'No'}")
        report_lines.append("")
    
    report_lines.extend([
        "CITY RANKINGS (Transport Efficiency)",
        "-" * 40,
    ])
    report_lines.append(rankings.to_string())
    
    report_lines.extend([
        "",
        "REGRESSION MODEL PERFORMANCE",
        "-" * 40,
    ])
    
    for model_name, metrics in model_results.items():
        report_lines.append(f"{model_name}:")
        report_lines.append(f"  Test R²: {metrics['test_r2']}")
        report_lines.append(f"  RMSE: {metrics['rmse']}")
        report_lines.append(f"  CV Mean: {metrics['cv_mean']}")
        report_lines.append("")
    
    report_lines.extend([
        "METHODOLOGY",
        "-" * 40,
        "1. Data collected from multiple sources (BPS, Ministry of Transport, TomTom)",
        "2. Data cleaned and integrated by city and year",
        "3. Exploratory analysis and correlation computation",
        "4. Multiple regression models trained and compared",
        "5. Feature importance analysis conducted",
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save report
    Path(output_dir).mkdir(exist_ok=True)
    report_path = Path(output_dir) / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    """
    Main execution function for the analysis pipeline.
    """
    print_section("URBAN TRAFFIC CONGESTION ANALYSIS")
    print("Analyzing relationship between public transportation and urban congestion")
    
    # Configuration
    OUTPUT_DIR = 'output'
    DATA_DIR = 'data'
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Step 1: Load Data
    print_section("STEP 1: DATA LOADING")
    save_sample_data(DATA_DIR)  # Generate sample data
    vehicle_df, transport_df, traffic_df = load_data(source='csv', data_dir=DATA_DIR)
    
    print(f"\nVehicle data: {len(vehicle_df)} records")
    print(f"Transport data: {len(transport_df)} records")
    print(f"Traffic data: {len(traffic_df)} records")
    
    # Step 2: Data Cleaning
    print_section("STEP 2: DATA CLEANING")
    vehicle_clean, transport_clean, traffic_clean, cleaning_report = clean_dataset(
        vehicle_df, transport_df, traffic_df, verbose=True
    )
    
    # Step 3: Data Integration
    print_section("STEP 3: DATA INTEGRATION")
    df = integrate_datasets(vehicle_clean, transport_clean, traffic_clean)
    
    # Save cleaned dataset
    df.to_csv(Path(OUTPUT_DIR) / 'cleaned_integrated_data.csv', index=False)
    print(f"\nCleaned data saved to '{OUTPUT_DIR}/cleaned_integrated_data.csv'")
    
    # Step 4: Exploratory Data Analysis
    print_section("STEP 4: EXPLORATORY DATA ANALYSIS")
    
    summary_stats = generate_summary_statistics(df)
    print("\n--- Summary Statistics ---")
    print(summary_stats)
    
    city_stats = analyze_by_city(df)
    print("\n--- Statistics by City ---")
    print(city_stats[['Traffic_Index_mean', 'Vehicle_Count_mean', 'Public_Transport_Capacity_mean']])
    
    # Step 5: Correlation Analysis
    print_section("STEP 5: CORRELATION ANALYSIS")
    
    corr_matrix = compute_correlation_matrix(df)
    print("\n--- Correlation Matrix (Key Variables) ---")
    key_cols = ['Traffic_Index', 'Vehicle_Count', 'Public_Transport_Capacity', 'Population']
    key_cols = [c for c in key_cols if c in corr_matrix.columns]
    print(corr_matrix.loc[key_cols, key_cols])
    
    key_correlations = compute_key_correlations(df)
    print("\n--- Key Correlation Results ---")
    for key, value in key_correlations.items():
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"  Correlation: {value['correlation']}, P-value: {value['p_value']}")
        print(f"  Significant: {'Yes' if value['significant'] else 'No'}")
    
    # Step 6: City Rankings
    print_section("STEP 6: CITY RANKINGS")
    
    rankings = rank_cities(df)
    print("\n--- City Rankings by Transport Efficiency ---")
    print(rankings)
    
    # ANOVA test
    anova_result = perform_anova(df)
    print(f"\n--- ANOVA Test ---")
    print(f"F-statistic: {anova_result['f_statistic']}, P-value: {anova_result['p_value']}")
    print(f"Interpretation: {anova_result['interpretation']}")
    
    # Step 7: Regression Modeling
    print_section("STEP 7: REGRESSION MODELING")
    
    predictor, model_results, feature_importance = build_congestion_model(
        df,
        target='Traffic_Index',
        features=['Vehicle_Count', 'Public_Transport_Capacity', 'Population'],
        verbose=True
    )
    
    # Generate predictions for visualization
    X, y = predictor.prepare_features(df, 'Traffic_Index', 
                                       ['Vehicle_Count', 'Public_Transport_Capacity', 'Population'])
    predictions = predictor.predict(X)
    
    # Coefficient interpretation
    print("\n--- Coefficient Interpretations ---")
    for interpretation in interpret_coefficients(predictor):
        print(f"• {interpretation}")
    
    # Save model results
    model_results_clean = {k: {kk: vv for kk, vv in v.items() if kk != 'model'} 
                          for k, v in model_results.items()}
    with open(Path(OUTPUT_DIR) / 'model_results.json', 'w') as f:
        json.dump(model_results_clean, f, indent=2)
    
    # Step 8: Visualization
    print_section("STEP 8: VISUALIZATION")
    
    saved_files = create_all_visualizations(
        df,
        feature_importance=feature_importance,
        predictions=predictions,
        output_dir=OUTPUT_DIR
    )
    
    # Step 9: Generate Insights
    print_section("STEP 9: INSIGHTS GENERATION")
    
    insights = generate_insights(df, key_correlations, rankings)
    print("\n--- Key Insights ---")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Step 10: Generate Report
    print_section("STEP 10: REPORT GENERATION")
    
    report = generate_report(
        df, summary_stats, city_stats, key_correlations,
        rankings, model_results_clean, insights, OUTPUT_DIR
    )
    
    print(f"\nReport saved to '{OUTPUT_DIR}/analysis_report.txt'")
    
    # Summary
    print_section("ANALYSIS COMPLETE")
    print(f"""
Output Files Generated:
-----------------------
• {OUTPUT_DIR}/cleaned_integrated_data.csv  - Cleaned and merged dataset
• {OUTPUT_DIR}/model_results.json           - Regression model performance
• {OUTPUT_DIR}/analysis_report.txt          - Full analysis report
• {OUTPUT_DIR}/*.png                        - Visualization charts

Research Questions Answered:
----------------------------
1. Does public transport capacity reduce urban congestion?
   → {'Yes' if key_correlations.get('transport_capacity_vs_congestion', {}).get('correlation', 0) < 0 else 'Correlation suggests otherwise'}
   
2. How strongly is vehicle count correlated with congestion?
   → Correlation: {key_correlations.get('vehicle_count_vs_congestion', {}).get('correlation', 'N/A')}
   
3. Which city has the best transport-traffic balance?
   → {rankings.iloc[0]['City'] if not rankings.empty else 'N/A'}
""")
    
    return df, predictor, insights


if __name__ == "__main__":
    df, predictor, insights = main()
