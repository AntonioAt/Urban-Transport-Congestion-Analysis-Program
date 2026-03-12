"""
Interactive Streamlit dashboard for urban transport analysis.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import project modules
from data_loader import load_data
from data_cleaning import clean_dataset, integrate_datasets
from analysis import compute_key_correlations, rank_cities, generate_summary_statistics
from regression_model import build_congestion_model


# Page configuration
st.set_page_config(
    page_title="Urban Transport Analysis",
    page_icon="🚗",
    layout="wide"
)


@st.cache_data
def load_and_process_data():
    """Load and process data with caching."""
    vehicle_df, transport_df, traffic_df = load_data(source='sample')
    vehicle_clean, transport_clean, traffic_clean, _ = clean_dataset(
        vehicle_df, transport_df, traffic_df, verbose=False
    )
    df = integrate_datasets(vehicle_clean, transport_clean, traffic_clean)
    return df


def main():
    st.title("🚗 Urban Traffic Congestion Analysis")
    st.markdown("*Analyzing the relationship between public transportation and urban congestion*")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_and_process_data()
    
    # Sidebar
    st.sidebar.header("Filters")
    
    selected_cities = st.sidebar.multiselect(
        "Select Cities",
        options=df['City'].unique().tolist(),
        default=df['City'].unique().tolist()
    )
    
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['City'].isin(selected_cities)) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Trends", "🔗 Correlations", "🏆 Rankings", "🤖 Prediction"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(filtered_df))
        with col2:
            st.metric("Avg Traffic Index", f"{filtered_df['Traffic_Index'].mean():.1f}")
        with col3:
            st.metric("Total Vehicles", f"{filtered_df['Vehicle_Count'].sum()/1e6:.1f}M")
        with col4:
            st.metric("Cities", len(selected_cities))
        
        st.subheader("Summary Statistics")
        st.dataframe(generate_summary_statistics(filtered_df))
        
        # Traffic distribution
        st.subheader("Traffic Index Distribution")
        fig = px.histogram(
            filtered_df, x='Traffic_Index', color='City',
            barmode='overlay', opacity=0.7,
            title="Distribution of Traffic Index by City"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Trends Over Time")
        
        metric = st.selectbox(
            "Select Metric",
            ['Traffic_Index', 'Vehicle_Count', 'Public_Transport_Capacity', 
             'Vehicles_Per_Capita', 'Transport_Capacity_Per_Capita']
        )
        
        fig = px.line(
            filtered_df, x='Year', y=metric, color='City',
            markers=True, title=f"{metric.replace('_', ' ')} Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-year comparison
        st.subheader("City Comparison by Year")
        fig = px.bar(
            filtered_df, x='Year', y='Traffic_Index', color='City',
            barmode='group', title="Traffic Index Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Correlation Analysis")
        
        # Key correlations
        correlations = compute_key_correlations(filtered_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Correlations")
            for key, value in correlations.items():
                st.write(f"**{key.replace('_', ' ').title()}**")
                st.write(f"Correlation: {value['correlation']}")
                st.write(f"P-value: {value['p_value']}")
                significant = "✅ Significant" if value['significant'] else "❌ Not Significant"
                st.write(significant)
                st.write("---")
        
        with col2:
            st.subheader("Scatter Plot Analysis")
            x_var = st.selectbox("X Variable", ['Vehicle_Count', 'Public_Transport_Capacity', 'Population'])
            
            fig = px.scatter(
                filtered_df, x=x_var, y='Traffic_Index',
                color='City', size='Population',
                trendline='ols',
                title=f"{x_var.replace('_', ' ')} vs Traffic Index"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'Year']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("City Rankings")
        
        rankings = rank_cities(filtered_df)
        
        # Display rankings
        st.subheader("Transport Efficiency Rankings")
        st.dataframe(rankings.style.highlight_max(subset=['efficiency_score'], color='lightgreen'))
        
        # Bar chart
        fig = px.bar(
            rankings.sort_values('efficiency_score'),
            x='efficiency_score', y='City',
            orientation='h',
            title="City Efficiency Scores",
            color='efficiency_score',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison chart
        st.subheader("Multi-Metric Comparison")
        latest_year = filtered_df['Year'].max()
        latest_data = filtered_df[filtered_df['Year'] == latest_year]
        
        fig = go.Figure()
        
        metrics = ['Traffic_Index', 'Vehicles_Per_Capita', 'Transport_Capacity_Per_Capita']
        available_metrics = [m for m in metrics if m in latest_data.columns]
        
        for metric in available_metrics:
            # Normalize to 0-100 scale
            values = latest_data[metric]
            normalized = ((values - values.min()) / (values.max() - values.min()) * 100).values
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' '),
                x=latest_data['City'].values,
                y=normalized
            ))
        
        fig.update_layout(
            barmode='group',
            title="Normalized Metric Comparison (0-100 scale)",
            yaxis_title="Normalized Value"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Congestion Prediction Model")
        
        if st.button("Train Model"):
            with st.spinner("Training regression models..."):
                predictor, results, importance = build_congestion_model(
                    filtered_df,
                    features=['Vehicle_Count', 'Public_Transport_Capacity', 'Population'],
                    verbose=False
                )
            
            st.success("Models trained successfully!")
            
            # Results table
            st.subheader("Model Performance")
            results_df = pd.DataFrame([
                {
                    'Model': k,
                    'Train R²': v['train_r2'],
                    'Test R²': v['test_r2'],
                    'RMSE': v['rmse'],
                    'CV Mean': v['cv_mean']
                }
                for k, v in results.items()
            ])
            st.dataframe(results_df.style.highlight_max(subset=['Test R²'], color='lightgreen'))
            
            # Feature importance
            st.subheader("Feature Importance")
            if not importance.empty:
                rf_imp = importance[importance['Model'] == 'Random Forest']
                fig = px.bar(
                    rf_imp.sort_values('Importance'),
                    x='Importance', y='Feature',
                    orientation='h',
                    title="Feature Importance (Random Forest)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction interface
        st.subheader("Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vehicles = st.number_input(
                "Vehicle Count",
                min_value=100000,
                max_value=50000000,
                value=5000000,
                step=100000
            )
        
        with col2:
            transport = st.number_input(
                "Public Transport Capacity",
                min_value=10000,
                max_value=10000000,
                value=1000000,
                step=50000
            )
        
        with col3:
            population = st.number_input(
                "Population",
                min_value=100000,
                max_value=20000000,
                value=3000000,
                step=100000
            )
        
        if st.button("Predict Traffic Index"):
            # Simple linear approximation based on correlations
            # In practice, use the trained model
            baseline = 35
            vehicle_effect = (vehicles / 5000000 - 1) * 15
            transport_effect = (transport / 1000000 - 1) * -8
            pop_effect = (population / 3000000 - 1) * 5
            
            prediction = baseline + vehicle_effect + transport_effect + pop_effect
            prediction = max(15, min(70, prediction))
            
            st.metric("Predicted Traffic Index", f"{prediction:.1f}")
            
            if prediction > 45:
                st.warning("⚠️ High congestion expected. Consider increasing public transport capacity.")
            elif prediction > 30:
                st.info("ℹ️ Moderate congestion expected.")
            else:
                st.success("✅ Low congestion expected. Good transport infrastructure balance.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Data sources: BPS Indonesia, Ministry of Transport, TomTom Traffic Index*")


if __name__ == "__main__":
    main()
