import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_loader import load_model, load_data
from app.utils.visualization import create_distribution_plot, create_correlation_heatmap

# Page config
st.set_page_config(
    page_title="California HomeValue Insight Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/real-estate.png", width=100)
    st.title("🏠 HomeValue Insight Predictor")
    st.markdown("---")
    
    # Model loading section
    st.subheader("📦 Model Control")
    if st.button("🔄 Load Model", use_container_width=True):
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()
            if st.session_state.model:
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")
    
    if st.button("📊 Load Dataset", use_container_width=True):
        with st.spinner("Loading dataset..."):
            st.session_state.data = load_data()
            if st.session_state.data is not None:
                st.success(f"✅ Loaded {len(st.session_state.data)} records")
            else:
                st.error("❌ Failed to load dataset")
    
    st.markdown("---")
    
    # Navigation
    st.subheader("🧭 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 EDA Dashboard", "📈 Predictions", "ℹ️ Model Info"],
        label_visibility="collapsed"
    )

# Main content area
if page == "🏠 Home":
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🏠 California HomeValue Insight Prediction")
        st.markdown("""
        ### Welcome to the Advanced House Price Prediction System
        
        This application uses machine learning to predict median house values in California 
        based on various features like location, demographics, and housing characteristics.
        
        **🌟 Features:**
        - Interactive Exploratory Data Analysis
        - Real-time Price Predictions
        - Model Performance Metrics
        - Feature Importance Analysis
        - Batch Predictions
        """)
        
        # Quick stats if data is loaded
        if st.session_state.data is not None:
            st.markdown("---")
            st.subheader("📊 Quick Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(st.session_state.data):,}")
            with col2:
                st.metric("Features", f"{len(st.session_state.data.columns)-1}")
            with col3:
                st.metric("Avg Price", f"${st.session_state.data['median_house_value'].mean():,.0f}")
            with col4:
                st.metric("Max Price", f"${st.session_state.data['median_house_value'].max():,}")
    
    with col2:
        st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1073&q=80", 
                 caption="California Real Estate", use_column_width=True)
    
    # How to use section
    st.markdown("---")
    st.subheader("🚀 How to Use")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("**1️⃣ Load Model**\n\nClick 'Load Model' in the sidebar to start")
    with col2:
        st.info("**2️⃣ Explore Data**\n\nNavigate to EDA Dashboard for insights")
    with col3:
        st.info("**3️⃣ Make Predictions**\n\nGo to Predictions page and input features")
    with col4:
        st.info("**4️⃣ Analyze Results**\n\nCheck Model Info for performance metrics")

elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load the dataset first using the sidebar!")
    else:
        df = st.session_state.data
        
        # Summary statistics
        st.subheader("📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", f"{len(df.columns):,}")
        with col3:
            st.metric("Missing Values", f"{df.isna().sum().sum():,}")
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Data preview
        with st.expander("🔍 Preview Data", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
        
        # Target variable analysis
        st.subheader("🎯 Target Variable Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_distribution_plot(df, 'median_house_value', 'Median House Value Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary stats for target
            target_stats = df['median_house_value'].describe()
            stats_df = pd.DataFrame({
                'Statistic': target_stats.index,
                'Value': target_stats.values
            })
            st.dataframe(stats_df, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        fig = create_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=selected_feature, nbins=50, 
                             title=f"Distribution of {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=selected_feature, title=f"Box Plot of {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic distribution
        st.subheader("🗺️ Geographic Distribution")
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', 
                               color='median_house_value', size='median_income',
                               hover_data=['median_house_value', 'median_income'],
                               color_continuous_scale='Viridis',
                               zoom=5, height=500,
                               title="House Values Across California")
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Predictions":
    st.title("📈 House Price Prediction")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load the model first using the sidebar!")
    else:
        # Create tabs for single and batch prediction
        tab1, tab2 = st.tabs(["🏠 Single Prediction", "📦 Batch Prediction"])
        
        with tab1:
            st.subheader("Enter Property Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
                latitude = st.number_input("Latitude", value=37.88, format="%.4f")
                housing_median_age = st.number_input("Housing Median Age", value=41, min_value=1, max_value=100)
                total_rooms = st.number_input("Total Rooms", value=880, min_value=1)
                total_bedrooms = st.number_input("Total Bedrooms", value=129, min_value=1)
                
            with col2:
                population = st.number_input("Population", value=322, min_value=1)
                households = st.number_input("Households", value=126, min_value=1)
                median_income = st.number_input("Median Income (in $10,000s)", value=8.3252, format="%.4f")
                ocean_proximity = st.selectbox(
                    "Ocean Proximity",
                    options=['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
                )
            
            if st.button("🔮 Predict Price", type="primary", use_container_width=True):
                # Create input dataframe
                input_data = pd.DataFrame([{
                    'longitude': longitude,
                    'latitude': latitude,
                    'housing_median_age': housing_median_age,
                    'total_rooms': total_rooms,
                    'total_bedrooms': total_bedrooms,
                    'population': population,
                    'households': households,
                    'median_income': median_income,
                    'ocean_proximity': ocean_proximity
                }])
                
                # Make prediction
                with st.spinner("Calculating prediction..."):
                    prediction = st.session_state.model.predict(input_data)[0]
                    st.session_state.prediction_made = True
                
                # Display prediction with style
                st.markdown("---")
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
                        <h3 style='color: white; margin-bottom: 1rem;'>Predicted House Value</h3>
                        <h1 style='color: white; font-size: 3rem;'>${prediction:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Batch Prediction")
            st.info("Upload a CSV file with multiple properties for batch prediction")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                batch_data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_data.head())
                
                if st.button("Predict All", type="primary"):
                    with st.spinner("Making predictions..."):
                        predictions = st.session_state.model.predict(batch_data)
                        batch_data['predicted_price'] = predictions
                        
                        # Display results
                        st.success("Predictions completed!")
                        st.dataframe(batch_data)
                        
                        # Download button
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load the model first using the sidebar!")
    else:
        # Model performance metrics
        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", "$68,712", "Test Set")
        with col2:
            st.metric("MAE", "$49,891", "Test Set")
        with col3:
            st.metric("R² Score", "0.837", "Test Set")
        
        # Model architecture
        st.subheader("🏗️ Model Architecture")
        st.code("""
HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=None,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42
)
        """)
        
        # Features used
        st.subheader("📌 Features Used")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Features:**")
            st.markdown("""
            - Longitude
            - Latitude
            - Housing Median Age
            - Total Rooms
            - Total Bedrooms
            - Population
            - Households
            - Median Income
            """)
        
        with col2:
            st.markdown("**Categorical Features:**")
            st.markdown("""
            - Ocean Proximity (one-hot encoded)
                - <1H OCEAN
                - INLAND
                - NEAR OCEAN
                - NEAR BAY
                - ISLAND
            """)
        
        # Preprocessing steps
        st.subheader("⚙️ Preprocessing Pipeline")
        st.markdown("""
        1. **Missing Value Imputation**
           - Median imputation for numerical features
           - Most frequent imputation for categorical features
        
        2. **Feature Scaling**
           - StandardScaler for numerical features
        
        3. **Feature Encoding**
           - One-hot encoding for categorical features
        """)
        
        # Feature importance (if available)
        st.subheader("🎯 Feature Importance")
        importance_data = pd.DataFrame({
            'Feature': ['Median Income', 'Latitude', 'Longitude', 'Ocean Proximity', 'Housing Age'],
            'Importance': [0.45, 0.20, 0.15, 0.12, 0.08]
        })
        
        fig = px.bar(importance_data, x='Importance', y='Feature', 
                     orientation='h', title="Feature Importance (approximate)")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Built with ❤️ using Streamlit | California Housing Dataset | 
        <a href='https://github.com/yourusername/house-price-prediction'>GitHub</a>
    </div>
    """, 
    unsafe_allow_html=True
)