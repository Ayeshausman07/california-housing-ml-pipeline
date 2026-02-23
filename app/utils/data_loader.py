import pandas as pd
import joblib
import streamlit as st
import os
from pathlib import Path

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent.parent / "model" / "best_model.pkl",
            Path("model/best_model.pkl"),
            Path("../model/best_model.pkl")
        ]
        
        for path in possible_paths:
            if path.exists():
                model_data = joblib.load(path)
                return model_data['model']
        
        st.error("Model file not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load the housing dataset"""
    try:
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent.parent / "data" / "housing.csv",
            Path("data/housing.csv"),
            Path("../data/housing.csv")
        ]
        
        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path)
                return df
        
        st.error("Dataset not found. Please ensure housing.csv is in the data folder.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def validate_input_data(data):
    """Validate input data for prediction"""
    required_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                       'total_bedrooms', 'population', 'households', 'median_income',
                       'ocean_proximity']
    
    missing_cols = [col for col in required_columns if col not in data.columns]
    
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check data types
    numeric_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                   'total_bedrooms', 'population', 'households', 'median_income']
    
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            return False, f"Column {col} must be numeric"
    
    return True, "Validation passed"