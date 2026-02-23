import streamlit as st
import time

def loading_animation():
    loading_html = """
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Loading...</div>
    </div>
    <style>
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-right: 5px solid #764ba2;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            margin-top: 1rem;
            color: #667eea;
            font-weight: 600;
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)