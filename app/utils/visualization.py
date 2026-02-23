import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_distribution_plot(df, column, title):
    """Create distribution plot for a column"""
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Histogram', 'Box Plot'))
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[column], nbinsx=50, name='Histogram', 
                    marker_color='#636EFA'),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=df[column], name='Box Plot', marker_color='#EF553B'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text=title, showlegend=False)
    fig.update_xaxes(title_text=column, row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, title=None):
    """Create scatter plot"""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                    title=title or f"{y_col} vs {x_col}",
                    trendline="ols")
    
    fig.update_layout(height=500)
    return fig

def create_geographic_map(df, lat_col='latitude', lon_col='longitude', 
                          color_col='median_house_value', size_col='median_income'):
    """Create geographic scatter map"""
    fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col,
                           color=color_col, size=size_col,
                           hover_data=['median_house_value', 'median_income'],
                           color_continuous_scale='Viridis',
                           zoom=5, height=500)
    
    fig.update_layout(mapbox_style="open-street-map",
                     title="Geographic Distribution of House Values")
    
    return fig