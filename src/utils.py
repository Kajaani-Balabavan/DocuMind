import streamlit as st
import plotly.express as px
import pandas as pd
from typing import List, Dict

def display_chat_message(role: str, content: str):
    """Display chat message with proper styling"""
    with st.chat_message(role):
        st.write(content)

def create_confidence_chart(confidence: float):
    """Create a confidence score visualization"""
    fig = px.bar(
        x=['Confidence'],
        y=[confidence],
        title="Answer Confidence Score",
        color=[confidence],
        color_continuous_scale="RdYlGn",
        range_color=[0, 1]
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        yaxis_range=[0, 1]
    )
    return fig

# def format_sources(sources: List[str], max_length: int = 200):
#     """Format source documents for display"""
#     formatted_sources = []
#     for i, source in enumerate(sources, 1):
#         truncated = source[:max_length] + "..." if len(source) > max_length else source
#         formatted_sources.append(f"**Source {i}:** {truncated}")
#     return formatted_sources

def format_sources(sources: List[str], max_length: int = None):
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        truncated = source if max_length is None else source[:max_length] + "..." if len(source) > max_length else source
        formatted_sources.append(f"**Source {i}:** {truncated}")
    return formatted_sources


def validate_file_upload(uploaded_file):
    """Validate uploaded file"""
    if uploaded_file is None:
        return False, "Please upload a file."
    
    allowed_extensions = ['.pdf', '.txt', '.docx']
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    if f'.{file_extension}' not in allowed_extensions:
        return False, f"Unsupported file type. Please upload: {', '.join(allowed_extensions)}"
    
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
        return False, "File size too large. Please upload files smaller than 10MB."
    
    return True, "File is valid."

def get_system_stats(vector_store):
    """Get system statistics"""
    return {
        "Documents Processed": len(set([meta.get('filename', 'Unknown') for meta in vector_store.metadata])),
        "Text Chunks": len(vector_store.texts),
        "Vector Dimension": vector_store.dimension,
        "Index Size": vector_store.index.ntotal
    }