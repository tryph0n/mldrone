"""
DroneDetect V2 - Streamlit Application

Entry point for the Streamlit application.
Run with: streamlit run interface/app.py
"""
import sys
from pathlib import Path

# Add interface to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from settings import PAGE_CONFIG

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Navigation
PAGES = {
    "Home": "home",
    "Model Comparison": "comparison",
    "Inference": "inference",
    "Glossary": "glossary",
}

# Sidebar navigation
st.sidebar.title("DroneDetect V2")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", list(PAGES.keys()))

st.sidebar.divider()

# Render selected page
if PAGES[page] == "home":
    from views.home_view import HomeView
    HomeView().render()

elif PAGES[page] == "comparison":
    from views.model_comparison_view import ModelComparisonView
    ModelComparisonView().render()

elif PAGES[page] == "inference":
    from views.inference_view import InferenceView
    st.title("Model Inference")
    InferenceView().render()

elif PAGES[page] == "glossary":
    from views.glossary_view import GlossaryView
    GlossaryView().render()
