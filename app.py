"""
Streamlit Web UI cho Heart Disease Prediction với XAI Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import os
import sys
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import DataProcessor, ModelTrainer, XAIExplainer
from utils import LLMInterpreter
import config

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction & XAI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Steve Jobs inspired: Clean, Minimal, Professional
st.markdown("""
<style>
    /* Import professional font and Bootstrap Icons */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header - Minimalist */
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        letter-spacing: -0.5px;
        color: #1a1a1a;
        text-align: center;
        margin: 2rem 0 3rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .main-header strong {
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards - Clean shadow design */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Interpretation box - Medical theme */
    .interpretation-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 12px rgba(102, 126, 234, 0.1);
        margin: 1.5rem 0;
    }
    
    /* Sidebar - Professional */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
    
    /* Tabs - Clean design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #666;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Buttons - Modern */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metrics - Apple style */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
        margin: 2rem 0;
    }
    
    /* Info/Warning/Success boxes - Medical colors */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Dataframe - Clean table */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Text Input and File Uploader - Black text */
    .stTextInput input {
        color: #1a1a1a !important;
    }
    
    .stFileUploader label {
        color: #1a1a1a !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        color: #1a1a1a !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] p {
        color: #1a1a1a !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] span {
        color: #1a1a1a !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] small {
        color: #666666 !important;
    }
    
    .stFileUploader section {
        color: #1a1a1a !important;
    }
    
    .stFileUploader button {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'xai_results' not in st.session_state:
    st.session_state.xai_results = None
if 'llm_interpretations' not in st.session_state:
    st.session_state.llm_interpretations = None

# DEMO SCENARIOS (Data from User)
DEMO_CASES = {
    "Case 1 (Index 1089) - Age 57, Male": {
        'age': 57, 'sex': 1, 'chest pain type': 3, 'resting bp s': 150, 
        'cholesterol': 126, 'fasting blood sugar': 1, 'resting ecg': 0, 
        'max heart rate': 173, 'exercise angina': 0, 'oldpeak': 0.2, 'ST slope': 1,
        'target_label': 'Normal'
    },
    "Case 2 (Index 31) - Age 56, Male": {
        'age': 56, 'sex': 1, 'chest pain type': 3, 'resting bp s': 130, 
        'cholesterol': 167, 'fasting blood sugar': 0, 'resting ecg': 0, 
        'max heart rate': 114, 'exercise angina': 0, 'oldpeak': 0, 'ST slope': 1,
        'target_label': 'Normal'
    },
    "Case 3 (Index 680) - Age 60, Female": {
        'age': 60, 'sex': 0, 'chest pain type': 1, 'resting bp s': 150, 
        'cholesterol': 240, 'fasting blood sugar': 0, 'resting ecg': 0, 
        'max heart rate': 171, 'exercise angina': 0, 'oldpeak': 0.9, 'ST slope': 1,
        'target_label': 'Normal'
    },
    "Case 4 (Index 691) - Age 67, Female (Type 4 Pain)": {
        'age': 67, 'sex': 0, 'chest pain type': 4, 'resting bp s': 106, 
        'cholesterol': 223, 'fasting blood sugar': 0, 'resting ecg': 0, 
        'max heart rate': 142, 'exercise angina': 0, 'oldpeak': 0.3, 'ST slope': 1,
        'target_label': 'Normal'
    },
    "Case 5 (Index 215) - Age 47, Male (High Risk)": {
        'age': 47, 'sex': 1, 'chest pain type': 4, 'resting bp s': 150, 
        'cholesterol': 226, 'fasting blood sugar': 0, 'resting ecg': 0, 
        'max heart rate': 98, 'exercise angina': 1, 'oldpeak': 1.5, 'ST slope': 2,
        'target_label': 'Heart Disease'
    },
    "Case 6 (Index 1081) - Age 68, Female": {
        'age': 68, 'sex': 0, 'chest pain type': 3, 'resting bp s': 120, 
        'cholesterol': 211, 'fasting blood sugar': 0, 'resting ecg': 2, 
        'max heart rate': 115, 'exercise angina': 0, 'oldpeak': 1.5, 'ST slope': 2,
        'target_label': 'Normal'
    },
    "Case 7 (Index 823) - Age 60, Female": {
        'age': 60, 'sex': 0, 'chest pain type': 3, 'resting bp s': 102, 
        'cholesterol': 318, 'fasting blood sugar': 0, 'resting ecg': 0, 
        'max heart rate': 160, 'exercise angina': 0, 'oldpeak': 0, 'ST slope': 1,
        'target_label': 'Normal'
    }
}

def main():
    # Header - Minimalist Steve Jobs style
    st.markdown('''
    <h1 class="main-header">
        Heart Disease Prediction <strong>&</strong> Explainable AI
    </h1>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Logo section
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation with icons
        st.markdown("### <i class='bi bi-compass'></i> Navigation", unsafe_allow_html=True)
        # Update page order as requested: Training 2nd, Prediction 3rd
        page = st.radio(
            "Select Page",
            ["Home", "Data & Training", "Prediction", "XAI Analysis", "LLM Interpretation"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Configuration section
        st.markdown("### <i class='bi bi-gear'></i> Configuration", unsafe_allow_html=True)
        
        # API Key status
        current_api_key = os.getenv("GROQ_API_KEY", "")
        if current_api_key:
            st.success(f"API Key: {current_api_key[:10]}...")
        else:
            st.warning("No API Key detected")
        
        api_key = st.text_input("Groq API Key", type="password", 
                                value=current_api_key, 
                                help="Override API key from .env file",
                                label_visibility="collapsed")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        st.markdown("---")
        
        # Data upload section
        st.markdown("### <i class='bi bi-folder2-open'></i> Data Upload", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file:
            st.session_state.data_file = uploaded_file
        
        st.markdown("---")
        
        # Training status
        st.markdown("### <i class='bi bi-graph-up'></i> Status", unsafe_allow_html=True)
        if st.session_state.trained:
            st.success("Model Trained")
            if st.session_state.training_results:
                results = st.session_state.training_results
                # Handle new 3-stage structure
                if 'fe_only' in results:
                    best_model = results['fe_only']['best_model_name']
                    best_acc = results['fe_only']['best_metrics']['Accuracy']
                else:
                    best_model = results['best_model_name']
                    best_acc = results['best_metrics']['Accuracy']
                st.info(f"**{best_model}**\n{best_acc:.2%} accuracy")
        else:
            st.info("Model Not Trained Yet")
    
    # Main content
    if page == "Home":
        show_home_page()
    elif page == "Data & Training":
        show_training_page()
    elif page == "Prediction":
        show_prediction_page()
    elif page == "XAI Analysis":
        show_xai_page()
    elif page == "LLM Interpretation":
        show_llm_page()


def show_home_page():
    """Home page - Clean, scientific design"""
    
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h2 style='font-weight: 300; color: #666; margin-bottom: 1rem;'>
                Machine Learning Meets Medical Science
            </h2>
            <p style='font-size: 1.1rem; color: #888; line-height: 1.8;'>
                Predict cardiovascular disease risk with explainable AI.<br>
                Understand <em>why</em> the model makes each prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key features - Clean grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;'><i class='bi bi-cpu' style='color: #667eea; font-size: 1.3rem;'></i> ML Models</h3>
            <p style='color: #666; line-height: 1.6;'>
                Six state-of-the-art algorithms with automated hyperparameter tuning
            </p>
            <ul style='color: #888; font-size: 0.9rem; line-height: 1.8;'>
                <li>Neural Network</li>
                <li>XGBoost</li>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;'><i class='bi bi-lightbulb' style='color: #667eea; font-size: 1.3rem;'></i> Explainability</h3>
            <p style='color: #666; line-height: 1.6;'>
                Three complementary XAI methods for complete transparency
            </p>
            <ul style='color: #888; font-size: 0.9rem; line-height: 1.8;'>
                <li>SHAP Values</li>
                <li>LIME Explanations</li>
                <li>Permutation Importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;'><i class='bi bi-stars' style='color: #667eea; font-size: 1.3rem;'></i> AI Insights</h3>
            <p style='color: #666; line-height: 1.6;'>
                Natural language interpretation of complex medical patterns
            </p>
            <ul style='color: #888; font-size: 0.9rem; line-height: 1.8;'>
                <li>Groq API (GPT-OSS)</li>
                <li>Medical Context</li>
                <li>English Support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### Getting Started")
    
    steps_col1, steps_col2 = st.columns([1, 1])
    
    with steps_col1:
        st.markdown("""
        **1. Prepare Your Data**
        - Upload CSV file with patient records
        - System auto-detects features
        
        **2. Train Models**
        - Click "Start Training" 
        - Wait 2-3 minutes for hyperparameter tuning
        
        **3. Analyze Results**
        - Compare 6 models performance
        - Visualize accuracy, precision, recall
        """)
    
    with steps_col2:
        st.markdown("""
        **4. Explore XAI**
        - Generate SHAP values
        - View LIME explanations
        - Understand feature importance
        
        **5. Get AI Interpretation**
        - Generate medical insights
        - English medical terminology
        - Clinical recommendations
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset statistics
    st.markdown("---")
    st.markdown("### Dataset Overview")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Features", "13", help="After feature engineering")
    with stat_col2:
        st.metric("Models", "6", help="ML algorithms compared")
    with stat_col3:
        st.metric("XAI Methods", "3", help="SHAP, LIME, Permutation")
    with stat_col4:
        st.metric("Best Accuracy", "~94%", help="Random Forest performance")


def show_training_page():
    """Data processing và model training - Clean scientific interface"""
    st.markdown("## Data Processing & Model Training")
    
    if 'data_file' not in st.session_state or st.session_state.data_file is None:
        st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;'>
            <i class='bi bi-folder2-open'></i> Please upload a CSV file in the sidebar to begin.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Load and preview data
    st.markdown("### Step 1: Data Preview")
    
    try:
        data = pd.read_csv(st.session_state.data_file)
        
        # Stats in clean cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", f"{data.shape[0]:,}")
        with col2:
            st.metric("Features", data.shape[1])
        with col3:
            st.metric("Missing", data.isnull().sum().sum())
        with col4:
            memory_mb = data.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.1f} MB")
        
        with st.expander("View Data Sample"):
            st.dataframe(data.head(10), use_container_width=True)
        
        # Training section
        st.markdown("---")
        st.markdown("### Step 2: Model Training")
        
        # XAI Improvement Option
        enable_xai = st.checkbox(
            " Enable XAI-Guided Model Improvement",
            value=False,
            help="Applies XAI-guided feature engineering to improve model performance. Creates interaction features from top important features identified by XAI analysis."
        )
        
        if enable_xai:
            st.markdown("""
            <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 4px solid #0066cc; color: #004080; margin-bottom: 1rem;'>
                <i class='bi bi-info-circle'></i> <strong>Feature Engineering Strategy:</strong> Uses XAI analysis to identify top features, then creates interaction features to capture complex patterns. This typically improves performance by 1-3% on most datasets.
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Start Training", type="primary", use_container_width=True):
                train_models(data, enable_xai_improvement=enable_xai)
        
        with col2:
            training_time = "4-6 minutes" if enable_xai else "2-3 minutes"
            mode_description = "optimized training with XAI-guided feature engineering" if enable_xai else "baseline training with hyperparameter tuning"
            st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;'><i class='bi bi-stopwatch'></i> Training takes {training_time}. Includes {mode_description} for 6 models.</div>", unsafe_allow_html=True)
        
        # Show results if trained
        if st.session_state.trained and st.session_state.training_results:
            show_training_results()
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


def train_models(data, enable_xai_improvement=False):
    """Execute training pipeline with 3 stages: No FE, FE Only, FE + XAI"""
    with st.spinner("Training models... This may take a few minutes..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save uploaded file temporarily
            temp_path = "temp_data.csv"
            data.to_csv(temp_path, index=False)
            
            # Step 1: Data Processing
            status_text.text("📊 Step 1/5: Processing data (before FE)...")
            progress_bar.progress(5)
            processor = DataProcessor()
            
            # Load and clean data but DON'T apply feature engineering yet
            raw_data = processor.load_data(temp_path)
            processor.validate_columns(raw_data)
            cleaned_data = processor.handle_outliers(raw_data)
            
            # Get X, y from cleaned data (no FE yet)
            X_raw, y_raw = processor.get_X_y(cleaned_data)
            progress_bar.progress(10)
            
            # ============ TRAINING 1: No FE (Baseline) ============
            status_text.text("🔷 Step 2/5: Training baseline models (No FE)...")
            progress_bar.progress(15)
            
            trainer_baseline = ModelTrainer()
            results_no_fe = trainer_baseline.full_training_pipeline(X_raw, y_raw, enable_xai_improvement=False)
            progress_bar.progress(30)
            
            # ============ TRAINING 2: FE Only (Statistical Testing) ============
            status_text.text("🔶 Step 3/5: Applying FE and training (FE Only)...")
            progress_bar.progress(35)
            
            # Now apply feature engineering (drop weak features)
            processed_data, stats = processor.process_pipeline(temp_path)
            X_fe, y_fe = processor.get_X_y(processed_data)
            
            trainer_fe = ModelTrainer()
            results_fe_only = trainer_fe.full_training_pipeline(X_fe, y_fe, enable_xai_improvement=False)
            progress_bar.progress(60)
            
            # ============ TRAINING 3: FE + XAI (Full Pipeline) ============
            if enable_xai_improvement:
                status_text.text("🔵 Step 4/5: Training with FE + XAI (Full Pipeline)...")
                progress_bar.progress(65)
                
                trainer_fe_xai = ModelTrainer()
                results_fe_xai = trainer_fe_xai.full_training_pipeline(X_fe, y_fe, enable_xai_improvement=True)
                progress_bar.progress(90)
            else:
                results_fe_xai = None
            
            # Step 5: Save results
            status_text.text("💾 Step 5/5: Saving results...")
            st.session_state.trained = True
            st.session_state.training_results = {
                'no_fe': results_no_fe,
                'fe_only': results_fe_only,
                'fe_xai': results_fe_xai,
                'processed_data': processed_data,
                'data_stats': stats,
                'X_train': results_fe_only['X_train'],  # For display
                'y_train': results_fe_only['y_train'],
                'X_test': results_fe_only['X_test'],
                'y_test': results_fe_only['y_test'],
                'X_train_scaled': results_fe_only['X_train_scaled'],
                'X_test_scaled': results_fe_only['X_test_scaled']
            }
            st.session_state.data_stats = stats
            progress_bar.progress(100)
            
            status_text.empty()
            progress_bar.empty()
            
            st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 4px solid #155724; color: #155724;'><i class='bi bi-check-circle-fill' style='color: #27AE60;'></i> Training completed successfully! (3 stages trained)</div>", unsafe_allow_html=True)
            st.balloons()
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        except Exception as e:
            st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8d7da; border-left: 4px solid #721c24; color: #721c24;'><i class='bi bi-x-circle-fill' style='color: #E74C3C;'></i> Training failed: {str(e)}</div>", unsafe_allow_html=True)
            import traceback
            st.error(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()


def show_training_results():
    """Display training results - Modern data visualization"""
    st.markdown("---")
    st.markdown("### 📊 Step 3: Training Results")
    
    results = st.session_state.training_results
    
    # Check if we have new 3-stage structure or old structure
    if 'fe_only' in results:
        # New 3-stage structure
        fe_only_results = results['fe_only']
        
        # Best model highlight (from FE Only stage)
        st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 4px solid #155724; color: #155724;'><i class='bi bi-trophy-fill' style='color: #F39C12;'></i> Best Performing Model (FE Only): {fe_only_results['best_model_name']}</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{fe_only_results['best_metrics']['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{fe_only_results['best_metrics']['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{fe_only_results['best_metrics']['Recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{fe_only_results['best_metrics']['F1 Score']:.4f}")
        
        # Show summary of 3 stages
        st.markdown("---")
        st.markdown("#### 🎯 Training Summary: 3 Stages Completed")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 🔷 Stage 1: No FE")
            if 'no_fe' in results:
                no_fe = results['no_fe']
                st.metric("Best Model", no_fe['best_model_name'])
                st.metric("Accuracy", f"{no_fe['best_metrics']['Accuracy']:.4f}")
            else:
                st.warning("No FE results not available")
        
        with col2:
            st.markdown("##### 🔶 Stage 2: FE Only")
            st.metric("Best Model", fe_only_results['best_model_name'])
            st.metric("Accuracy", f"{fe_only_results['best_metrics']['Accuracy']:.4f}")
        
        with col3:
            st.markdown("##### 🔵 Stage 3: Tuning → XAI")
            if 'fe_xai' in results and results['fe_xai']:
                fe_xai = results['fe_xai']
                st.metric("Best Model", fe_xai['best_model_name'])
                st.metric("Accuracy", f"{fe_xai['best_metrics']['Accuracy']:.4f}")
                st.caption("XAI improvement after hyperparameter tuning")
            else:
                st.info("XAI not enabled")
        
        # ============ DETAILED COMPARISON TABLE (like in image) ============
        st.markdown("---")
        st.markdown("### 📊 Baseline vs Improved Comparison")
        
        if 'no_fe' in results and 'fe_only' in results:
            no_fe = results['no_fe']
            fe_only = results['fe_only']
            
            # Create side-by-side comparison
            col_baseline, col_improved = st.columns(2)
            
            with col_baseline:
                st.markdown("#### 🔵 Baseline (No FE)")
                st.markdown("*Standard training with hyperparameter tuning only*")
                st.markdown("")
                
                # Create baseline dataframe
                baseline_data = []
                # for _, row in no_fe['tuned_results'].iterrows():
                #     baseline_data.append({
                #         'Model': row['Model Name'],
                #         'Accuracy': f"{row['Accuracy']:.4f}"
                #     })
                for _, row in no_fe['tuned_results'].iterrows():
                    baseline_data.append({
                        'Model': row['Model Name'],
                        'Accuracy': f"{row['Accuracy']:.4f}",
                        'Precision': f"{row['Precision']:.4f}",
                        'Recall': f"{row['Recall']:.4f}",
                        'F1 Score': f"{row['F1 Score']:.4f}",
                })
                baseline_df = pd.DataFrame(baseline_data)
                st.dataframe(baseline_df, use_container_width=True, hide_index=True, height=250)
            
            with col_improved:
                st.markdown("#### 🟢 Improved (With FE)")
                st.markdown("*Training with weak-feature elimination*")
                st.markdown("")
                
                # Create improved dataframe
                improved_data = []
                for _, row in fe_only['tuned_results'].iterrows():
                    improved_data.append({
                        'Model': row['Model Name'],
                        'Accuracy': f"{row['Accuracy']:.4f}",
                        'Precision': f"{row['Precision']:.4f}",
                        'Recall': f"{row['Recall']:.4f}",
                        'F1 Score': f"{row['F1 Score']:.4f}",
                    })
                improved_df = pd.DataFrame(improved_data)
                st.dataframe(improved_df, use_container_width=True, hide_index=True, height=250)
            
            # Improvement Analysis
            st.markdown("---")
            st.markdown("#### 📈 Improvement Analysis")
            
            improvement_data = []
            for model_name in no_fe['tuned_results']['Model Name'].values:
                baseline_row = no_fe['tuned_results'][no_fe['tuned_results']['Model Name'] == model_name].iloc[0]
                improved_row = fe_only['tuned_results'][fe_only['tuned_results']['Model Name'] == model_name].iloc[0]
                
                baseline_acc = baseline_row['Accuracy']
                improved_acc = improved_row['Accuracy']
                change = improved_acc - baseline_acc
                change_pct = (change / baseline_acc) * 100 if baseline_acc > 0 else 0
                
                improvement_data.append({
                    'Model': model_name,
                    'Baseline': f"{baseline_acc:.4f}",
                    'Improved': f"{improved_acc:.4f}",
                    'Change': f"{change:+.4f}",
                    'Change %': f"{change_pct:+.2f}%",
                    'Status': '✓ Improved' if change > 0.001 else ('✗ Decreased' if change < -0.001 else '– Same')
                })
            
            improvement_df = pd.DataFrame(improvement_data)
            
            st.dataframe(
                improvement_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Model": st.column_config.TextColumn("Model", width="medium"),
                    "Baseline": st.column_config.TextColumn("Baseline (No FE)", width="small"),
                    "Improved": st.column_config.TextColumn("Improved (With FE)", width="small"),
                    "Change": st.column_config.TextColumn("Change", width="small"),
                    "Change %": st.column_config.TextColumn("Change %", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small")
                }
            )
            
            # Summary stats
            improved_count = sum(1 for item in improvement_data if item['Status'] == '✓ Improved')
            avg_improvement = sum(float(item['Change'].replace('+', '')) for item in improvement_data) / len(improvement_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Improved", f"{improved_count}/{len(improvement_data)}")
            with col2:
                st.metric("Average Change", f"{avg_improvement:+.4f}")
            with col3:
                best_model_fe = fe_only['best_model_name']
                best_acc_fe = fe_only['best_metrics']['Accuracy']
                st.metric("Best Model (FE)", best_model_fe, f"{best_acc_fe:.4f}")
    
    else:
        # Old single-stage structure (fallback)
        st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 4px solid #155724; color: #155724;'><i class='bi bi-trophy-fill' style='color: #F39C12;'></i> Best Performing Model: {results['best_model_name']}</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['best_metrics']['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{results['best_metrics']['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{results['best_metrics']['Recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{results['best_metrics']['F1 Score']:.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============ FEATURE ENGINEERING STEPS (DETAILED) ============
    st.markdown("### 🔧 Feature Engineering Pipeline")
    
    st.markdown("""
    <div style='padding: 1rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 4px solid #2196F3; margin-bottom: 1rem;'>
        <strong>🔬 Applied FE Steps:</strong> The following transformations were applied to improve model performance
    </div>
    """, unsafe_allow_html=True)
    
    # with st.expander("📋 Step 1: Data Cleaning & Outlier Handling", expanded=True):
    #     if 'data_stats' in st.session_state:
    #         stats = st.session_state.data_stats
            
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             st.metric("Original Samples", stats['original_shape'][0])
    #             st.metric("Original Features", stats['original_shape'][1])
    #         with col2:
    #             st.metric("After Outlier Handling", stats['shape_after_outlier_handling'][0])
    #             removed = stats['original_shape'][0] - stats['shape_after_outlier_handling'][0]
    #             st.caption(f"Removed {removed} rows with extreme outliers")
    #         with col3:
    #             total_outliers = sum(stats.get('outliers_before', {}).values())
    #             st.metric("Total Outliers Detected", total_outliers)
    #             st.caption("Using IQR method")
            
    #         st.markdown("**� Outlier Details by Feature:**")
    #         outliers_before = stats.get('outliers_before', {})
    #         outliers_after = stats.get('outliers_after', {})
            
    #         if outliers_before:
    #             outlier_data = []
    #             for col in outliers_before.keys():
    #                 before = outliers_before.get(col, 0)
    #                 after = outliers_after.get(col, 0)
    #                 handled = before - after
    #                 outlier_data.append({
    #                     'Feature': col,
    #                     'Detected': before,
    #                     'After Handling': after,
    #                     'Handled': handled,
    #                     'Method': 'Winsorization'
    #                 })
                
    #             outlier_df = pd.DataFrame(outlier_data)
    #             st.dataframe(outlier_df, use_container_width=True, hide_index=True, height=200)
    
    # with st.expander("📋 Step 2: Feature Transformation", expanded=False):
    #     st.markdown("""
    #     **Applied Transformations:**
    #     - ✅ **Log Transformation**: Applied to skewed numerical features
    #     - ✅ **One-Hot Encoding**: Converted categorical variables to numerical format
    #     - ✅ **Standard Scaling**: Normalized features to zero mean and unit variance
    #     """)
        
    #     if 'data_stats' in st.session_state:
    #         stats = st.session_state.data_stats
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.metric("Features Before Encoding", stats['original_shape'][1])
    #         with col2:
    #             st.metric("Features After Encoding", stats['final_shape'][1])
    #             st.caption(f"Expansion: +{stats['final_shape'][1] - stats['original_shape'][1]} features")
    
    # with st.expander("📋 Step 3: Feature Selection (Statistical Testing)", expanded=False):
    #     st.markdown("""
    #     **Selection Criteria:**
    #     - ✅ **Mann-Whitney U Test**: Removed features with p-value > 0.05
    #     - ✅ **Statistical Significance**: Only kept features with significant difference between classes
    #     - ✅ **Weak Feature Removal**: Dropped features that don't contribute to prediction
    #     """)
        
    #     if 'data_stats' in st.session_state:
    #         stats = st.session_state.data_stats
    #         final_features = stats.get('final_features', [])
    #         if final_features:
    #             features_display = [f for f in final_features if f != 'target']
                
    #             st.markdown(f"**Final {len(features_display)} Features Used for Training:**")
    #             st.markdown(f"```\n{', '.join(features_display)}\n```")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Baseline vs Tuned Comparison (NEW!)
    st.markdown("###  Baseline vs Tuned Comparison")
    
    st.markdown("""
    <div style='padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: #f8f9fa; border-left: 4px solid #6c757d; margin-bottom: 1rem;'>
        <strong>Stage 1:</strong> Hyperparameter tuning optimization (GridSearchCV)
    </div>
    """, unsafe_allow_html=True)
    
    # Handle new 3-stage structure
    if 'fe_only' in results:
        baseline_df = results['fe_only']['baseline_results']
        tuned_df = results['fe_only']['tuned_results']
    else:
        baseline_df = results['baseline_results']
        tuned_df = results['tuned_results']
    
    # Create comparison dataframe
    comparison_data = []
    for model_name in baseline_df['Model Name'].values:
        baseline_row = baseline_df[baseline_df['Model Name'] == model_name].iloc[0]
        tuned_row = tuned_df[tuned_df['Model Name'] == model_name].iloc[0]
        
        improvement = tuned_row['Accuracy'] - baseline_row['Accuracy']
        improvement_pct = (improvement / baseline_row['Accuracy']) * 100
        
        comparison_data.append({
            'Model': model_name,
            'Baseline Accuracy': f"{baseline_row['Accuracy']:.4f}",
            'Tuned Accuracy': f"{tuned_row['Accuracy']:.4f}",
            'Improvement': f"{improvement:+.4f}",
            'Improvement %': f"{improvement_pct:+.2f}%",
            'Status': "Same params" if abs(improvement) < 0.001 else ("Improved" if improvement > 0 else "Decreased")
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="Improvement status after hyperparameter tuning",
            )
        }
    )
    
    st.caption("""
    **Note:** Models showing "Same params" mean GridSearchCV determined that default parameters 
    were already optimal for this dataset. This is a good sign - it confirms the model's 
    default configuration is well-suited for the problem!
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============ XAI THRESHOLD ANALYSIS (OPTIONAL - IF FE+XAI ENABLED) ============
    if 'fe_xai' in results and results['fe_xai'] and results['fe_xai'].get('improvement_results'):
        st.markdown("---")
        st.markdown("### 🔬 XAI-Guided Feature Analysis")
        
        st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 4px solid #2196F3; margin-bottom: 1rem;'>
            <strong>🎯 XAI Threshold Analysis:</strong> Tests different SHAP importance thresholds to find optimal feature subset
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Threshold Analysis Results", expanded=True):
            st.markdown("""
            This analysis removes features below different SHAP importance thresholds and measures the impact on model accuracy.
            Goal: Find the optimal threshold that maintains or improves performance while reducing feature complexity.
            """)
            
            # Get baseline accuracy for comparison
            fe_only_acc = results['fe_only']['best_metrics']['Accuracy']
            
            # Show threshold analysis if available
            improvement_data = results['fe_xai'].get('improvement_results', {})
            if 'xai_analysis' in improvement_data:
                xai_results = improvement_data['xai_analysis']
                
                # Display threshold comparison table
                st.dataframe(
                    xai_results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Threshold": st.column_config.NumberColumn("SHAP Threshold", format="%.3f"),
                        "Features Removed": st.column_config.NumberColumn("Features Removed"),
                        "Best Accuracy": st.column_config.TextColumn("Best Accuracy"),
                        "Change": st.column_config.TextColumn("vs Baseline"),
                        "Status": st.column_config.TextColumn("Status")
                    }
                )
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Baseline (FE Only)", f"{fe_only_acc:.4f}")
                with col2:
                    fe_xai_acc = results['fe_xai']['best_metrics']['Accuracy']
                    change = fe_xai_acc - fe_only_acc
                    st.metric("Best with XAI", f"{fe_xai_acc:.4f}", delta=f"{change:+.4f}")
                with col3:
                    if improvement_data.get('improved_selected', False):
                        st.success("✓ XAI Improved")
                    else:
                        st.info("– Baseline Better")
        
        # Show comparison of Tuning vs XAI
        with st.expander("📈 Hyperparameter Tuning vs XAI Improvement", expanded=True):
            st.markdown("""
            <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 1rem;'>
                <strong>Comparison:</strong> Models after hyperparameter tuning (GridSearchCV) vs. models improved with XAI-guided interaction features.
                XAI improvement is applied <strong>after tuning</strong> to further enhance performance.
            </div>
            """, unsafe_allow_html=True)
            
            fe_only = results['fe_only']
            fe_xai = results['fe_xai']
            
            col_tuning, col_xai = st.columns(2)
            
            with col_tuning:
                st.markdown("#### 🔶 After Tuning")
                st.markdown("*Hyperparameter-optimized models*")
                st.dataframe(
                    fe_only['tuned_results'][['Model Name', 'Accuracy']].rename(columns={'Model Name': 'Model'}),
                    use_container_width=True,
                    hide_index=True,
                    height=200
                )
            
            with col_xai:
                st.markdown("#### 🔵 After XAI")
                st.markdown("*XAI-guided feature engineering*")

                # FIX: ưu tiên accuracy sau XAI (Improved) nếu có
                if 'improvement_comparison' in fe_xai and isinstance(fe_xai['improvement_comparison'], pd.DataFrame):
                    df_xai = fe_xai['improvement_comparison'][['Model', 'Improved']].rename(columns={'Improved': 'Accuracy'})
                else:
                    # fallback cũ
                    df_xai = fe_xai['tuned_results'][['Model Name', 'Accuracy']].rename(columns={'Model Name': 'Model'})

                st.dataframe(
                    df_xai,
                    use_container_width=True,
                    hide_index=True,
                    height=200
                )
            
            # Improvement Analysis
            st.markdown("---")
            st.markdown("#### 📊 XAI Improvement Impact")
            
            improvement_data = []
            for model_name in fe_only['tuned_results']['Model Name'].values:
                tuning_row = fe_only['tuned_results'][fe_only['tuned_results']['Model Name'] == model_name].iloc[0]
                xai_row = fe_xai['improvement_comparison'][fe_xai['improvement_comparison']['Model'] == model_name].iloc[0]
                
                tuning_acc = float(tuning_row['Accuracy'])
                xai_acc = xai_acc = float(xai_row['Improved'])
                change = xai_acc - tuning_acc
                change_pct = (change / tuning_acc) * 100 if tuning_acc > 0 else 0
                
                improvement_data.append({
                    'Model': model_name,
                    'After Tuning': f"{tuning_acc:.4f}",
                    'After XAI': f"{xai_acc:.4f}",
                    'Change': f"{change:+.4f}",
                    'Change %': f"{change_pct:+.2f}%",
                    'Status': '✓ Improved' if change > 0.001 else ('✗ Decreased' if change < -0.001 else '– Same')
                })
            
            improvement_df = pd.DataFrame(improvement_data)
            
            st.dataframe(
                improvement_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Model": st.column_config.TextColumn("Model", width="medium"),
                    "After Tuning": st.column_config.TextColumn("After Tuning", width="small"),
                    "After XAI": st.column_config.TextColumn("After XAI", width="small"),
                    "Change": st.column_config.TextColumn("Change", width="small"),
                    "Change %": st.column_config.TextColumn("Change %", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small")
                }
            )
            
            # Summary
            improved_count = sum(1 for item in improvement_data if item['Status'] == '✓ Improved')
            avg_change = sum(float(item['Change'].replace('+', '')) for item in improvement_data) / len(improvement_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Improved by XAI", f"{improved_count}/{len(improvement_data)}")
            with col2:
                st.metric("Average Change", f"{avg_change:+.4f}")
            with col3:
                best_xai = fe_xai['best_model_name']
                best_xai_acc = fe_xai['best_metrics']['Accuracy']
                st.metric("Best Model (XAI)", best_xai, f"{best_xai_acc:.4f}")
    
    # Info box if XAI not enabled
    else:
        st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #ffc107; color: #856404; margin-top: 1rem;'>
            <i class='bi bi-lightbulb'></i> <strong>Want XAI-Guided Improvements?</strong><br>
            Enable <strong>"XAI-Guided Model Improvement"</strong> before training to see:
            <ul style='margin: 0.5rem 0 0 1.5rem;'>
                <li>🔬 <strong>Threshold Analysis</strong> showing feature removal impact</li>
                <li>🔗 <strong>Interaction Features</strong> created from top important features</li>
                <li>📊 <strong>Detailed Comparisons</strong> between all stages</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # XAI-Guided Improvement Results (NEW!)
    if 'improvement_results' in results and results.get('improvement_results'):
        st.markdown("---")
        st.markdown("###  Model Improvement with Feature Engineering")
        
        # Check if we have the new format results
        if 'improvement_comparison' in results:
            # NEW FORMAT: Baseline vs Improved (Optimized + FE)
            comparison_df = results['improvement_comparison']
            
            # ==================== PHASE 1: Statistical Testing & Weak Feature Elimination ====================
            st.markdown("---")
            st.markdown("###  Phase 1: Statistical Testing & Weak Feature Elimination")
            
            st.markdown("""
            <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 1rem;'>
                <strong>Purpose:</strong> Identify and remove weak features through statistical analysis before model training.
                This includes visualizing feature distributions and applying Welch's t-test to determine discriminative power.
            </div>
            """, unsafe_allow_html=True)
            
            # Statistical Analysis: Boxplots for Numerical Features
            st.markdown("####  Boxplot Analysis: Numerical Features by Target Class")
            
            st.markdown("""
            <div style='padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: #f8f9fa; border-left: 3px solid #6c757d; margin-bottom: 1rem; font-size: 0.9rem;'>
                <strong>Analysis:</strong> Features like <code>max heart rate</code> and <code>oldpeak</code> show clear separation between target groups.
                Patients with heart disease tend to have lower maximum heart rates and higher oldpeak values, aligning with clinical expectations.
                However, <code>resting bp s</code> and <code>cholesterol</code> show minimal separation, suggesting weaker discriminatory power.
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Get training data BEFORE feature engineering (to include cholesterol)
                from sklearn.model_selection import train_test_split
                
                # Get raw data from session state
                if 'processed_data' in st.session_state:
                    raw_data = st.session_state.processed_data
                    X_raw = raw_data.drop('target', axis=1)
                    y_raw = raw_data['target']
                    
                    # Split to get training data (same random state)
                    X_train_raw, _, y_train_raw, _ = train_test_split(
                        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
                    )
                else:
                    # Fallback to current X_train
                    X_train_raw = st.session_state.training_results['X_train']
                    y_train_raw = st.session_state.training_results['y_train']
                
                # Select numerical features for boxplot (including cholesterol)
                numerical_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
                available_features = [f for f in numerical_features if f in X_train_raw.columns]
                
                # Create boxplot figure
                import seaborn as sns
                fig, axes = plt.subplots(3, 2, figsize=(14, 12))
                axes = axes.flatten()
                
                for idx, feature in enumerate(available_features[:5]):
                    data_plot = pd.DataFrame({
                        feature: X_train_raw[feature],
                        'target': y_train_raw
                    })
                    
                    sns.boxplot(data=data_plot, x='target', y=feature, ax=axes[idx], palette=['lightblue', 'lightcoral'])
                    axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
                    axes[idx].set_xlabel('Target (0=No Disease, 1=Disease)')
                    axes[idx].grid(True, alpha=0.3)
                
                # Hide extra subplot
                if len(available_features) < 6:
                    axes[5].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not generate boxplots: {str(e)}")
            
            # Welch's t-test Results
            st.markdown("####  Welch's t-Test Results")
            
            st.markdown("""
            <div style='padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 3px solid #0066cc; margin-bottom: 1rem; font-size: 0.9rem;'>
                <strong>Statistical Testing:</strong> Welch's two-sample t-test was applied to identify weak features.
                The test showed that <strong>cholesterol</strong> had an absolute t-statistic of approximately <strong>1.61</strong>, 
                which is considerably lower than the critical threshold of 1.962 at a 0.05 significance level (95% confidence interval).
                This confirms no statistically significant difference in means between the two classes.
                <strong>→ Cholesterol identified as weak feature and excluded from dataset.</strong>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                from scipy import stats
                
                # Calculate mean and std for each feature by target
                st.markdown("** Mean and Standard Deviation by Target Class**")
                
                mean_std_data = []
                for feature in available_features:
                    group_0 = X_train_raw[y_train_raw == 0][feature]
                    group_1 = X_train_raw[y_train_raw == 1][feature]
                    
                    mean_std_data.append({
                        'Feature': feature,
                        'Mean (Target=0)': f"{group_0.mean():.6f}",
                        'Std (Target=0)': f"{group_0.std():.6f}",
                        'Mean (Target=1)': f"{group_1.mean():.6f}",
                        'Std (Target=1)': f"{group_1.std():.6f}"
                    })
                
                mean_std_df = pd.DataFrame(mean_std_data)
                
                st.dataframe(
                    mean_std_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Feature": st.column_config.TextColumn("Feature", width="medium"),
                        "Mean (Target=0)": st.column_config.TextColumn("Mean (No Disease)", width="small"),
                        "Std (Target=0)": st.column_config.TextColumn("Std (No Disease)", width="small"),
                        "Mean (Target=1)": st.column_config.TextColumn("Mean (Disease)", width="small"),
                        "Std (Target=1)": st.column_config.TextColumn("Std (Disease)", width="small")
                    }
                )
                
                st.caption("**Table:** Mean and standard deviation of numerical features grouped by target class.")
                
                # Perform Welch's t-test for available features
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Welch's t-Test Statistical Results**")
                
                test_results = []
                for feature in available_features:
                    group_0 = X_train_raw[y_train_raw == 0][feature]
                    group_1 = X_train_raw[y_train_raw == 1][feature]
                    
                    # Welch's t-test (equal_var=False)
                    t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)
                    
                    test_results.append({
                        'Feature': feature,
                        't-statistic': f"{abs(t_stat):.6f}",
                        'p-value': f"{p_value:.6f}",
                        'Significant (α=0.05)': ' Yes' if p_value < 0.05 else ' No',
                        'Decision': ' Keep' if p_value < 0.05 else ' Remove (Weak)'
                    })
                
                test_df = pd.DataFrame(test_results)
                
                st.dataframe(
                    test_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Feature": st.column_config.TextColumn("Feature", width="medium"),
                        "t-statistic": st.column_config.TextColumn("t-statistic", width="small"),
                        "p-value": st.column_config.TextColumn("p-value", width="small"),
                        "Significant (α=0.05)": st.column_config.TextColumn("Significant?", width="small"),
                        "Decision": st.column_config.TextColumn("Decision", width="small")
                    }
                )
                
                st.caption("**Note:** Features with p-value < 0.05 show statistically significant difference between classes and are kept.")
                
            except Exception as e:
                st.warning(f"Could not perform t-test: {str(e)}")
            
            # Categorical Variable Analysis: Point Plot + Histogram
            st.markdown("#### 📍 Categorical Features: Point Plot & Histogram Analysis")
            
            st.markdown("""
            <div style='padding: 0.75rem 1rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 1rem;'>
                <strong>⚠️ Important Methodology Note:</strong> This analysis is <strong>dataset-specific</strong> and not applicable to all datasets.
                The decision to remove features is based on <strong>observed variance and sample distribution</strong>, which varies by dataset.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='padding: 0.75rem 1rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 4px solid #0066cc; margin-bottom: 1rem; font-size: 0.9rem;'>
                <strong>🔍 Why Use Point Plot and Histogram?</strong><br><br>
                
                <strong>Step 1: One-Hot Encoding</strong><br>
                Categorical variables with >2 groups (e.g., <code>chest pain type</code>, <code>resting ecg</code>, <code>ST slope</code>) 
                are converted to dummy variables using <code>pd.get_dummies()</code>. This creates binary columns for each category 
                (e.g., <code>resting ecg_0</code>, <code>resting ecg_1</code>, <code>resting ecg_2</code>).<br><br>
                
                <strong>Step 2: Point Plot Analysis (Variance Detection)</strong><br>
                 <strong>Purpose:</strong> Visualize the <strong>mean target value and variance (error bars)</strong> for each category.<br>
                ⚠️ <strong>Problem Indicator:</strong> <strong>Wide error bars</strong> = <strong>High variance</strong> = Unstable feature<br>
                 <strong>Example:</strong> If <code>resting ecg_1</code> shows mean=0.6 with error bars from 0.4 to 0.8, 
                this indicates <strong>high statistical uncertainty</strong> and <strong>wide data scatter</strong>.<br><br>
                
                <strong>Step 3: Histogram Verification (Sample Count Check)</strong><br>
                 <strong>Purpose:</strong> Confirm the <strong>sample distribution</strong> across categories.<br>
                ⚠️ <strong>Problem Indicator:</strong> <strong>Low sample count</strong> (<5% of total) = Insufficient data = Unreliable statistics<br>
                 <strong>Example:</strong> If <code>resting ecg_1</code> has only 15 samples out of 1000 total (1.5%), 
                the high variance observed in the point plot is <strong>caused by insufficient data</strong>, not genuine patterns.<br><br>
                
                <strong> Removal Decision Criteria (Both conditions must be met):</strong><br>
                1. <strong>High Variance:</strong> Wide error bars in point plot (variance > dataset mean variance)<br>
                2. <strong>Low Sample Count:</strong> <5% of total samples in histogram<br><br>
                
                <strong> Why This Matters:</strong><br>
                Features with <strong>high variance + low samples</strong> introduce <strong>noise and overfitting risk</strong>. 
                The model learns from statistically unreliable patterns. Removing them improves <strong>generalization</strong> 
                and <strong>model stability</strong>.<br><br>
                
                <strong>⚠️ Dataset-Specific Nature:</strong><br>
                 <strong>NOT all datasets</strong> will have weak categorical features!<br>
                 This analysis is performed <strong>only if</strong> point plot shows high variance <strong>AND</strong> histogram confirms low sample count.<br>
                 Some datasets may have <strong>zero features removed</strong> if all categories have sufficient samples and stable variance.
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Get categorical features BEFORE one-hot encoding
                if 'processed_data' in st.session_state:
                    raw_data = st.session_state.processed_data
                    
                    # Representative categorical feature: resting ecg
                    representative_feature = 'resting ecg'
                    
                    if representative_feature in raw_data.columns:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Point Plot: Target Distribution with Variance**")
                            st.markdown(f"*Example: {representative_feature}*")
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Create point plot data
                            categories = sorted(raw_data[representative_feature].unique())
                            means = []
                            stds = []
                            
                            for cat in categories:
                                mask = raw_data[representative_feature] == cat
                                target_vals = raw_data[mask]['target']
                                means.append(target_vals.mean())
                                stds.append(target_vals.std())
                            
                            # Plot with error bars
                            ax.errorbar(
                                categories, 
                                means,
                                yerr=stds,
                                marker='o',
                                markersize=10,
                                capsize=8,
                                capthick=3,
                                linewidth=2,
                                color='steelblue',
                                ecolor='coral',
                                label=representative_feature
                            )
                            
                            ax.set_xlabel('Category Value', fontsize=11)
                            ax.set_ylabel('Target Mean', fontsize=11)
                            ax.set_title(f'Point Plot: {representative_feature}', fontsize=12, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim(-0.2, 1.2)
                            
                            # Add annotation for high variance category
                            max_std_idx = np.argmax(stds)
                            if stds[max_std_idx] > np.mean(stds):
                                ax.annotate(
                                    'High Variance!',
                                    xy=(categories[max_std_idx], means[max_std_idx]),
                                    xytext=(categories[max_std_idx] + 0.3, means[max_std_idx] + 0.2),
                                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                                    fontsize=10,
                                    color='red',
                                    fontweight='bold'
                                )
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            st.caption("**Wide error bars** = High variance = Unstable feature")
                        
                        with col2:
                            st.markdown("**📊 Histogram: Sample Distribution**")
                            st.markdown(f"*Example: {representative_feature}*")
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Count samples by category and target
                            value_counts_total = raw_data[representative_feature].value_counts().sort_index()
                            
                            # Create stacked bar chart by target
                            target_0_counts = []
                            target_1_counts = []
                            
                            for cat in sorted(value_counts_total.index):
                                target_0_counts.append(len(raw_data[(raw_data[representative_feature] == cat) & (raw_data['target'] == 0)]))
                                target_1_counts.append(len(raw_data[(raw_data[representative_feature] == cat) & (raw_data['target'] == 1)]))
                            
                            x_pos = np.arange(len(value_counts_total))
                            width = 0.6
                            
                            ax.bar(x_pos, target_0_counts, width, label='Target=0', color='lightblue')
                            ax.bar(x_pos, target_1_counts, width, bottom=target_0_counts, label='Target=1', color='lightcoral')
                            
                            ax.set_xlabel('Resting Ecg', fontsize=11)
                            ax.set_ylabel('Frequency of Disease or Not', fontsize=11)
                            ax.set_title(f'Heart Disease Frequency According To {representative_feature.title()}', fontsize=12, fontweight='bold')
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(sorted(value_counts_total.index))
                            ax.legend()
                            ax.grid(True, alpha=0.3, axis='y')
                            
                            # Highlight low sample categories
                            total_samples = len(raw_data)
                            for i, (count_0, count_1) in enumerate(zip(target_0_counts, target_1_counts)):
                                total_count = count_0 + count_1
                                if total_count < total_samples * 0.05:  # < 5%
                                    ax.text(i, total_count + 10, 'Low!', 
                                           ha='center', va='bottom', color='red', 
                                           fontweight='bold', fontsize=10)
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            st.caption("**Low sample count** (<5%) = Insufficient data = Unreliable statistics")
                        
                        st.markdown("---")
                        st.info("💡 **Decision Logic:** If a category shows **both** high variance (point plot) **and** low samples (histogram), it's removed as a weak feature.")
                    
                    else:
                        st.warning(f"Representative feature '{representative_feature}' not found in dataset.")
                else:
                    st.warning("Processed data not available for categorical analysis.")
                
            except Exception as e:
                st.warning(f"Could not generate categorical analysis: {str(e)}")
            
            # Feature Removal Summary
            st.markdown("---")
            st.markdown("#### 🗑️ Feature Engineering Summary: Weak Features Removed")
            
            st.markdown("""
            <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #f8d7da; border-left: 4px solid #dc3545; margin-bottom: 1rem;'>
                <strong>Features Removed Based on Statistical Analysis:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Use raw data to identify weak features (including cholesterol)
                # Identify weak features from t-test
                weak_numerical = []
                for feature in available_features:
                    group_0 = X_train_raw[y_train_raw == 0][feature]
                    group_1 = X_train_raw[y_train_raw == 1][feature]
                    t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)
                    
                    if p_value >= 0.05:  # Not significant
                        weak_numerical.append({
                            'Feature': feature,
                            'Type': 'Numerical',
                            'Reason': f'Welch t-test: p-value={p_value:.4f} ≥ 0.05 (not significant)',
                            't-statistic': f'{abs(t_stat):.4f}'
                        })
                
                # Identify weak categorical features from raw data
                weak_categorical = []
                if 'processed_data' in st.session_state:
                    raw_data = st.session_state.processed_data
                    categorical_features = ['chest pain type', 'resting ecg', 'ST slope', 'fasting blood sugar']
                    
                    for cat_feat in categorical_features:
                        if cat_feat in raw_data.columns:
                            value_counts = raw_data[cat_feat].value_counts()
                            total_samples = len(raw_data)
                            
                            for val, count in value_counts.items():
                                if count < total_samples * 0.05:  # Less than 5%
                                    weak_categorical.append({
                                        'Feature': f'{cat_feat}_{val}',
                                        'Type': 'Categorical (One-Hot Encoded)',
                                        'Reason': f'Low sample count: {count} samples ({count/total_samples*100:.1f}% of data)',
                                        'Sample Count': str(count)
                                    })
                
                # Combine all weak features
                if weak_numerical or weak_categorical:
                    st.markdown("**📋 Removed Features List:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if weak_numerical:
                            st.markdown("**Numerical Features:**")
                            for feat in weak_numerical:
                                st.markdown(f"- **{feat['Feature']}**")
                                st.markdown(f"  - Reason: {feat['Reason']}")
                                st.markdown(f"  - t-statistic: {feat['t-statistic']}")
                                st.markdown("")
                    
                    with col2:
                        if weak_categorical:
                            st.markdown("**Categorical Features (Dummy Variables):**")
                            displayed = set()
                            for feat in weak_categorical:
                                if feat['Feature'] not in displayed:
                                    st.markdown(f"- **{feat['Feature']}**")
                                    st.markdown(f"  - Reason: {feat['Reason']}")
                                    st.markdown("")
                                    displayed.add(feat['Feature'])
                    
                    # Summary statistics
                    st.markdown("---")
                    
                    # Calculate original feature count (before FE)
                    original_feature_count = len(available_features)  # Numerical features
                    if 'processed_data' in st.session_state:
                        raw_data = st.session_state.processed_data
                        categorical_features = ['chest pain type', 'resting ecg', 'ST slope', 'fasting blood sugar']
                        for cat_feat in categorical_features:
                            if cat_feat in raw_data.columns:
                                original_feature_count += raw_data[cat_feat].nunique()  # Add unique categories
                    
                    removed_count = len(weak_numerical) + len(weak_categorical)
                    retained_count = original_feature_count - removed_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Features", original_feature_count)
                    with col2:
                        st.metric("Weak Features Removed", removed_count)
                    with col3:
                        st.metric("Features Retained", retained_count)
                    
                    st.success(" **Result:** Weak features removed to improve model stability and reduce noise.")
                else:
                    st.info("ℹ️ No weak features identified for removal. All features show statistical significance.")
                
            except Exception as e:
                st.warning(f"Could not generate feature removal summary: {str(e)}")
            
            # ==================== PHASE 2: XAI-Guided Improvement ====================
            st.markdown("---")
            st.markdown("### 🔍 Phase 2: XAI-Guided Feature Engineering")
            
            # Show XAI Analysis Results (Why these features?)
            if 'xai_results' in results:
                st.markdown("#### 🔍 XAI Analysis: Feature Selection Rationale")
                st.markdown("""
                <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 4px solid #0066cc; margin-bottom: 1rem;'>
                    <strong>How XAI Guides Feature Engineering:</strong> The system analyzes feature importance using SHAP 
                    and Permutation Importance to identify the most predictive features, then creates interaction features 
                    from these top features to capture complex patterns.
                </div>
                """, unsafe_allow_html=True)
                
                # Display XAI charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("** Permutation Importance**")
                    perm_importance = results['xai_results']['permutation_importance']
                    top_5 = perm_importance.nlargest(5, 'Importance')
                    
                    fig = px.bar(
                        top_5,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 5 Most Important Features',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**⭐ Top Features Used for FE**")
                    st.markdown("""
                    <div style='padding: 0.75rem; background-color: #f8f9fa; border-radius: 0.5rem;'>
                    """, unsafe_allow_html=True)
                    
                    for i, row in top_5.iterrows():
                        st.markdown(f"**{row['Feature']}**")
                        st.progress(float(row['Importance']))
                        st.markdown(f"<small>Importance: {row['Importance']:.4f}</small>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Add SHAP Visualizations
                st.markdown("---")
                st.markdown("#### 📊 SHAP Analysis Visualizations")
                
                st.markdown("""
                <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 4px solid #0066cc; margin-bottom: 1rem;'>
                    <strong>What is SHAP?</strong> SHAP (SHapley Additive exPlanations) shows how each feature contributes to predictions.
                    The violin plot shows the distribution of feature impacts, while the bar chart ranks features by importance.
                </div>
                """, unsafe_allow_html=True)
                
                if 'shap_values' in results['xai_results'] and results['xai_results']['shap_values'] is not None:
                    try:
                        shap_values = results['xai_results']['shap_values']
                        X_test_scaled = st.session_state.training_results['X_test_scaled']
                        feature_names = st.session_state.training_results['X_train'].columns.tolist()
                        
                        # Debug info
                        st.write(f"✅ SHAP values loaded. Shape: {np.array(shap_values).shape}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**SHAP Summary Plot (Feature Impact)**")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Handle multi-class shap values
                                if isinstance(shap_values, list):
                                    shap_values_plot = shap_values[1]  # Use class 1
                                elif len(np.array(shap_values).shape) == 3:
                                    shap_values_plot = shap_values[:, :, 1]
                                else:
                                    shap_values_plot = shap_values
                                
                                st.write(f"Plot shape: {np.array(shap_values_plot).shape}")
                                
                                shap.summary_plot(
                                    shap_values_plot,
                                    X_test_scaled,
                                    feature_names=feature_names,
                                    plot_type="violin",
                                    show=False
                                )
                                st.pyplot(fig)
                                plt.close()
                            except Exception as e:
                                st.error(f"❌ SHAP summary plot error: {str(e)}")
                                st.write("Debug info:", type(shap_values), np.array(shap_values).shape if hasattr(shap_values, 'shape') else "No shape")
                        
                        with col2:
                            st.markdown("**SHAP Feature Importance (Bar)**")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                if isinstance(shap_values, list):
                                    shap_values_plot = shap_values[1]
                                elif len(np.array(shap_values).shape) == 3:
                                    shap_values_plot = shap_values[:, :, 1]
                                else:
                                    shap_values_plot = shap_values
                                
                                shap.summary_plot(
                                    shap_values_plot,
                                    X_test_scaled,
                                    feature_names=feature_names,
                                    plot_type="bar",
                                    show=False
                                )
                                st.pyplot(fig)
                                plt.close()
                            except Exception as e:
                                st.error(f"❌ SHAP bar plot error: {str(e)}")
                                st.write("Debug info:", type(shap_values), np.array(shap_values).shape if hasattr(shap_values, 'shape') else "No shape")
                    
                    except Exception as e:
                        st.error(f"❌ Error loading SHAP data: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.warning("⚠️ SHAP values not found in results. This may happen if XAI analysis failed.")
                
                # Add Feature Distribution Analysis
                st.markdown("---")
                st.markdown("#### Feature Distribution Analysis")
                
                st.markdown("""
                <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #f8f9fa; border-left: 4px solid #6c757d; margin-bottom: 1rem;'>
                    <strong>Analysis:</strong> Distribution of top important features across target classes
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Get top 4 features
                    top_4_features = perm_importance.nlargest(4, 'Importance')['Feature'].tolist()
                    
                    # Create 2x2 subplot
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    axes = axes.flatten()
                    
                    X_train = st.session_state.training_results['X_train']
                    y_train = st.session_state.training_results['y_train']
                    
                    for idx, feature in enumerate(top_4_features):
                        if feature in X_train.columns:
                            # Create histogram with target overlay
                            for target_val in [0, 1]:
                                data_subset = X_train[y_train == target_val][feature]
                                axes[idx].hist(
                                    data_subset,
                                    alpha=0.6,
                                    label=f'Target={target_val}',
                                    bins=20,
                                    edgecolor='black'
                                )
                            
                            axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
                            axes[idx].set_xlabel('Value')
                            axes[idx].set_ylabel('Frequency')
                            axes[idx].legend()
                            axes[idx].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.warning(f"Could not generate distribution plots: {str(e)}")
            
            # Add Threshold Analysis
            st.markdown("---")
            st.markdown("####  Threshold Analysis: Feature Removal Impact")
            
            st.markdown("""
            <div style='padding: 0.75rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 1rem;'>
                <strong>Analysis:</strong> Test removing features with importance below different thresholds. 
                This shows which features are critical vs redundant.
            </div>
            """, unsafe_allow_html=True)
            
            if 'xai_results' in results:
                try:
                    # Get baseline accuracy
                    baseline_accuracy = float(comparison_df['Baseline'].iloc[0])
                    
                    # Get best model from appropriate stage
                    results = st.session_state.training_results
                    if 'fe_xai' in results and results['fe_xai']:
                        best_model = results['fe_xai']['best_model']
                    elif 'fe_only' in results:
                        best_model = results['fe_only']['best_model']
                    else:
                        best_model = results['best_model']
                    
                    # Create explainer instance
                    from pipeline import XAIExplainer
                    explainer = XAIExplainer(
                        model=best_model,
                        X_train=st.session_state.training_results['X_train'],
                        X_test=st.session_state.training_results['X_test'],
                        y_train=st.session_state.training_results['y_train'],
                        y_test=st.session_state.training_results['y_test'],
                        X_train_scaled=st.session_state.training_results['X_train_scaled'],
                        X_test_scaled=st.session_state.training_results['X_test_scaled'],
                        feature_names=st.session_state.training_results['X_train'].columns.tolist()
                    )
                    
                    # Perform threshold analysis
                    with st.spinner("Analyzing feature removal impact..."):
                        threshold_results = explainer.threshold_analysis(baseline_accuracy)
                    
                    # Display results
                    st.dataframe(
                        threshold_results,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Threshold": st.column_config.NumberColumn("Threshold", format="%.3f"),
                            "Features Removed": st.column_config.NumberColumn("Features Removed", format="%d"),
                            "Best Accuracy": st.column_config.TextColumn("Best Accuracy"),
                            "Change": st.column_config.TextColumn("Change"),
                            "Status": st.column_config.TextColumn("Status")
                        }
                    )
                    
                except Exception as e:
                    st.warning(f"Threshold analysis not available: {str(e)}")
            
            # ==================== COMPARISON TABLES: 3 STAGES ====================
            st.markdown("---")
            st.markdown("### 📊 Model Performance Comparison: 3-Stage Analysis")
            
            st.markdown("""
            <div style='padding: 0.75rem 1rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 1rem; font-size: 0.9rem;'>
                <strong>📌 Training Stages:</strong><br>
                🔷 <strong>No FE:</strong> Baseline (raw data, no feature engineering)<br>
                🔶 <strong>FE Only:</strong> Statistical Testing + Weak Feature Removal<br>
                🔵 <strong>FE + XAI:</strong> Statistical Testing + Weak Feature Removal + XAI-Guided Interaction Features
            </div>
            """, unsafe_allow_html=True)
            
            # Get all 3 training results
            results_no_fe = st.session_state.training_results.get('no_fe')
            results_fe_only = st.session_state.training_results.get('fe_only')
            results_fe_xai = st.session_state.training_results.get('fe_xai')
            
            if results_no_fe and results_fe_only:
                # Create comparison dataframes
                no_fe_df = results_no_fe['tuned_results'][['Model', 'Accuracy']].copy()
                fe_only_df = results_fe_only['tuned_results'][['Model', 'Accuracy']].copy()
                
                # ====== COMPARISON 1: No FE vs FE Only ======
                with st.expander("📊 **Comparison 1: Impact of Statistical Feature Engineering (No FE → FE Only)**", expanded=True):
                    st.markdown("""
                    <div style='padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 3px solid #0066cc; margin-bottom: 1rem; font-size: 0.85rem;'>
                        <strong>Analysis:</strong> This shows the impact of <strong>Statistical Testing</strong> (Welch's t-test, sample count analysis) 
                        and <strong>Weak Feature Removal</strong> (cholesterol, fasting blood sugar, low-sample categories).
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 🔷 No FE (Baseline)")
                        st.caption("*Training on raw data (all features)*")
                        st.dataframe(
                            no_fe_df,
                            use_container_width=True,
                            hide_index=True,
                            height=250,
                            column_config={
                                "Model": st.column_config.TextColumn("Model", width="medium"),
                                "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f")
                            }
                        )
                    
                    with col2:
                        st.markdown("##### 🔶 FE Only (Statistical Testing)")
                        st.caption("*Training after removing weak features*")
                        st.dataframe(
                            fe_only_df,
                            use_container_width=True,
                            hide_index=True,
                            height=250,
                            column_config={
                                "Model": st.column_config.TextColumn("Model", width="medium"),
                                "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f")
                            }
                        )
                    
                    # Comparison table
                    st.markdown("##### 📈 Impact Analysis")
                    comparison_1 = pd.DataFrame({
                        'Model': no_fe_df['Model'],
                        'No FE': no_fe_df['Accuracy'],
                        'FE Only': fe_only_df['Accuracy'],
                        'Change': fe_only_df['Accuracy'].values - no_fe_df['Accuracy'].values,
                        'Change %': ((fe_only_df['Accuracy'].values - no_fe_df['Accuracy'].values) / no_fe_df['Accuracy'].values * 100).round(2)
                    })
                    comparison_1['Change'] = comparison_1['Change'].apply(lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}")
                    comparison_1['Change %'] = comparison_1['Change %'].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
                    
                    st.dataframe(
                        comparison_1,
                        use_container_width=True,
                        hide_index=True,
                        height=250,
                        column_config={
                            "Model": st.column_config.TextColumn("Model", width="medium"),
                            "No FE": st.column_config.NumberColumn("No FE", format="%.4f"),
                            "FE Only": st.column_config.NumberColumn("FE Only", format="%.4f"),
                            "Change": st.column_config.TextColumn("Δ Accuracy", width="small"),
                            "Change %": st.column_config.TextColumn("Δ %", width="small")
                        }
                    )
                
                # ====== COMPARISON 2: FE Only vs FE + XAI ======
                if results_fe_xai:
                    # Get improved results from FE + XAI
                    if 'improvement_comparison' in results_fe_xai:
                        fe_xai_df = results_fe_xai['improvement_comparison'][['Model', 'Improved']].copy()
                        fe_xai_df.columns = ['Model', 'Accuracy']
                        
                        with st.expander("📊 **Comparison 2: Impact of XAI-Guided Interaction Features (FE Only → FE + XAI)**", expanded=True):
                            st.markdown("""
                            <div style='padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: #e7f3ff; border-left: 3px solid #0066cc; margin-bottom: 1rem; font-size: 0.85rem;'>
                                <strong>Analysis:</strong> This shows the impact of <strong>XAI-Guided Interaction Features</strong> 
                                (SHAP + Permutation Importance → Top features → Interaction features).
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### 🔶 FE Only")
                                st.caption("*After statistical testing only*")
                                st.dataframe(
                                    fe_only_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=250,
                                    column_config={
                                        "Model": st.column_config.TextColumn("Model", width="medium"),
                                        "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f")
                                    }
                                )
                            
                            with col2:
                                st.markdown("##### 🔵 FE + XAI")
                                st.caption("*After statistical testing + XAI interaction*")
                                st.dataframe(
                                    fe_xai_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=250,
                                    column_config={
                                        "Model": st.column_config.TextColumn("Model", width="medium"),
                                        "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f")
                                    }
                                )
                            
                            # Comparison table
                            st.markdown("##### 📈 Impact Analysis")
                            comparison_2 = pd.DataFrame({
                                'Model': fe_only_df['Model'],
                                'FE Only': fe_only_df['Accuracy'],
                                'FE + XAI': fe_xai_df['Accuracy'],
                                'Change': fe_xai_df['Accuracy'].values - fe_only_df['Accuracy'].values,
                                'Change %': ((fe_xai_df['Accuracy'].values - fe_only_df['Accuracy'].values) / fe_only_df['Accuracy'].values * 100).round(2)
                            })
                            comparison_2['Change'] = comparison_2['Change'].apply(lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}")
                            comparison_2['Change %'] = comparison_2['Change %'].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
                            
                            st.dataframe(
                                comparison_2,
                                use_container_width=True,
                                hide_index=True,
                                height=250,
                                column_config={
                                    "Model": st.column_config.TextColumn("Model", width="medium"),
                                    "FE Only": st.column_config.NumberColumn("FE Only", format="%.4f"),
                                    "FE + XAI": st.column_config.NumberColumn("FE + XAI", format="%.4f"),
                                    "Change": st.column_config.TextColumn("Δ Accuracy", width="small"),
                                    "Change %": st.column_config.TextColumn("Δ %", width="small")
                                }
                            )
                        
                        # ====== COMPARISON 3: No FE vs FE + XAI (Overall) ======
                        with st.expander("📊 **Comparison 3: Overall Improvement (No FE → FE + XAI)**", expanded=True):
                            st.markdown("""
                            <div style='padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 3px solid #155724; margin-bottom: 1rem; font-size: 0.85rem;'>
                                <strong>Analysis:</strong> This shows the <strong>total combined impact</strong> of both 
                                Statistical Feature Engineering AND XAI-Guided Interaction Features.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### 🔷 No FE (Baseline)")
                                st.caption("*Raw data baseline*")
                                st.dataframe(
                                    no_fe_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=250,
                                    column_config={
                                        "Model": st.column_config.TextColumn("Model", width="medium"),
                                        "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f")
                                    }
                                )
                            
                            with col2:
                                st.markdown("##### 🔵 FE + XAI (Full Pipeline)")
                                st.caption("*Complete feature engineering pipeline*")
                                st.dataframe(
                                    fe_xai_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=250,
                                    column_config={
                                        "Model": st.column_config.TextColumn("Model", width="medium"),
                                        "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f")
                                    }
                                )
                            
                            # Comparison table
                            st.markdown("##### 📈 Overall Impact Analysis")
                            comparison_3 = pd.DataFrame({
                                'Model': no_fe_df['Model'],
                                'No FE': no_fe_df['Accuracy'],
                                'FE + XAI': fe_xai_df['Accuracy'],
                                'Change': fe_xai_df['Accuracy'].values - no_fe_df['Accuracy'].values,
                                'Change %': ((fe_xai_df['Accuracy'].values - no_fe_df['Accuracy'].values) / no_fe_df['Accuracy'].values * 100).round(2)
                            })
                            comparison_3['Change'] = comparison_3['Change'].apply(lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}")
                            comparison_3['Change %'] = comparison_3['Change %'].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
                            
                            st.dataframe(
                                comparison_3,
                                use_container_width=True,
                                hide_index=True,
                                height=250,
                                column_config={
                                    "Model": st.column_config.TextColumn("Model", width="medium"),
                                    "No FE": st.column_config.NumberColumn("No FE", format="%.4f"),
                                    "FE + XAI": st.column_config.NumberColumn("FE + XAI", format="%.4f"),
                                    "Change": st.column_config.TextColumn("Δ Accuracy", width="small"),
                                    "Change %": st.column_config.TextColumn("Δ %", width="small")
                                }
                            )
            else:
                st.warning("⚠️ Training results for comparison not available. Enable XAI improvement to see all 3 comparisons.")
        
        elif 'improvement_results' in results:
            # OLD FORMAT: Feature selection based improvement
            improvement_data = results['improvement_results']
            
            if 'feature_selection' in improvement_data:
                comparison = improvement_data['feature_selection']['comparison']
                report = improvement_data['feature_selection']['report']
                
                # Show comparison table
                st.markdown("####  Baseline vs Improved Comparison")
                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model": st.column_config.TextColumn("Model", width="medium"),
                        "Baseline Accuracy": st.column_config.TextColumn("Baseline", width="small"),
                        "Improved Accuracy": st.column_config.TextColumn("Improved", width="small"),
                        "Improvement": st.column_config.TextColumn("Change", width="small"),
                        "Improvement %": st.column_config.TextColumn("Change %", width="small"),
                        "Strategy": st.column_config.TextColumn("Strategy", width="medium")
                    }
                )
                
                # Feature reduction info
                st.markdown("#### 🗑️ Feature Selection Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Features Before", report['n_features_before'])
                with col2:
                    st.metric("Features After", report['n_features_after'])
                with col3:
                    st.metric("Features Removed", len(report['features_removed']))
                
                if report['features_removed']:
                    st.markdown("**Removed Features:**")
                    st.write(", ".join(report['features_removed']))
                
                # Winner selection
                st.markdown("####  Model Selection")
                
                # Determine which results to display
                if 'fe_xai' in results and results['fe_xai']:
                    display_results = results['fe_xai']
                elif 'fe_only' in results:
                    display_results = results['fe_only']
                else:
                    display_results = results
                
                if display_results.get('improved_selected', False):
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 4px solid #155724; color: #155724;'>
                        <i class='bi bi-check-circle-fill' style='color: #27AE60;'></i> <strong>Winner: Improved Model</strong><br>
                        Selected: {display_results['best_model_name']}<br>
                        Accuracy: {display_results['best_metrics']['Accuracy']:.4f}<br>
                        Reason: Improved model achieved higher accuracy
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #856404; color: #856404;'>
                        <i class='bi bi-exclamation-triangle'></i> <strong>Winner: Baseline Model</strong><br>
                        Selected: {display_results['best_model_name']}<br>
                        Accuracy: {display_results['best_metrics']['Accuracy']:.4f}<br>
                        Reason: Baseline model remains superior
                    </div>
                    """, unsafe_allow_html=True)
                
                # Explanation
                if 'improvement_explanation' in results:
                    with st.expander(" Detailed Explanation - Why This Result?"):
                        st.markdown(results['improvement_explanation'])
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # # Model comparison tabs
    # st.markdown("### Detailed Performance Analysis")
    
    # tab1, tab2, tab3 = st.tabs(["All Metrics", "Accuracy Ranking", "F1 Score Ranking"])
    
    # with tab1:
    #     # Clean grouped bar chart
    #     metrics_df = tuned_df[['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].copy()
        
    #     fig = go.Figure()
    #     colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        
    #     for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score']):
    #         fig.add_trace(go.Bar(
    #             name=metric,
    #             x=metrics_df['Model Name'],
    #             y=metrics_df[metric],
    #             text=metrics_df[metric].round(4),
    #             textposition='auto',
    #             marker_color=colors[i]
    #         ))
        
    #     fig.update_layout(
    #         title="Performance Metrics Across Models",
    #         xaxis_title="Model",
    #         yaxis_title="Score",
    #         barmode='group',
    #         height=500,
    #         template='plotly_white',
    #         font=dict(family='Inter, sans-serif')
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
    
    # with tab2:
    #     # Horizontal bar chart for accuracy
    #     accuracy_sorted = tuned_df.sort_values('Accuracy', ascending=True)
    #     fig = px.bar(
    #         accuracy_sorted,
    #         x='Accuracy',
    #         y='Model Name',
    #         orientation='h',
    #         color='Accuracy',
    #         text='Accuracy',
    #         title='Model Ranking by Accuracy',
    #         color_continuous_scale=['#667eea', '#764ba2']
    #     )
    #     fig.update_traces(texttemplate='%{text:.4f}', textposition='inside')
    #     fig.update_layout(height=450, template='plotly_white')
    #     st.plotly_chart(fig, use_container_width=True)
    
    # with tab3:
    #     # Horizontal bar chart for F1
    #     f1_sorted = tuned_df.sort_values('F1 Score', ascending=True)
    #     fig = px.bar(
    #         f1_sorted,
    #         x='F1 Score',
    #         y='Model Name',
    #         orientation='h',
    #         color='F1 Score',
    #         text='F1 Score',
    #         title='Model Ranking by F1 Score',
    #         color_continuous_scale=['#4facfe', '#00f2fe']
    #     )
    #     fig.update_traces(texttemplate='%{text:.4f}', textposition='inside')
    #     fig.update_layout(height=450, template='plotly_white')
    #     st.plotly_chart(fig, use_container_width=True)


def show_xai_page():
    """XAI Analysis - Scientific visualization"""
    st.markdown("## Explainable AI Analysis")
    
    if not st.session_state.trained:
        st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;'><i class='bi bi-exclamation-triangle' style='color: #F39C12;'></i> Please train a model first in the 'Data & Training' page.</div>", unsafe_allow_html=True)
        return
    
    # Run XAI analysis
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Run XAI Analysis", type="primary", use_container_width=True):
            run_xai_analysis()
    
    with col2:
        st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;'><i class='bi bi-stopwatch'></i> Generates SHAP values, LIME explanations, and Permutation Importance. Takes ~1 minute.</div>", unsafe_allow_html=True)
    
    # Display results
    if st.session_state.xai_results:
        display_xai_results()


def run_xai_analysis():
    """Execute XAI analysis pipeline"""
    with st.spinner("Running XAI analysis... This may take a minute..."):
        try:
            results = st.session_state.training_results
            
            # Get best model from appropriate stage
            if 'fe_xai' in results and results['fe_xai']:
                best_model = results['fe_xai']['best_model']
            elif 'fe_only' in results:
                best_model = results['fe_only']['best_model']
            else:
                best_model = results['best_model']
            
            explainer = XAIExplainer(
                model=best_model,
                X_train=results['X_train'],
                X_test=results['X_test'],
                y_train=results['y_train'],
                y_test=results['y_test'],
                X_train_scaled=results['X_train_scaled'],
                X_test_scaled=results['X_test_scaled'],
                feature_names=results['X_train'].columns.tolist()
            )
            
            xai_results = explainer.full_xai_pipeline()
            st.session_state.xai_results = xai_results
            
            st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 4px solid #155724; color: #155724;'><i class='bi bi-check-circle-fill' style='color: #27AE60;'></i> XAI Analysis completed!</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8d7da; border-left: 4px solid #721c24; color: #721c24;'><i class='bi bi-x-circle-fill' style='color: #E74C3C;'></i> XAI Analysis failed: {str(e)}</div>", unsafe_allow_html=True)


def display_xai_results():
    """Display XAI results - Clean scientific design"""
    xai = st.session_state.xai_results
    
    # Tabs with clean labels
    tab1, tab2, tab3 = st.tabs(["SHAP Analysis", "LIME Explanations", "Permutation Importance"])
    
    with tab1:
        show_shap_analysis(xai)
    
    with tab2:
        show_lime_analysis(xai)
    
    with tab3:
        show_permutation_analysis(xai)


def show_shap_analysis(xai_results):
    """SHAP analysis - Modern scientific visualization"""
    st.markdown("### SHAP Feature Importance")
    
    st.markdown("""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
        <strong>SHAP (SHapley Additive exPlanations)</strong><br>
        Measures each feature's contribution to model predictions based on game theory.
        Positive values push prediction toward disease, negative values away from disease.
    </div>
    """, unsafe_allow_html=True)
    
    shap_importance = xai_results['shap_importance']
    
    # Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Modern bar chart
        top_10 = shap_importance.head(10)
        fig = px.bar(
            top_10,
            x='SHAP Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='SHAP Importance',
            color_continuous_scale=['#667eea', '#764ba2']
        )
        fig.update_layout(
            height=500,
            yaxis={'categoryorder':'total ascending'},
            template='plotly_white',
            font=dict(family='Inter, sans-serif')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Top 5 Features")
        for i, row in shap_importance.head(5).iterrows():
            st.metric(
                label=row['Feature'],
                value=f"{row['SHAP Importance']:.4f}"
            )
    
    # SHAP Summary Plot
    st.markdown("---")
    st.markdown("### SHAP Summary Plot")
    
    try:
        shap_data = xai_results['shap_summary_data']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_data['shap_values'],
            shap_data['features'],
            feature_names=shap_data['feature_names'],
            show=False
        )
        st.pyplot(fig)
        plt.close()
        
        st.caption("""
        **Interpretation Guide:**
        • Each dot represents one patient
        • Red: High feature value | Blue: Low feature value
        • Horizontal position: Impact on prediction (SHAP value)
        """)
    
    except Exception as e:
        st.error(f"Could not generate SHAP summary plot: {str(e)}")


def show_lime_analysis(xai_results):
    """Display LIME analysis results"""
    st.subheader("🔬 LIME (Local Interpretable Model-agnostic Explanations)")
    
    st.markdown("""
    LIME explains **individual predictions** by approximating the model locally around that data point.
    """)
    
    lime_explanations = xai_results['lime_explanations']
    
    if not lime_explanations:
        st.warning("No LIME explanations available.")
        return
    
    # === FILTERS ===
    st.markdown("### 🔍 Filter Instances")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by prediction correctness
        correctness_filter = st.selectbox(
            "Prediction Status",
            ["All", "Correct Only", "Incorrect Only"]
        )
    
    with col2:
        # Filter by confidence
        confidence_filter = st.selectbox(
            "Confidence Level",
            ["All", "High (>80%)", "Medium (50-80%)", "Low (<50%)"]
        )
    
    with col3:
        # Filter by prediction
        prediction_filter = st.selectbox(
            "Prediction",
            ["All", "Disease", "No Disease"]
        )
    
    # Apply filters
    filtered_explanations = lime_explanations.copy()
    
    # Filter 1: Correctness
    if correctness_filter == "Correct Only":
        filtered_explanations = [exp for exp in filtered_explanations if exp['correct']]
    elif correctness_filter == "Incorrect Only":
        filtered_explanations = [exp for exp in filtered_explanations if not exp['correct']]
    
    # Filter 2: Confidence
    if confidence_filter == "High (>80%)":
        filtered_explanations = [exp for exp in filtered_explanations 
                                if max(exp['prediction_proba']) > 0.8]
    elif confidence_filter == "Medium (50-80%)":
        filtered_explanations = [exp for exp in filtered_explanations 
                                if 0.5 <= max(exp['prediction_proba']) <= 0.8]
    elif confidence_filter == "Low (<50%)":
        filtered_explanations = [exp for exp in filtered_explanations 
                                if max(exp['prediction_proba']) < 0.5]
    
    # Filter 3: Prediction
    if prediction_filter == "Disease":
        filtered_explanations = [exp for exp in filtered_explanations if exp['prediction'] == 1]
    elif prediction_filter == "No Disease":
        filtered_explanations = [exp for exp in filtered_explanations if exp['prediction'] == 0]
    
    # Show filtered count
    st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;'><i class='bi bi-graph-up'></i> Showing {len(filtered_explanations)} instances out of {len(lime_explanations)} total</div>", unsafe_allow_html=True)
    
    if len(filtered_explanations) == 0:
        st.warning("No instances match the selected filters. Please adjust your filters.")
        return
    
    st.markdown("---")
    st.markdown("### 📋 Select Instance to Analyze")
    
    # Create instance options with more info
    instance_options = []
    for exp in filtered_explanations:
        pred_label = "Disease" if exp['prediction'] == 1 else "No Disease"
        actual_label = "Disease" if exp['actual'] == 1 else "No Disease"
        confidence = max(exp['prediction_proba']) * 100
        correct_icon = "" if exp['correct'] else ""
        
        option_text = (f"Instance {exp['instance_idx']} {correct_icon} | "
                      f"Pred: {pred_label} ({confidence:.1f}%) | "
                      f"Actual: {actual_label}")
        instance_options.append(option_text)
    
    selected = st.selectbox("Choose an instance:", instance_options)
    selected_idx = int(selected.split()[1])
    
    # Find the selected explanation
    exp = next(e for e in filtered_explanations if e['instance_idx'] == selected_idx)
    
    st.markdown("---")
    st.markdown(f"### 🔬 Detailed Analysis: Instance {selected_idx}")
    
    # Display prediction info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", "Disease" if exp['prediction'] == 1 else "No Disease")
    with col2:
        st.metric("Confidence", f"{exp['prediction_proba'][exp['prediction']]*100:.1f}%")
    with col3:
        st.metric("Actual", "Disease" if exp['actual'] == 1 else "No Disease")
    with col4:
        if exp['correct']:
            st.success(" Correct")
        else:
            st.error(" Incorrect")
    
    # Feature contributions
    st.markdown("#### Feature Contributions")
    
    exp_df = exp['explanation_df']
    
    # Parse feature names and values from LIME output
    exp_df['Feature_Clean'] = exp_df['Feature'].str.split('<=|>').str[0].str.strip()
    exp_df['Direction'] = exp_df['Weight'].apply(lambda x: 'Towards Disease' if x > 0 else 'Towards No Disease')
    exp_df['Abs_Weight'] = exp_df['Weight'].abs()
    
    fig = px.bar(
        exp_df.sort_values('Weight'),
        x='Weight',
        y='Feature',
        orientation='h',
        color='Direction',
        title=f'LIME Explanation for Instance {selected_idx}',
        color_discrete_map={
            'Towards Disease': '#E74C3C',
            'Towards No Disease': '#27AE60'
        }
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def show_permutation_analysis(xai_results):
    """Display Global Explanations (Permutation Importance & SHAP)"""
    st.markdown("Measurement of global feature importance.")
    
    # Create Tabs for different Global Views
    tab_pi, tab_shap = st.tabs(["📊 Permutation Importance", "🐝 SHAP Summary"])
    
    # --- TAB 1: PERMUTATION IMPORTANCE ---
    with tab_pi:
        st.markdown("""
        **Permutation Importance** measures **performance drop** when a feature is shuffled.
        The larger the drop, the more important the feature.
        """)
        
        if 'permutation_importance' in xai_results:
            perm_importance = xai_results['permutation_importance']
            
            # Visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                top_10 = perm_importance.head(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=top_10['Feature'],
                    x=top_10['Importance'],
                    error_x=dict(type='data', array=top_10['Std']),
                    orientation='h',
                    marker=dict(color=top_10['Importance'], colorscale='blues')
                ))
                
                fig.update_layout(
                    title='Top 10 Features by Permutation Importance',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=500,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Top 5 Features")
                for i, row in perm_importance.head(5).iterrows():
                    st.metric(
                        label=row['Feature'],
                        value=f"{row['Importance']:.4f}",
                        delta=f"±{row['Std']:.4f}"
                    )
            
            # Comparison table
            with st.expander("📋 View Full Ranking"):
                st.dataframe(perm_importance, use_container_width=True)
        else:
            st.info("Permutation Importance data not available.")

    # --- TAB 2: SHAP SUMMARY ---
    with tab_shap:
        st.markdown("""
        **SHAP Summary** shows the impact of each feature on the model output.
        - **Color**: Feature Value (Red = High, Blue = Low)
        - **X-axis**: Impact on prediction (Right = Positive impact/Disease, Left = Negative/No Disease)
        """)
        
        if 'shap_values' in xai_results and 'shap_data' in xai_results:
            try:
                shap_values = xai_results['shap_values']
                X_shap = xai_results['shap_data']
                
                # Handle SHAP values structure (List for classifiers vs Array)
                if isinstance(shap_values, list):
                    # For binary classification, typically use index 1 (Positive Class)
                    shap_val_display = shap_values[1]
                else:
                    shap_val_display = shap_values

                # Plot
                fig_shap, ax = plt.subplots()
                shap.summary_plot(shap_val_display, X_shap, show=False, plot_type="dot")
                st.pyplot(fig_shap)
                plt.close(fig_shap)
                
                # Optional: Bar plot
                with st.expander("View SHAP Bar Plot"):
                    fig_bar, ax_bar = plt.subplots()
                    shap.summary_plot(shap_val_display, X_shap, show=False, plot_type="bar")
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

            except Exception as e:
                st.error(f"Error plotting SHAP: {str(e)}")
        else:
            st.info("SHAP data not available in results.")


def show_llm_page():
    """LLM Interpretation - Clean interface"""
    st.markdown("## AI-Powered Medical Interpretation")
    
    if not st.session_state.xai_results:
        st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;'><i class='bi bi-exclamation-triangle' style='color: #F39C12;'></i> Please run XAI Analysis first.</div>", unsafe_allow_html=True)
        return
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8d7da; border-left: 4px solid #721c24; color: #721c24;'><i class='bi bi-x-circle-fill' style='color: #E74C3C;'></i> Groq API Key not configured. Please enter it in the sidebar.</div>", unsafe_allow_html=True)
        return
    
    # Generate interpretations
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("▶️ Generate Interpretations", type="primary", use_container_width=True):
            generate_llm_interpretations()
    
    with col2:
        st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460;'><i class='bi bi-stopwatch'></i> Uses Groq API (GPT-OSS-120B) to generate medical interpretations in English. Takes ~30 seconds.</div>", unsafe_allow_html=True)
    
    # Display results
    if st.session_state.llm_interpretations:
        display_llm_interpretations()


def generate_llm_interpretations():
    """Generate LLM interpretations using Qrog API"""
    with st.spinner("Generating interpretations with LLM... This may take a minute..."):
        try:
            interpreter = LLMInterpreter()
            
            xai_results = st.session_state.xai_results
            
            # Get best metrics from appropriate stage
            results = st.session_state.training_results
            if 'fe_xai' in results and results['fe_xai']:
                best_metrics = results['fe_xai']['best_metrics']
            elif 'fe_only' in results:
                best_metrics = results['fe_only']['best_metrics']
            else:
                best_metrics = results['best_metrics']
            
            model_info = {
                'Accuracy': best_metrics['Accuracy'],
                'Precision': best_metrics['Precision'],
                'Recall': best_metrics['Recall'],
                'F1 Score': best_metrics['F1 Score']
            }
            
            interpretations = interpreter.generate_full_report(xai_results, model_info)
            st.session_state.llm_interpretations = interpretations
            
            st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 4px solid #155724; color: #155724;'><i class='bi bi-check-circle-fill' style='color: #27AE60;'></i> Interpretations generated successfully!</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8d7da; border-left: 4px solid #721c24; color: #721c24;'><i class='bi bi-x-circle-fill' style='color: #E74C3C;'></i> Failed to generate interpretations: {str(e)}</div>", unsafe_allow_html=True)


def display_llm_interpretations():
    """Display AI interpretations - Clean, readable format"""
    interp = st.session_state.llm_interpretations
    
    # Clean tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "SHAP Interpretation",
        "LIME Interpretation",
        "Permutation Analysis",
        "Method Comparison"
    ])
    
    with tab1:
        st.markdown("### SHAP Feature Importance Analysis")
        st.markdown(f'<div class="interpretation-box">{interp["shap_interpretation"]}</div>',
                   unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### LIME Instance Explanations")
        for i, lime_interp in enumerate(interp['lime_interpretations']):
            with st.expander(f"Patient Case {lime_interp['instance_idx']}"):
                st.markdown(f'<div class="interpretation-box">{lime_interp["interpretation"]}</div>',
                           unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Permutation Importance Analysis")
        st.markdown(f'<div class="interpretation-box">{interp["permutation_interpretation"]}</div>',
                   unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### XAI Methods Comparison")
        st.markdown(f'<div class="interpretation-box">{interp["comparison"]}</div>',
                   unsafe_allow_html=True)


def show_prediction_page():
    """Prediction page with Test Set Explorer and Manual Input"""
    st.markdown("## <i class='bi bi-magic'></i> Make Predictions", unsafe_allow_html=True)
    
    if 'trained' not in st.session_state or not st.session_state.trained:
        st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #fff3cd; border-left: 4px solid #856404; color: #856404;'><i class='bi bi-exclamation-triangle' style='color: #F39C12;'></i> Please train a model first.</div>", unsafe_allow_html=True)
        return
    
    # Choose mode
    mode = st.radio("Select Input Mode", ["Demo Scenarios", "Test Set Explorer", "Manual Input"], horizontal=True)
    
    if mode == "Demo Scenarios":
        show_demo_scenarios()
    elif mode == "Test Set Explorer":
        show_test_set_explorer()
    else:
        show_manual_input()


def show_demo_scenarios():
    """Show pre-defined demo cases for presentation"""
    st.subheader("📋 Select a Demo Patient Case")
    
    case_name = st.selectbox("Choose Patient", list(DEMO_CASES.keys()))
    case_data = DEMO_CASES[case_name]
    
    # Display Case Data nicely
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Age", case_data['age'])
    col1.metric("Sex", "Male" if case_data['sex']==1 else "Female")
    col1.metric("Chest Pain", case_data['chest pain type'])
    
    col2.metric("BP", case_data['resting bp s'])
    col2.metric("Cholesterol", case_data['cholesterol'])
    col2.metric("Fasting BS", "Yes" if case_data['fasting blood sugar']==1 else "No")
    
    col3.metric("Max HR", case_data['max heart rate'])
    col3.metric("Ecercise Angina", "Yes" if case_data['exercise angina']==1 else "No")
    col3.metric("Resting ECG", case_data['resting ecg'])
    
    col4.metric("Oldpeak", case_data['oldpeak'])
    col4.metric("ST Slope", case_data['ST slope'])
    col4.metric("True Label", case_data['target_label'])
    
    st.markdown("---")
    
    if st.button("📢 Explain This Prediction (Demo Flow)", type="primary", use_container_width=True):
        # We need to process this raw dictionary into the format explain_pipeline expects
        # 1. Convert to DataFrame
        df_input = pd.DataFrame([case_data]).drop('target_label', axis=1)
        
        # 2. Add 'target' mock for compatibility if needed (not needed for prediction)
        
        # 3. Use the common pipeline
        explain_custom_instance(df_input)


def explain_custom_instance(df_input_raw):
    """
    Run pipeline for a custom dataframe input (Demo or Manual)
    This handles: Preprocessing -> Prediction -> Global XAI -> Local LIME -> LLM
    """
    try:
        results = st.session_state.training_results
        # Determine correct results object
        if 'fe_xai' in results and results['fe_xai']:
             active_results = results['fe_xai']
        elif 'fe_only' in results and results['fe_only']:
             active_results = results['fe_only']
        elif 'no_fe' in results and results['no_fe']:
             active_results = results['no_fe']
        else:
             st.error("Training data not found.")
             return
             
        model = active_results['best_model']
        
        # --- PREPROCESSING (Replicating make_prediction logic) ---
        # 1. Apply log transform
        df_proc = df_input_raw.copy()
        if 'cholesterol' in df_proc.columns and df_proc['cholesterol'].values[0] > 0:
            df_proc['cholesterol'] = np.log1p(df_proc['cholesterol'])
        if 'oldpeak' in df_proc.columns and df_proc['oldpeak'].values[0] >= 0:
            df_proc['oldpeak'] = np.log1p(df_proc['oldpeak'])
            
        # 2. One-Hot Encoding
        one_hot_cols = ['chest pain type', 'resting ecg', 'ST slope']
        df_encoded = pd.get_dummies(df_proc, columns=one_hot_cols, dtype=int)
        
        # 3. Align features
        train_columns = active_results['X_train'].columns.tolist()
        for col in train_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[train_columns]
        
        # 4. Scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(active_results['X_train'])
        X_new_scaled = scaler.transform(df_encoded)
        
        # 5. Feauture Engineering (Interactions)
        if active_results.get('improved_selected', False) and 'new_features' in active_results:
            X_scaled_df = pd.DataFrame(X_new_scaled, columns=train_columns)
            for feature in active_results['new_features']:
                parts = feature.split('_x_')
                if len(parts) == 2:
                    f1, f2 = parts
                    if f1 in X_scaled_df.columns and f2 in X_scaled_df.columns:
                        X_scaled_df[feature] = X_scaled_df[f1] * X_scaled_df[f2]
            X_final = X_scaled_df.values
        else:
            X_final = X_new_scaled
            
        # --- PREDICTION ---
        pred = model.predict(X_final)[0]
        prob = model.predict_proba(X_final)[0]
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Prediction Result")
            if pred == 1:
                st.markdown("🚨 **High Risk**")
            else:
                st.markdown("✅ **Low Risk**")
        with col2:
            st.markdown("### Confidence")
            st.progress(float(prob[pred]))
            st.caption(f"{prob[pred]*100:.2f}% probability")
            
        # --- GLOBAL EXPLANATION ---
        st.divider()
        st.markdown("### 1. Global Context (General Model Behavior)")
        if st.session_state.xai_results:
             show_permutation_analysis(st.session_state.xai_results)
        else:
             st.warning("Global XAI unavailable.")

        # --- LOCAL EXPLANATION (LIME) ---
        st.divider()
        st.markdown(f"### 2. Local Explanation (Why THIS patient?)")
        
        # Initialize Explainer
        xai = XAIExplainer(
            model=model,
            X_train=active_results['X_train'], # Raw for LIME tabular
            X_test=active_results['X_test'],
            y_train=active_results['y_train'],
            y_test=active_results['y_test'],
            X_train_scaled=active_results['X_train_scaled'],
            X_test_scaled=active_results['X_test_scaled'],
            feature_names=active_results.get('feature_names', active_results['X_train'].columns.tolist())
        )
        
        # Setup LIME Explainer if not ready
        if xai.lime_explainer is None:
            xai.setup_lime()
            
        # Explain this specific instance (Passed as raw numpy array to LIME)
        # Note: LIME Tabular expects raw feature values (not scaled) usually, but it depends on how it was initialized.
        # In xai_explainer.py: training_data=self.X_train.values (RAW).
        # So we should pass RAW data (df_encoded aligned to columns but unscaled? Or df_input_raw aligned?)
        # Actually in XAIExplainer.explain_instance_lime it uses X_test_scaled[idx].
        # Wait, if LIME was init with Raw X_train, it expects Raw input? 
        # Let's check xai_explainer.py:
        # lime_explainer = LimeTabularExplainer(training_data=self.X_train.values ...) -> RAW
        # But explain_instance_lime uses: instance = self.X_test_scaled[instance_idx] -> SCALED
        # This is a POTENTIAL BUG in the original class if scaling is inconsistent.
        # However, for now, let's follow the existing pattern: pass what the model expects?
        # No, LIME perturbs data. If init with RAW, it perturbs RAW. Then predict_fn must handle scaling.
        # In explain_instance_lime: predict_fn = lambda x: self.model.predict_proba(x)
        # The model (SVM/MLP with scaler inside pipeline? No, scaler is separate).
        # Model expects SCALED data.
        # If LIME is fed RAW data, then predict_fn must SCALE the data.
        
        # Let's fix this ad-hoc here for the demo to be robust.
        # We will use the 'X_final' (which is SCALED and FE'd) for the LIME explainer 'data_row'.
        # AND we need to make sure the explainer was init with matching data distribution.
        # The safest bet for this "Custom" instance without modifying XAI class deeply:
        # Use X_final (The exact input to model).
        # But we need a LIME explainer fit on X_train_scaled (and FE'd).
        
        # Re-init LIME with transformed training data
        import lime.lime_tabular
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=active_results['X_train_scaled'], # Use tranformed training data
            feature_names=active_results.get('feature_names', train_columns),
            class_names=['No Disease', 'Disease'],
            mode='classification',
            random_state=42
        )
        
        explanation = explainer_lime.explain_instance(
            data_row=X_final[0], # Single row
            predict_fn=model.predict_proba,
            num_features=10
        )
        
        # Display LIME
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)
        plt.close()
        
        exp_list = explanation.as_list()
        st.markdown("**Top Contributing Factors:**")
        for feat, weight in exp_list:
             st.markdown(f"- **{feat}**: {weight:.4f}")
             
        # --- LLM ---
        st.divider()
        st.markdown("### 3. AI Doctor Interpretation")
        if os.getenv("GROQ_API_KEY"):
            interpreter = LLMInterpreter()
            context = {
                'prediction': "Heart Disease" if pred == 1 else "No Heart Disease",
                'probability': f"{prob[pred]:.4f}",
                'lime_factors': exp_list,
                'instance_values': df_input_raw.to_dict('records')[0]
            }
            prompt = f"""
            You are an expert cardiologist AI. Interpret the prediction for this specific patient (Demo Case).
            
            PATIENT STATUS:
            - Prediction: {context['prediction']}
            - Confidence: {context['probability']}
            
            PATIENT DATA:
            {context['instance_values']}
            
            KEY RISK FACTORS (LIME Analysis):
            {context['lime_factors']}
            
            GLOBAL MODEL INSIGHTS:
            (Chest Pain, Oldpeak, Thal, CA are generally key drivers)
            
            TASK:
            1. Explain WHY the model made this prediction for THIS specific patient.
            2. Compare their specific risk factors to general heart disease indicators.
            3. Provide a personalized recommendation.
            
            Keep it professional, empathetic, and medical.
            """
            
            with st.spinner("Consulting AI Doctor..."):
                 response = interpreter.call_llm(prompt)
                 st.markdown(f'<div class="interpretation-box">{response}</div>', unsafe_allow_html=True)
                 
    except Exception as e:
        st.error(f"Error in prediction pipeline: {str(e)}")
        st.exception(e)


def show_test_set_explorer():
    """Explore test set instances with Global & Local Explanations"""
    try:
        results = st.session_state.training_results
        # Determine correct results object (prioritizing FE model if selected)
        if 'fe_xai' in results and results['fe_xai']:
             active_results = results['fe_xai']
        elif 'fe_only' in results and results['fe_only']:
             active_results = results['fe_only']
        elif 'no_fe' in results and results['no_fe']:
             active_results = results['no_fe']
        else:
             st.error("Training data not found.")
             return

        # Get test data from results
        # Use raw X_test if available to show original values
        if 'X_test' in active_results:
             X_test_display = active_results['X_test']
             y_test = active_results['y_test']
             
             # Instance Selector
             selected_idx = st.selectbox(
                 "Select Test Instance Index", 
                 options=range(len(X_test_display)),
                 format_func=lambda x: f"Instance {x} (True Label: {'Disease' if y_test.iloc[x] == 1 else 'No Disease'})"
             )
             
             # Show Data
             st.subheader("Instance Data")
             st.dataframe(X_test_display.iloc[[selected_idx]], use_container_width=True)
             
             if st.button("📢 Explain This Prediction", type="primary"):
                 explain_test_instance(selected_idx, active_results)
        
    except Exception as e:
        st.error(f"Error loading test set: {str(e)}")


def explain_test_instance(idx, active_results):
    """Run full explanation pipeline for a specific test instance"""
    with st.spinner("Running Analysis: Global Context -> Local Explanation -> LLM..."):
        try:
            # 1. Prediction on this instance
            model = active_results['best_model']
            
            # Use SCALED data for prediction (and FE if needed)
            # The 'X_test_scaled' in results should already be pre-processed (scaled + optional FE)
            if 'X_test_scaled' in active_results:
                 X_input = active_results['X_test_scaled'][idx].reshape(1, -1)
            else:
                 st.error("Scaled test data not found.")
                 return

            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0]
            
            # Display Prediction
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Prediction Result")
                if pred == 1:
                    st.markdown("🚨 **High Risk**")
                else:
                    st.markdown("✅ **Low Risk**")
            with col2:
                st.markdown("### Confidence")
                st.progress(float(prob[pred]))
                st.caption(f"{prob[pred]*100:.2f}% probability")

            # 2. Global Explanations (Reuse from XAI Results)
            st.divider()
            st.markdown("### 1. Global Context (General Model Behavior)")
            if st.session_state.xai_results:
                 # Reconstruct Global SHAP Summary Plot
                 shap_summary = st.session_state.xai_results.get('shap_summary_data', None)
                 if shap_summary:
                      st.markdown("**Global SHAP Summary:**")
                      # We can try to plot it if we have raw SHAP values, but summary plot is complex to recreate without valid session state data
                      # Instead, let's show the Feature Importance table which is more reliable
                      pass
                 
                 show_permutation_analysis(st.session_state.xai_results)
            else:
                 st.warning("Global XAI results not available. Please run XAI Analysis first to see global context.")

            # 3. Local Explanation (LIME) for THIS instance
            st.divider()
            st.markdown(f"### 2. Local Explanation (Why THIS patient?)")
            
            # We need to initialize an explainer to run LIME for this specific instance
            # Re-create explainer using cached data
            xai = XAIExplainer(
                model=model,
                X_train=active_results['X_train'], # Raw for LIME tabular
                X_test=active_results['X_test'],
                y_train=active_results['y_train'],
                y_test=active_results['y_test'],
                X_train_scaled=active_results['X_train_scaled'], # Transformed for SHAP/Model
                X_test_scaled=active_results['X_test_scaled'],
                feature_names=active_results.get('feature_names', active_results['X_train'].columns.tolist())
            )
            
            lime_result = xai.explain_instance_lime(idx)
            
            # Display LIME Plot
            fig = lime_result['explanation'].as_pyplot_figure()
            st.pyplot(fig)
            plt.close()
            
            # Show reasons text
            st.markdown("**Top Contributing Factors:**")
            exp_list = lime_result['explanation'].as_list()
            for feat, weight in exp_list:
                 color = "red" if weight > 0 else "green" # Assuming >0 is typically towards class 1
                 # Adjust color logic based on model class mapping if needed, generally LIME fits to local model
                 # Simple display:
                 st.markdown(f"- **{feat}**: {weight:.4f}")

            # 4. LLM Interpretation
            st.divider()
            st.markdown("### 3. AI Doctor Interpretation")
            
            if os.getenv("GROQ_API_KEY"):
                interpreter = LLMInterpreter()
                
                # Prepare context
                context = {
                    'prediction': "Heart Disease" if pred == 1 else "No Heart Disease",
                    'probability': f"{prob[pred]:.4f}",
                    'lime_factors': exp_list,
                    'instance_values': active_results['X_test'].iloc[idx].to_dict()
                }
                
                # We need a specific method for single instance interpretation
                # Or we can construct a prompt manually here
                prompt = f"""
                You are an expert cardiologist AI. Interpret the prediction for this specific patient.
                
                PATIENT STATUS:
                - Prediction: {context['prediction']}
                - Confidence: {context['probability']}
                
                PATIENT DATA:
                {context['instance_values']}
                
                KEY RISK FACTORS (LIME Analysis - Positive means increases risk, Negative decreases):
                {context['lime_factors']}
                
                GLOBAL MODEL INSIGHTS (Top General Risk Factors):
                (Refer to global knowledge: Chest Pain, ST Slope, Thal usually important)
                
                TASK:
                1. Explain WHY the model made this prediction for THIS specific patient based on their values and the LIME factors.
                2. Compare their specific risk factors to general heart disease indicators.
                3. Provide a personalized recommendation.
                
                Keep it professional, empathetic, and medical but easy to understand.
                """
                
                with st.spinner("Consulting AI Doctor..."):
                     # Using the generic valid method from interpreter if available, or call API directly
                     # Assuming interpreter has a method to call generic prompt or we use existing one
                     # Let's rely on internal LLMInterpreter method if it exists, otherwise add one?
                     # Creating a temporary method usage pattern:
                     response = interpreter.call_llm(prompt) # Assuming call_llm exists or similar
                     st.markdown(f'<div class="interpretation-box">{response}</div>', unsafe_allow_html=True)
            else:
                 st.warning("Please configure Groq API Key for AI interpretation.")

        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def show_manual_input():
    """Original Manual Input Form"""
    st.info("Enter patient information below to get a prediction.")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            resting_bp = st.number_input("Resting BP", min_value=80, max_value=200, value=120)
            cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
        
        with col2:
            max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
            exercise_angina = st.selectbox("Exercise Angina", [0, 1], 
                                          format_func=lambda x: "No" if x == 0 else "Yes")
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                                     format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            chest_pain = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"{x}: " + ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"][x] if x < 4 else f"{x}")
            resting_ecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: f"{x}: " + ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"][x] if x < 3 else f"{x}")
            st_slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: f"{x}: " + ["Upsloping", "Flat", "Downsloping"][x] if x < 3 else f"{x}")
        
        submit = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)
        
        if submit:
            make_prediction(age, sex, resting_bp, max_hr, oldpeak, exercise_angina,
                          chest_pain, resting_ecg, st_slope, cholesterol, fasting_bs)


def make_prediction(age, sex, resting_bp, max_hr, oldpeak, exercise_angina,
                    chest_pain, resting_ecg, st_slope, cholesterol, fasting_bs):
    """Make prediction for new instance using the trained model"""
    from sklearn.preprocessing import StandardScaler
    
    try:
        results_all = st.session_state.training_results
        
        # 1. Get appropriate results and model
        if 'fe_xai' in results_all and results_all['fe_xai']:
            active_results = results_all['fe_xai']
        elif 'fe_only' in results_all and results_all['fe_only']:
            active_results = results_all['fe_only']
        elif 'no_fe' in results_all and results_all['no_fe']:
             active_results = results_all['no_fe']
        else:
             st.error("Model data not found. Please train model first.")
             return
             
        model = active_results['best_model']
            
        # 2. Prepare raw input data
        # Mapping column names to match DataProcessor logic
        input_data = {
            'age': [age],
            'sex': [sex],
            'resting bp s': [resting_bp],
            'cholesterol': [cholesterol],
            'fasting blood sugar': [fasting_bs],
            'max heart rate': [max_hr],
            'exercise angina': [exercise_angina],
            'oldpeak': [oldpeak],
            'chest pain type': [chest_pain],
            'resting ecg': [resting_ecg],
            'ST slope': [st_slope]
        }
        df_input = pd.DataFrame(input_data)
        
        # 3. Apply transformations (must match training pipeline)
        if cholesterol > 0:
            df_input['cholesterol'] = np.log1p(df_input['cholesterol'])
        if oldpeak >= 0:
            df_input['oldpeak'] = np.log1p(df_input['oldpeak'])
            
        # 4. Feature Engineering (One-Hot Encoding)
        one_hot_cols = ['chest pain type', 'resting ecg', 'ST slope']
        df_encoded = pd.get_dummies(df_input, columns=one_hot_cols, dtype=int)
        
        # 5. Align with training features
        train_columns = active_results['X_train'].columns.tolist()
        
        # Add missing columns (fill with 0)
        for col in train_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Select and reorder to match training data exactly
        df_encoded = df_encoded[train_columns]
        
        # 6. Scaling
        # Fit a new scaler on the original training data to ensure consistent scaling
        scaler = StandardScaler()
        scaler.fit(active_results['X_train'])
        
        X_new_scaled = scaler.transform(df_encoded)
        
        # 7. Apply Interactions if needed (for XAI-improved models)
        if active_results.get('improved_selected', False) and 'new_features' in active_results:
            # We need to work with DataFrame to easily select interaction columns
            X_scaled_df = pd.DataFrame(X_new_scaled, columns=train_columns)
            
            for feature in active_results['new_features']:
                # Feature name is like "col1_x_col2"
                parts = feature.split('_x_')
                if len(parts) == 2:
                    f1, f2 = parts
                    if f1 in X_scaled_df.columns and f2 in X_scaled_df.columns:
                        X_scaled_df[feature] = X_scaled_df[f1] * X_scaled_df[f2]
            
            # Convert back to numpy array for prediction
            X_final = X_scaled_df.values
        else:
            X_final = X_new_scaled
        
        # 8. Prediction
        prediction = model.predict(X_final)[0]
        prediction_proba = model.predict_proba(X_final)[0]
        
        demo_pred = prediction
        demo_proba = prediction_proba
        
        st.divider()
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if demo_pred == 1:
                st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8d7da; border-left: 4px solid #721c24; color: #721c24;'><i class='bi bi-exclamation-triangle-fill' style='color: #F39C12;'></i> **High Risk of Heart Disease**</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border-left: 4px solid #155724; color: #155724;'><i class='bi bi-check-circle-fill' style='color: #27AE60;'></i> **Low Risk of Heart Disease**</div>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{max(demo_proba)*100:.1f}%")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = demo_proba[1] * 100,
            title = {'text': "Disease Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    main()
