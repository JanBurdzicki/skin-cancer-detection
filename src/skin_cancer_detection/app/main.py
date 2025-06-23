"""
Simple Streamlit application for skin cancer detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import io
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from skin_cancer_detection.api.main import app as api_app
from skin_cancer_detection.api.main import predict_endpoint, health_check, get_models

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">🔬 Skin Cancer Detection System</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Welcome to the Skin Cancer Detection System</h3>
        <p>This application uses machine learning models to analyze dermatoscopic images
        and clinical features for skin cancer detection. Upload an image or enter clinical
        data to get predictions with explanations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "📊 Clinical Data", "ℹ️ About"])

    with tab1:
        render_image_analysis()

    with tab2:
        render_clinical_analysis()

    with tab3:
        render_about()

    # Medical disclaimer
    render_disclaimer()


def render_image_analysis():
    """Render image analysis interface."""
    st.markdown('<h2 class="sub-header">Dermatoscopic Image Analysis</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Model selection
        model_type = st.selectbox(
            "Select Image Model:",
            ["cnn", "resnet18"],
            help="Choose the deep learning model for image analysis"
        )

        # Model source
        model_source = st.selectbox(
            "Model Source:",
            ["local", "wandb"],
            help="Choose whether to load model from local files or Weights & Biases"
        )

        # Include explanations
        include_explanations = st.checkbox(
            "Include XAI Explanations",
            value=True,
            help="Generate explainable AI visualizations"
        )

    with col2:
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose a dermatoscopic image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a dermatoscopic image for analysis"
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

    # Analysis button
    if uploaded_file is not None:
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                result = analyze_image(uploaded_file, model_type, model_source, include_explanations)

                if result:
                    display_results(result, "image")
                else:
                    st.error("Analysis failed. Please try again.")


def render_clinical_analysis():
    """Render clinical data analysis interface."""
    st.markdown('<h2 class="sub-header">Clinical Feature Analysis</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Model selection
        model_type = st.selectbox(
            "Select Tabular Model:",
            ["xgboost", "lightgbm", "random_forest", "logistic_regression"],
            help="Choose the machine learning model for tabular data analysis"
        )

        # Model source
        model_source = st.selectbox(
            "Model Source:",
            ["local", "wandb"],
            help="Choose whether to load model from local files or Weights & Biases",
            key="tabular_source"
        )

        # Include explanations
        include_explanations = st.checkbox(
            "Include XAI Explanations",
            value=True,
            help="Generate explainable AI feature importance",
            key="tabular_explanations"
        )

    with col2:
        st.markdown("**Enter Clinical Features:**")

        # Sample clinical features (you can expand this based on your dataset)
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        localization = st.selectbox("Localization", [
            "back", "lower extremity", "trunk", "upper extremity",
            "abdomen", "face", "chest", "foot", "hand", "scalp", "neck"
        ])

        # Additional features
        diameter = st.number_input("Diameter (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        asymmetry = st.slider("Asymmetry Score", 0.0, 1.0, 0.5, 0.1)
        border = st.slider("Border Irregularity", 0.0, 1.0, 0.5, 0.1)
        color = st.slider("Color Variation", 0.0, 1.0, 0.5, 0.1)

    # Create clinical data dictionary
    clinical_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "localization": localization,
        "diameter": diameter,
        "asymmetry": asymmetry,
        "border": border,
        "color": color
    }

    # Analysis button
    if st.button("Analyze Clinical Data", type="primary"):
        with st.spinner("Analyzing clinical features..."):
            result = analyze_clinical_data(clinical_data, model_type, model_source, include_explanations)

            if result:
                display_results(result, "tabular")
            else:
                st.error("Analysis failed. Please try again.")


def analyze_image(uploaded_file, model_type: str, model_source: str, include_explanations: bool):
    """Analyze uploaded image using the API."""
    try:
        # Prepare the file for API call
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        # Prepare parameters
        params = {
            "model_type": model_type,
            "model_name": model_type,  # Using model_type as model_name for simplicity
            "model_source": model_source,
            "include_explanations": include_explanations
        }

        # Call the predict function directly (since we're in the same process)
        from fastapi import UploadFile
        from fastapi.datastructures import Headers
        import tempfile

        # Create a temporary file for the API
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()

            # Create UploadFile object
            upload_file = UploadFile(
                filename=uploaded_file.name,
                file=open(tmp_file.name, 'rb'),
                headers=Headers({'content-type': uploaded_file.type})
            )

            # Call the API endpoint
            result = predict_endpoint(
                file=upload_file,
                model_type=model_type,
                model_name=model_type,
                model_source=model_source,
                include_explanations=include_explanations
            )

            # Clean up
            upload_file.file.close()
            os.unlink(tmp_file.name)

            return result

    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None


def analyze_clinical_data(clinical_data: Dict, model_type: str, model_source: str, include_explanations: bool):
    """Analyze clinical data using the API."""
    try:
        # Convert clinical data to CSV format
        df = pd.DataFrame([clinical_data])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')

        # Create a temporary file for the API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(csv_data)
            tmp_file.flush()

            # Create UploadFile object
            from fastapi import UploadFile
            from fastapi.datastructures import Headers

            upload_file = UploadFile(
                filename="clinical_data.csv",
                file=open(tmp_file.name, 'rb'),
                headers=Headers({'content-type': 'text/csv'})
            )

            # Call the API endpoint
            result = predict_endpoint(
                file=upload_file,
                model_type=model_type,
                model_name=model_type,
                model_source=model_source,
                include_explanations=include_explanations
            )

            # Clean up
            upload_file.file.close()
            os.unlink(tmp_file.name)

            return result

    except Exception as e:
        st.error(f"Error analyzing clinical data: {str(e)}")
        return None


def display_results(result: Dict, data_type: str):
    """Display prediction results."""
    if not result.get("success", False):
        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
        return

    # Main prediction results
    st.markdown("### 🎯 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        prediction = result.get("prediction", {})
        pred_class = prediction.get("prediction", 0)
        confidence = prediction.get("confidence", 0.0)

        # Display prediction
        if pred_class == 1:
            st.markdown("""
            <div class="error-box">
                <h3>⚠️ MALIGNANT</h3>
                <p><strong>Confidence:</strong> {:.1f}%</p>
            </div>
            """.format(confidence * 100), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <h3>✅ BENIGN</h3>
                <p><strong>Confidence:</strong> {:.1f}%</p>
            </div>
            """.format(confidence * 100), unsafe_allow_html=True)

    with col2:
        st.metric("Model Used", result.get("model_used", "Unknown"))
        if "probabilities" in prediction:
            probs = prediction["probabilities"]
            st.metric("Benign Probability", f"{probs[0]:.3f}")
            st.metric("Malignant Probability", f"{probs[1]:.3f}")

    with col3:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Confidence"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)

    # Display explanations if available
    if result.get("explanations") and result["explanations"].get("methods"):
        st.markdown("### 🧠 Model Explanations")
        display_explanations(result["explanations"], data_type)


def display_explanations(explanations: Dict, data_type: str):
    """Display XAI explanations."""
    methods = explanations.get("methods", {})

    if not methods:
        st.info("No explanations available.")
        return

    # Create tabs for different explanation methods
    method_names = list(methods.keys())
    if len(method_names) == 1:
        display_single_explanation(method_names[0], methods[method_names[0]], data_type)
    else:
        tabs = st.tabs(method_names)
        for i, method_name in enumerate(method_names):
            with tabs[i]:
                display_single_explanation(method_name, methods[method_name], data_type)


def display_single_explanation(method_name: str, method_data: Dict, data_type: str):
    """Display a single explanation method."""
    st.markdown(f"**{method_name} Analysis**")

    if data_type == "tabular" and "feature_importance" in method_data:
        # Display feature importance for tabular data
        importance = method_data["feature_importance"]
        feature_names = method_data.get("feature_names", [f"Feature_{i}" for i in range(len(importance))])

        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", key=abs, ascending=False)

        # Create bar chart
        fig = px.bar(
            df.head(10),  # Show top 10 features
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"{method_name} Feature Importance",
            color="Importance",
            color_continuous_scale="RdYlBu"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Show feature values if available
        if "feature_values" in method_data:
            st.markdown("**Feature Values:**")
            values_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": method_data["feature_values"],
                "Importance": importance
            }).sort_values("Importance", key=abs, ascending=False)
            st.dataframe(values_df, use_container_width=True)

    elif "description" in method_data:
        st.write(method_data["description"])

    else:
        st.info(f"Explanation data available but visualization not implemented for this method.")


def render_about():
    """Render about/information tab."""
    st.markdown('<h2 class="sub-header">About This System</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Purpose
    This system is designed to assist in the early detection of skin cancer using machine learning
    and deep learning techniques. It analyzes both dermatoscopic images and clinical features.

    ### 🤖 Available Models

    **Image Analysis Models:**
    - **CNN**: Convolutional Neural Network for image classification
    - **ResNet18**: Deep residual network with 18 layers

    **Clinical Data Models:**
    - **XGBoost**: Gradient boosting framework
    - **LightGBM**: Light gradient boosting machine
    - **Random Forest**: Ensemble of decision trees
    - **Logistic Regression**: Linear classification model

    ### 🔍 Explainable AI (XAI)
    The system provides explanations for its predictions using:
    - **SHAP**: SHapley Additive exPlanations
    - **LIME**: Local Interpretable Model-agnostic Explanations
    - **Feature Importance**: Model-specific feature rankings
    - **Permutation Importance**: Feature importance via permutation

    ### 📊 Input Data
    - **Images**: Dermatoscopic images (PNG, JPG, JPEG)
    - **Clinical Features**: Age, sex, location, diameter, asymmetry, border, color variation

    ### 🏥 Clinical Features Description
    - **Age**: Patient's age in years
    - **Sex**: Male or Female
    - **Localization**: Anatomical location of the lesion
    - **Diameter**: Maximum diameter of the lesion in mm
    - **Asymmetry**: Asymmetry score (0-1, higher = more asymmetric)
    - **Border**: Border irregularity score (0-1, higher = more irregular)
    - **Color**: Color variation score (0-1, higher = more varied)
    """)


def render_disclaimer():
    """Render medical disclaimer."""
    st.markdown("""
    ---
    ### ⚠️ Important Medical Disclaimer

    **This application is for research and educational purposes only.**

    - 🩺 **Not a Medical Device**: This tool is not approved as a medical device
    - 👨‍⚕️ **Consult Professionals**: Always consult qualified healthcare professionals for medical concerns
    - 🎯 **Not 100% Accurate**: Machine learning models are not perfect and can make errors
    - 🔬 **Research Tool**: This is a research prototype, not a clinical diagnostic tool
    - 🚨 **Seek Medical Attention**: If you have concerns about a skin lesion, please see a dermatologist immediately

    **The predictions should never be used as the sole basis for medical decisions.**
    """)


if __name__ == "__main__":
    main()