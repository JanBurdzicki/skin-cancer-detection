"""
Simple Streamlit web application module for skin cancer detection.

This module provides a user-friendly web interface for interacting with
the skin cancer detection models, including image upload, data input,
predictions, and explanations.
"""

from .main import main as run_app

__all__ = [
    "run_app"
]