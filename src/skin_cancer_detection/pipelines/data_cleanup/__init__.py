"""
Data cleanup pipeline for removing invalid images and patches.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]