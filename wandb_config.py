"""
WandB configuration for skin cancer detection project.
Run this script to initialize WandB settings.
"""

import wandb
import os
from pathlib import Path

def setup_wandb():
    """Setup WandB for the skin cancer detection project."""

    # Set environment variables for offline mode if needed
    # os.environ["WANDB_MODE"] = "offline"  # Uncomment for offline mode

    # Initialize wandb
    wandb.login()

    # Set default project
    os.environ["WANDB_PROJECT"] = "skin-cancer-detection"

    print("WandB setup completed!")
    print("Project: skin-cancer-detection")
    print("You can now run the ML pipeline with WandB logging enabled.")

if __name__ == "__main__":
    setup_wandb()