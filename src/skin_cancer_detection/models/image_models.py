"""
Image models for skin cancer detection using PyTorch and PyTorch Lightning.

This module implements CNN and ResNet architectures with comprehensive training,
validation, and testing capabilities, integrated with wandb for experiment tracking.
"""

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import wandb
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageModelConfig:
    """Configuration for image models."""

    # Model architecture
    model_type: str = "resnet18"  # "cnn", "resnet18"
    num_classes: int = 2
    input_channels: int = 3
    dropout_rate: float = 0.5

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    weight_decay: float = 1e-4

    # Optimizer settings
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"

    # Loss function
    loss_function: str = "bce"  # "bce", "focal", "cross_entropy"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Data augmentation
    use_augmentation: bool = True
    image_size: Tuple[int, int] = (224, 224)

    # Class imbalance handling
    use_class_weights: bool = True
    pos_weight: Optional[float] = 11.27  # Default based on data imbalance ratio

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Checkpoint settings
    save_top_k: int = 3
    monitor_metric: str = "val_f1"

    # wandb settings
    project_name: str = "skin-cancer-detection"
    experiment_name: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["image_classification", self.model_type]


class CNNClassifier(pl.LightningModule):
    """Custom CNN classifier for skin cancer detection."""

    def __init__(self, config: ImageModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Build CNN architecture
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Fifth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes if config.num_classes > 2 else 1)
        )

        # Loss function
        self.criterion = self._get_loss_function()

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _get_loss_function(self):
        """Get the appropriate loss function."""
        if self.config.loss_function == "bce":
            pos_weight = None
            if self.config.pos_weight is not None:
                pos_weight = torch.tensor(self.config.pos_weight)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.config.loss_function == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif self.config.loss_function == "focal":
            return FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.config.num_classes == 2:
            y = y.float().unsqueeze(1)

        loss = self.criterion(y_hat, y)

        # Calculate metrics
        if self.config.num_classes == 2:
            preds = torch.sigmoid(y_hat)
            binary_preds = (preds > 0.5).float()
            acc = (binary_preds == y).float().mean()
        else:
            preds = F.softmax(y_hat, dim=1)
            binary_preds = torch.argmax(preds, dim=1)
            acc = (binary_preds == y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        output = {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach()
        }
        self.training_step_outputs.append(output)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.config.num_classes == 2:
            y = y.float().unsqueeze(1)

        loss = self.criterion(y_hat, y)

        # Calculate metrics
        if self.config.num_classes == 2:
            preds = torch.sigmoid(y_hat)
            binary_preds = (preds > 0.5).float()
            acc = (binary_preds == y).float().mean()
        else:
            preds = F.softmax(y_hat, dim=1)
            binary_preds = torch.argmax(preds, dim=1)
            acc = (binary_preds == y).float().mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        output = {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach()
        }
        self.validation_step_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.config.num_classes == 2:
            y = y.float().unsqueeze(1)

        loss = self.criterion(y_hat, y)

        # Calculate metrics
        if self.config.num_classes == 2:
            preds = torch.sigmoid(y_hat)
            binary_preds = (preds > 0.5).float()
            acc = (binary_preds == y).float().mean()
        else:
            preds = F.softmax(y_hat, dim=1)
            binary_preds = torch.argmax(preds, dim=1)
            acc = (binary_preds == y).float().mean()

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        output = {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach()
        }
        self.test_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        self._compute_epoch_metrics(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._compute_epoch_metrics(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self._compute_epoch_metrics(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def _compute_epoch_metrics(self, outputs, stage):
        """Compute comprehensive metrics at the end of each epoch."""
        if not outputs:
            return

        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])

        # Convert to numpy for sklearn metrics (fix BFloat16 issue)
        if self.config.num_classes == 2:
            preds_np = all_preds.float().cpu().numpy().flatten()
            targets_np = all_targets.float().cpu().numpy().flatten()
            binary_preds_np = (preds_np > 0.5).astype(int)
        else:
            preds_np = all_preds.float().cpu().numpy()
            targets_np = all_targets.float().cpu().numpy()
            binary_preds_np = np.argmax(preds_np, axis=1)

        # Calculate comprehensive metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_np, binary_preds_np, average='binary' if self.config.num_classes == 2 else 'weighted'
        )

        # F2 score (higher weight on recall)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-8)

        # ROC AUC
        try:
            if self.config.num_classes == 2:
                roc_auc = roc_auc_score(targets_np, preds_np)
            else:
                roc_auc = roc_auc_score(targets_np, preds_np, multi_class='ovr')
        except ValueError:
            roc_auc = 0.0

        # Log metrics
        self.log(f'{stage}_precision', precision, on_epoch=True)
        self.log(f'{stage}_recall', recall, on_epoch=True)
        self.log(f'{stage}_f1', f1, on_epoch=True)
        self.log(f'{stage}_f2', f2, on_epoch=True)
        self.log(f'{stage}_roc_auc', roc_auc, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.config.optimizer == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Scheduler
        if self.config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs
            )
        elif self.config.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" if self.config.scheduler == "plateau" else None,
            }
        }


class ResNet18Classifier(pl.LightningModule):
    """ResNet18 classifier for skin cancer detection."""

    def __init__(self, config: ImageModelConfig, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Modify the first layer if input channels != 3
        if config.input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                config.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace the final layer
        num_features = self.backbone.fc.in_features
        if config.num_classes > 2:
            self.backbone.fc = nn.Linear(num_features, config.num_classes)
        else:
            self.backbone.fc = nn.Linear(num_features, 1)

        # Add dropout
        if config.dropout_rate > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                self.backbone.fc
            )

        # Loss function
        self.criterion = self._get_loss_function()

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _get_loss_function(self):
        """Get the appropriate loss function."""
        if self.config.loss_function == "bce":
            pos_weight = None
            if self.config.pos_weight is not None:
                pos_weight = torch.tensor(self.config.pos_weight)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.config.loss_function == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif self.config.loss_function == "focal":
            return FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.config.num_classes == 2:
            y = y.float().unsqueeze(1)

        loss = self.criterion(y_hat, y)

        # Calculate metrics
        if self.config.num_classes == 2:
            preds = torch.sigmoid(y_hat)
            binary_preds = (preds > 0.5).float()
            acc = (binary_preds == y).float().mean()
        else:
            preds = F.softmax(y_hat, dim=1)
            binary_preds = torch.argmax(preds, dim=1)
            acc = (binary_preds == y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        output = {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach()
        }
        self.training_step_outputs.append(output)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.config.num_classes == 2:
            y = y.float().unsqueeze(1)

        loss = self.criterion(y_hat, y)

        # Calculate metrics
        if self.config.num_classes == 2:
            preds = torch.sigmoid(y_hat)
            binary_preds = (preds > 0.5).float()
            acc = (binary_preds == y).float().mean()
        else:
            preds = F.softmax(y_hat, dim=1)
            binary_preds = torch.argmax(preds, dim=1)
            acc = (binary_preds == y).float().mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        output = {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach()
        }
        self.validation_step_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.config.num_classes == 2:
            y = y.float().unsqueeze(1)

        loss = self.criterion(y_hat, y)

        # Calculate metrics
        if self.config.num_classes == 2:
            preds = torch.sigmoid(y_hat)
            binary_preds = (preds > 0.5).float()
            acc = (binary_preds == y).float().mean()
        else:
            preds = F.softmax(y_hat, dim=1)
            binary_preds = torch.argmax(preds, dim=1)
            acc = (binary_preds == y).float().mean()

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        output = {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach()
        }
        self.test_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        self._compute_epoch_metrics(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._compute_epoch_metrics(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self._compute_epoch_metrics(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def _compute_epoch_metrics(self, outputs, stage):
        """Compute comprehensive metrics at the end of each epoch."""
        if not outputs:
            return

        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])

        # Convert to numpy for sklearn metrics (fix BFloat16 issue)
        if self.config.num_classes == 2:
            preds_np = all_preds.float().cpu().numpy().flatten()
            targets_np = all_targets.float().cpu().numpy().flatten()
            binary_preds_np = (preds_np > 0.5).astype(int)
        else:
            preds_np = all_preds.float().cpu().numpy()
            targets_np = all_targets.float().cpu().numpy()
            binary_preds_np = np.argmax(preds_np, axis=1)

        # Calculate comprehensive metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_np, binary_preds_np, average='binary' if self.config.num_classes == 2 else 'weighted'
        )

        # F2 score (higher weight on recall)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-8)

        # ROC AUC
        try:
            if self.config.num_classes == 2:
                roc_auc = roc_auc_score(targets_np, preds_np)
            else:
                roc_auc = roc_auc_score(targets_np, preds_np, multi_class='ovr')
        except ValueError:
            roc_auc = 0.0

        # Log metrics
        self.log(f'{stage}_precision', precision, on_epoch=True)
        self.log(f'{stage}_recall', recall, on_epoch=True)
        self.log(f'{stage}_f1', f1, on_epoch=True)
        self.log(f'{stage}_f2', f2, on_epoch=True)
        self.log(f'{stage}_roc_auc', roc_auc, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.config.optimizer == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Scheduler
        if self.config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs
            )
        elif self.config.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" if self.config.scheduler == "plateau" else None,
            }
        }


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ImageModelTrainer:
    """Trainer class for image models with comprehensive training pipeline."""

    def __init__(self, config: ImageModelConfig):
        self.config = config
        self.wandb_logger = None

    def setup_wandb(self):
        """Setup wandb logging."""
        self.wandb_logger = WandbLogger(
            project=self.config.project_name,
            name=self.config.experiment_name,
            tags=self.config.tags,
            log_model=True
        )

        # Log configuration - allow value changes to prevent conflicts during hyperparameter optimization
        self.wandb_logger.experiment.config.update(self.config.__dict__, allow_val_change=True)

    def get_model(self, pretrained: bool = True) -> pl.LightningModule:
        """Get the appropriate model based on configuration."""
        if self.config.model_type == "cnn":
            return CNNClassifier(self.config)
        elif self.config.model_type == "resnet18":
            return ResNet18Classifier(self.config, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def get_callbacks(self):
        """Get training callbacks."""
        callbacks = []

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{self.config.experiment_name or 'default'}",
            filename="{epoch}-{val_f1:.3f}",
            monitor=self.config.monitor_metric,
            mode="max",
            save_top_k=self.config.save_top_k,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.monitor_metric,
            mode="max",
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        callbacks.append(early_stopping)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

        return callbacks

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              test_dataloader: Optional[DataLoader] = None,
              pretrained: bool = True) -> pl.LightningModule:
        """
        Train the image model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Optional test data loader
            pretrained: Whether to use pretrained weights

        Returns:
            Trained model
        """
        # Setup wandb
        self.setup_wandb()

        # Get model
        model = self.get_model(pretrained=pretrained)

        # Get callbacks
        callbacks = self.get_callbacks()

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            logger=self.wandb_logger,
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            strategy="auto",
            precision=32,  # Use float32 to avoid BFloat16 issues on CPU
            gradient_clip_val=1.0,
            log_every_n_steps=10
        )

        # Train model
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        # Test model if test dataloader provided
        if test_dataloader is not None:
            trainer.test(model, dataloaders=test_dataloader)

        # Save model to wandb
        if self.wandb_logger:
            model_artifact = wandb.Artifact(
                name=f"{self.config.experiment_name or 'model'}",
                type="model",
                description=f"Trained {self.config.model_type} model"
            )

            # Save checkpoint
            checkpoint_path = trainer.checkpoint_callback.best_model_path
            if checkpoint_path:
                model_artifact.add_file(checkpoint_path)
                self.wandb_logger.experiment.log_artifact(model_artifact)

        logger.info("Training completed successfully!")
        return model

    def get_data_transforms(self, stage: str = "train"):
        """Get data transforms for different stages."""
        if stage == "train" and self.config.use_augmentation:
            return transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.config.image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


def create_image_model_config(**kwargs) -> ImageModelConfig:
    """Create image model configuration with custom parameters."""
    return ImageModelConfig(**kwargs)