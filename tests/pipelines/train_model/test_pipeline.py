import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import Callback
from src.skin_cancer_detection.pipelines.train_model.nodes import train_model


@pytest.fixture
def dummy_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 1),
        nn.Sigmoid()
    )


@pytest.fixture
def dummy_dataloaders():
    X = torch.rand(64, 1, 28, 28).to(dtype=torch.float32)
    y = torch.randint(2, (64, 1)).to(dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8)
    return loader, loader  # train_loader, val_loader


@pytest.fixture
def dummy_callbacks():
    class DummyCallback(Callback):
        pass
    return [DummyCallback(), DummyCallback()]


@pytest.fixture
def dummy_params():
    return {
        "learning_rate": 0.01,
        "epochs": 1,
        "device": "cpu"
    }


def test_train_model_node(dummy_model, dummy_dataloaders, dummy_callbacks, dummy_params):
    train_loader, val_loader = dummy_dataloaders
    trained_model = train_model(
        model=dummy_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=dummy_callbacks,
        params=dummy_params
    )

