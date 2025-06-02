import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from src.skin_cancer_detection.pipelines.train_model.nodes import train_model, prepare_callbacks
import os


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
    checkpoints_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
        'data',
        '06_models',
        'checkpoints'
    )

    class DummyCallback(Callback):
        pass

    return [DummyCallback(), ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-model",
        save_top_k=0
    )]


@pytest.fixture
def dummy_params():
    return {
        "learning_rate": 0.01,
        "epochs": 1,
        "accelerator": "auto"
    }


@pytest.fixture
def dummy_prep_callbacks_params():
    return {
        'es_patience': 1,
        'ckpt_save_top_k': 0
    }


def test_prepare_callbacks(dummy_prep_callbacks_params):
    callbacks = prepare_callbacks(dummy_prep_callbacks_params)
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], EarlyStopping)
    assert isinstance(callbacks[1], ModelCheckpoint)


def test_train_model_node(dummy_model, dummy_dataloaders, dummy_callbacks, dummy_params):
    # jak chcemy żeby ten test wywoływał nodea prepare_callbacks, to będziemy do tej funkcji
    # podać dummy_prep_callbacks_params, zamiast dummy callbacks, czy to będzie lepsze?
    # jak powinno być
    train_loader, val_loader = dummy_dataloaders
    trained_model = train_model(
        model=dummy_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=dummy_callbacks,
        params=dummy_params
    )
