import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from kedro.config import OmegaConfigLoader
from src.skin_cancer_detection.pipelines.train_model.nodes import prepare_callbacks, train_model

@pytest.fixture
def config_loader():
    return OmegaConfigLoader(
        conf_source="conf",
        env="test",
        config_patterns={
            "parameters_train_model": ["parameters_train_model.yml"]
        }
    )

@pytest.fixture
def params(config_loader):
    return config_loader["parameters_train_model"]


@pytest.fixture
def dummy_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 1),
        nn.Sigmoid()
    )


@pytest.fixture
def dummy_dataloaders():
    X = torch.rand(10, 1, 28, 28).to(dtype=torch.float32)
    y = torch.randint(0, 2, (10, 1)).to(dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)
    return loader, loader


def test_prepare_callbacks(params):
    callbacks = prepare_callbacks(params["callback_params"])
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], EarlyStopping)
    assert isinstance(callbacks[1], ModelCheckpoint)


def test_train_model_with_config(params, dummy_model, dummy_dataloaders):
    callbacks = prepare_callbacks(params["callback_params"])
    train_loader, val_loader = dummy_dataloaders

    ckpt_callback = train_model(
        model=dummy_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=callbacks,
        params=params["train_hparams"]
    )

    assert ckpt_callback is not None
    assert isinstance(ckpt_callback, ModelCheckpoint)
