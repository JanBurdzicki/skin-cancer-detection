"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.12
"""
from src.skin_cancer_detection.nn_framework.trainer_module import TrainerModule
from src.skin_cancer_detection.nn_framework.dataset import BatchedDataset

from torch.utils.data import DataLoader

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import lightning as L

import os


def prepare_data_loaders(params: dict):
    train_dataset = BatchedDataset('train')
    val_dataset = BatchedDataset('valid')

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    return train_loader, val_loader


def prepare_callbacks(params: dict):
    checkpoints_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
        'data',
        '06_models',
        'checkpoints'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=checkpoints_dir,
        filename="best-model",
        save_top_k=1
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=params["es_patience"],
        verbose=True
    )

    return [es, checkpoint_callback]


def train_model(model, train_dataloader, val_dataloader, callbacks, params: dict):
    ckpt_callback: ModelCheckpoint = callbacks[1]
    module = TrainerModule(model, learning_rate=params['learning_rate'])
    trainer = L.Trainer(
        max_epochs=params['epochs'],
        callbacks=callbacks,
        accelerator=params['accelerator']
    )
    trainer.fit(module, train_dataloader, val_dataloader)
    module = TrainerModule.load_from_checkpoint(checkpoint_path=ckpt_callback.best_model_path)
    return module.model


if __name__ == "__main__":
    print(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            'data',
            '06_models',
            'checkpoints'
        )
    )