from lightning import Callback

from src.skin_cancer_detection.nn_framework.trainer_module import TrainerModule
from src.skin_cancer_detection.nn_framework.dataset import BatchedDataset

from torch.utils.data import DataLoader

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import lightning as L


def prepare_data_loaders(params: dict):
    train_dataset = BatchedDataset(path=params['train_path'],
                                   set_label='train',
                                   dtype=params['dtype'])

    val_dataset = BatchedDataset(path=params['val_path'],
                                 set_label='valid',
                                 dtype=params['dtype'])

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    return train_loader, val_loader


def prepare_callbacks(params: list[dict[str, str | dict]]) -> list[Callback]:
    callback_mapping = {
        "ModelCheckpoint": ModelCheckpoint,
        "EarlyStopping": EarlyStopping,
    }

    callbacks = []
    for callback_config in params:
        name = callback_config["name"]
        if name not in callback_mapping:
            raise ValueError(f"Unknown callback name {name}")

        callback_cls = callback_mapping[name]
        callbacks.append(callback_cls(**callback_config["args"]))

    return callbacks


def train_model(model, train_dataloader, val_dataloader, callbacks, params: dict):
    ckpt_callback = next((c for c in callbacks if isinstance(c, ModelCheckpoint)))
    module = TrainerModule(model, learning_rate=params['learning_rate'])
    trainer = L.Trainer(
        max_epochs=params['epochs'],
        callbacks=callbacks,
        accelerator=params['accelerator'],
        precision=params['precision']
    )
    trainer.fit(module, train_dataloader, val_dataloader)
    return ckpt_callback.best_model_path


def load_trained_module(best_model_ckpt_path: str):
    module = TrainerModule.load_from_checkpoint(checkpoint_path=best_model_ckpt_path)
    return module
