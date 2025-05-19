"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline
from src.skin_cancer_detection.pipelines.train_model.nodes import (
    prepare_callbacks,
    prepare_data_loaders,
    train_model,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_data_loaders,
            inputs="params:train_hparams",
            outputs=['train_loader', 'val_loader'],
            name='node_prepare_data_loaders'
        ),
        node(
            func=prepare_callbacks,
            inputs='params:callback_params',
            outputs='callbacks',
            name='node_prepare_callbacks'
        ),
        node(
            func=train_model,
            inputs=['model', 'train_loader', 'val_loader', 'callbacks', 'params:train_hparams'],
            outputs='trained_model',
            name='node_train_model'
        )
    ])
