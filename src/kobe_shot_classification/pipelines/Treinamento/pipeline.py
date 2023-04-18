"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_logist_regression, train_classifier


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logist_regression,
            name='train_logist_regression',
            inputs=[
                'data_filtered',
                'params:target',
                'params:n_folds',
                'params:param_grid',
            ],
            outputs='lr_model'
        ),
        node(
            func=train_classifier,
            name='train_classifier',
            inputs=[
                'data_filtered',
                'params:target',
                'params:n_folds'
            ],
            outputs='classifier_model'
        )
    ])
