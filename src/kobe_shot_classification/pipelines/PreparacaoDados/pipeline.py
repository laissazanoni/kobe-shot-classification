"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_data, split_train_test, data_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_data,
            name='prepare_data',
            inputs=[
                'data_raw',
                'params:shot_type'
            ],
            outputs='data_filtered',
        ),
        node(
            func=split_train_test,
            name='split_train_test',
            inputs=[
                'data_filtered',
                'params:test_size',
                'params:test_split_random_state',
                'params:target'
            ],
            outputs=['data_train', 'data_test'],
        ),
        node(
            func=data_metrics,
            name='data_metrics',
            inputs=[
                'data_train',
                'data_test'],
            outputs='data_metrics',
        )
    ])
