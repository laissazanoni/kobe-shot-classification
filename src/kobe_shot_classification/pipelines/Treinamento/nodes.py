"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

import mlflow
from pycaret.classification import *
from sklearn.metrics import log_loss


def train_logist_regression(data, target, n_folds, param_grid):
    """Train the Logist Regression classifier with Pycaret."""

    experiment_name = "kobe_shot_classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True, run_name='logistic_regression'):

        s = setup(data, target=target, session_id=123)

        exp = ClassificationExperiment()
        exp.setup(data, target=target, session_id=123)

        # add new metric
        exp.add_metric('log_loss', 'Log Loss', log_loss, greater_is_better=False)

        # train a logistic regression model with default params
        lr = exp.create_model('lr')

        # tune hyperparameters of lr
        tuned_lr = exp.tune_model(lr, custom_grid=param_grid,
                                  optimize='Log Loss', fold=n_folds)

        # save  model
        exp.save_model(tuned_lr, 'lr_model')
        mlflow.log_artifact('lr_model.pkl')

        # save tags
        mlflow.set_tag('model', lr)
        mlflow.set_tag('algorithm', 'PyCaret')

        # save  parameters
        tuned_lr_params = tuned_lr.get_params()
        mlflow.log_param('lr_best_params', tuned_lr_params)

        predictions = exp.predict_model(tuned_lr)

        # save metrics
        mlflow.log_metric('Acurácia', exp.pull()["Accuracy"])
        mlflow.log_metric('AUC', exp.pull()["AUC"])
        mlflow.log_metric('F1', exp.pull()["F1"])
        mlflow.log_metric('Log Loss', exp.pull()["Log Loss"])
        mlflow.log_metric('Recall', exp.pull()["Recall"])

        return tuned_lr


def train_classifier(data, target, n_folds):
    """Train the Logist Regression classifier with Pycaret."""

    experiment_name = "kobe_shot_classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True, run_name='best_classifier'):

        s = setup(data, target=target, session_id=123)

        exp = ClassificationExperiment()
        exp.setup(data, target=target, session_id=123)

        # add new metric
        exp.add_metric('log_loss', 'Log Loss', log_loss, greater_is_better=False)

       # compare models performance and select model with best performance
        best_model = exp.compare_models()

        # tune model hyperparameters
        tuned_model = exp.tune_model(best_model, optimize='Log Loss', fold=n_folds)

        # save  model
        exp.save_model(tuned_model, 'classifier_model')
        mlflow.log_artifact('classifier_model.pkl')

        # save model tags
        mlflow.set_tag('model', best_model)
        mlflow.set_tag('algorithm', 'PyCaret')

        # save  parameters
        tuned_model_params = tuned_model.get_params()
        mlflow.log_param('lr_best_params', tuned_model_params)

        predictions = exp.predict_model(tuned_model)

        # save metrics
        mlflow.log_metric('Acurácia', exp.pull()["Accuracy"])
        mlflow.log_metric('AUC', exp.pull()["AUC"])
        mlflow.log_metric('F1', exp.pull()["F1"])
        mlflow.log_metric('Log Loss', exp.pull()["Log Loss"])
        mlflow.log_metric('Recall', exp.pull()["Recall"])

        return tuned_model
