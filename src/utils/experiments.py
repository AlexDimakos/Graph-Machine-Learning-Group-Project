from contextlib import nullcontext

import mlflow

import src.config as config


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFlowConfig.tracking_uri)
    mlflow.set_experiment(config.MLFlowConfig.experiment_name)


def start_run(run_name=None):
    if config.USE_MLFLOW:
        setup_mlflow()
        return mlflow.start_run(run_name=run_name)
    else:
        return nullcontext()
