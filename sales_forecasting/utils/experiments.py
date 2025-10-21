from contextlib import nullcontext

import mlflow

import sales_forecasting.config as config


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFlowConfig.tracking_uri)
    mlflow.set_experiment(config.MLFlowConfig.experiment_name)


def start_run(run_name=None):
    if config.USE_MLFLOW:
        setup_mlflow()
        run_context = mlflow.start_run(run_name=run_name)
        config_path = config.CONFIG_PATH
        mlflow.log_artifact(
            config_path, artifact_path="config", run_id=run_context.info.run_id
        )
        return run_context
    else:
        return nullcontext()
