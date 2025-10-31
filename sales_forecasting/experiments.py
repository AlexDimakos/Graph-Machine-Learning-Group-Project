import random
from dataclasses import asdict, replace

import mlflow

import sales_forecasting.config as config
import sales_forecasting.models.train as train_module
from sales_forecasting.utils.experiments import start_run


# TODO: maybe this should take seed as input so that it's reproducible
def random_search(n_trials: int = 10, param_space: dict | None = None):
    """Run a simple random search over TrainingConfig hyperparameters.

    Args:
        n_trials: number of sampled configurations to run
        param_space: optional dict specifying sampling functions or lists for
            keys: 'lr', 'batch_size', 'window_size', 'hidden_size', 'K'

    Returns:
        List of tuples (train_config, results) where results is the return
        value from train_module.run_experiment(train_config)
    """
    if param_space is None:
        param_space = {
            "lr": lambda: 10 ** random.uniform(-3, -1),
            "window_size": lambda: random.choice([3, 4, 6, 8]),
            "hidden_size": lambda: random.choice([4, 8, 12, 16]),
            "K": lambda: random.choice([1, 2]),
            "weight_decay": lambda: 10 ** random.uniform(-5, -3),
        }

    for i in range(n_trials):
        # sample params
        sampled = {}
        for k, sampler in param_space.items():
            sampled[k] = sampler() if callable(sampler) else random.choice(sampler)

        # build a TrainingConfig from defaults and replace fields
        base = config.TrainingConfig()
        tc = replace(
            base,
            lr=sampled.get("lr", base.lr),
            batch_size=sampled.get("batch_size", base.batch_size),
            window_size=sampled.get("window_size", base.window_size),
            hidden_size=sampled.get("hidden_size", base.hidden_size),
            K=sampled.get("K", base.K),
        )

        run_name = f"initial_random_search_{i}"
        with start_run(run_name=run_name):
            mlflow.log_params(asdict(tc))
            print(f"Starting run '{run_name}' with hyperparameters:")
            for k in sorted(sampled):
                print(f"  {k:12}: {sampled[k]}")
            print(
                f"Derived TrainingConfig -> lr={tc.lr}, batch_size={tc.batch_size}, "
                f"window_size={tc.window_size}, hidden_size={tc.hidden_size}, K={tc.K}"
            )
            train_module.run_experiment(tc)


if __name__ == "__main__":
    random_search(n_trials=15)
