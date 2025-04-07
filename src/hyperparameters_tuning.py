import os
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from src.train import train_model
from utils.generic import save_config

def hyperparameter_tuning(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    config = {
        "model_name": tune.choice(["fasterrcnn", "retinanet"]),
        "backbone_name": tune.choice(["resnet50", "mobilenet_v3_large"]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "optimizer": tune.choice(["adam", "sgd"]),
        "train_backbone": tune.choice([True, False]),
        "epochs": tune.choice([5, 10, 15]),
        "models_dir": os.path.join(os.getcwd(), 'models')
    }

    scheduler = ASHAScheduler(
        metric="map_50",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["map_50", "Total Loss", "training_iteration"]
    )

    search_alg = BayesOptSearch()

    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        storage_path=os.path.join(os.getcwd(), 'ray_results'),
        name="hyperparameter_tuning"
    )

    best_trial = result.get_best_trial("map_50", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation mAP@50: {}".format(
        best_trial.last_result["map_50"]))

    # Save the best config
    best_config = best_trial.config
    save_config(best_config, 'config/best_config.json')

if __name__ == "__main__":
    hyperparameter_tuning()
