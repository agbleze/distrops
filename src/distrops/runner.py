from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import RunConfig
import ray
from utils import load_config_file
import os
import argparse
from train_worker import  train_model  
from ray.tune.logger import TBXLoggerCallback

def parse_args():
    parser = argparse.ArgumentParser(description="HPO with ray")
    parser.add_argument("--config_path", type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()
    

def main():
    args = parse_args()
    config_path = args.config_path
    config = load_config_file(config_path=config_path)
    infra_config = config.get("infrastructure", {})
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    storage_path = infra_config.get("storage_path") 
    hpo_config = config.get("hpo_bounds")
    temp_dir = infra_config.get("temp_dir")
    directory_path = infra_config.get("directory_path")
    
    optuna_search = OptunaSearch(metric="loss", mode="min")
    
    batche_size = hpo_config.get("batch_size")
    optimizer = hpo_config.get("optimizer")
    schedule_type = hpo_config.get("schedulers")
    learning_rate = hpo_config.get("learning_rate")
    weight_decay = hpo_config.get("weight_decay")
    num_samples = hpo_config.get("num_samples")
    grace_period = hpo_config.get("grace_period", 1)
    reduction_factor = hpo_config.get("reduction_factor", 2)

    param_search_space = {"batch_size": tune.choice(batche_size),
                            "learning_rate": tune.loguniform(float(learning_rate[0]), float(learning_rate[1])),
                            "lr_scheduler_type": tune.choice(categories=schedule_type),
                            "optimizer": tune.choice(categories=optimizer),
                            "weight_decay": tune.uniform(lower=weight_decay[0], upper=weight_decay[1]),
                            **dataset_config,
                            **model_config,
                            **infra_config,
                            }
    
    os.makedirs(storage_path, exist_ok=True)
    
    run_config_name=f"{dataset_config.get('dataset_name', 'default')}_tune_demo"
    
    ray.init(ignore_reinit_error=True,
             _temp_dir=temp_dir,
             object_spilling_directory=directory_path,
            )
    
    print("Configuring scheduler")
    scheduler = ASHAScheduler(time_attr="training_iteration",
                              max_t=model_config.get("epochs"),
                              grace_period=grace_period,
                              reduction_factor=reduction_factor
                              )
    
    print("Configuring Tuner")
    tuner = tune.Tuner(trainable=tune.with_resources(
                                            trainable=train_model,
                                            resources={"cpu": infra_config.get("cpu"),
                                                        "gpu": infra_config.get("gpu")
                                                        }
                                        ),
                       tune_config=tune.TuneConfig(metric=hpo_config.get("metric"),
                                                   mode=hpo_config.get("mode"),
                                                   scheduler=scheduler,
                                                   num_samples=num_samples,
                                                   search_alg=optuna_search,
                                                   ),
                       run_config=RunConfig(name=run_config_name,
                                            storage_path=storage_path,
                                            callbacks=[TBXLoggerCallback()]
                                            ),
                       param_space=param_search_space,
                    )
    
    print(f"fitting tuner")
    results = tuner.fit()
    print(f"successfully fitted tuner")
    best_result = results.get_best_result(metric=hpo_config.get("metric"), mode=hpo_config.get("mode"))
    print(f"Best Validation loss: {best_result.metrics[hpo_config.get('metric')]}")
    print(f"Best validation accuracy: {best_result.metrics['eval_accuracy']}")
    return results, best_result

if __name__ == "__main__":
    res = main()
