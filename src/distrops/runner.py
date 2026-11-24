from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import RunConfig
import ray
from .utils import load_config_file, resolve_config_targets
import os
import argparse
from .train_worker import  train_model  
from ray.tune.logger import TBXLoggerCallback
from typing import Any, Dict
import copy
from distrops.registry import registry
from distrops.utils import get_expected_arguments
from distrops.plugin import load_plugins
from distrops.validator import validate_pipeline_dag
from distrops.engine import PipelineOrchestrator




def parse_args():
    parser = argparse.ArgumentParser(description="Distributed training and Hyperparameter Optimization pipeline tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")
    
    train_parser = subparsers.add_parser("train", help="Train a model with the provided configuration")
    train_parser.add_argument("--config_path", type=str, required=True, help='Path to the YAML configuration file')
    
    hpo_parser = subparsers.add_parser("hpo", help="Perform distributed hyperparameter sweep")
    hpo_parser.add_argument("--config_path", type=str, required=True, help='Path to the YAML configuration file')
    
    return parser.parse_args()
    

def train_hyperparameter_search(config_path: str):
    raw_config = load_config_file(config_path=config_path)
    infra_config = raw_config.get("infrastructure", {})
    hpo_config = raw_config.get("hpo_bounds", {})
    
    load_plugins(raw_config.get("plugins", []))
    #sample_config = _extract_sample_config_for_validation(copy.deepcopy(raw_config))
    validate_pipeline_dag(raw_config)
    
    #param_search_space = resolve_search_spaces(copy.deepcopy(raw_config))
    param_search_space = extract_flat_search_space(copy.deepcopy(raw_config))
    if not param_search_space:
        raise ValueError(
            "No '_search_space_: true' blocks found in configuration for HPO sweep."
        )
        
    sampled_values = {
        k: v.sample() if hasattr(v, "sample") else v 
        for k, v in param_search_space.items()
    }
    sampled_config = _apply_sampled_params(raw_config, sampled_values) #_sample_resolved_search_spaces(copy.deepcopy(param_search_space))
    validate_pipeline_dag(sampled_config)
    
    #_parse_tune_search_space(raw_config)
    
    storage_path = infra_config.get("storage_path", "./raw-results")
    os.makedirs(storage_path, exist_ok=True)
    run_config_name = infra_config.get("experiment_name", "distrops_dag_hpo")
    
    _init_ray_cluster(infra_config)
    
    metric = hpo_config.get("metric", "eval_loss")
    mode = hpo_config.get("mode", "min")
    
    search_alg_cfg = hpo_config.get("search_alg")
    search_alg = resolve_config_targets(search_alg_cfg) if search_alg_cfg else None
    
    scheduler_cfg = hpo_config.get("scheduler")
    scheduler = resolve_config_targets(scheduler_cfg) if scheduler_cfg else None
    
    # trial_resources = {"cpu": infra_config.get("trial_cpu", 1),
    #                    "gpu": infra_config.get("trial_gpu", 0.0)
    #                    }
    
    workflows = raw_config.get("workflows", [])
    max_stage_cpu = max((step.get("resources", {}).get("cpu", 1) for step in workflows),
                        default=1
                        )
    max_stage_gpu = max((step.get("resources", {}).get("gpu", 0.0) for step in workflows),
                        default=0.0
                        )
    
    coordinator_bundle = {"CPU": infra_config.get("trial_cpu", 1),
                            "GPU": infra_config.get("trial_gpu", 0.0)
                        }
    stage_worker_bundle = {"CPU": max_stage_cpu, "GPU": max_stage_gpu}
    trial_placement_group = tune.PlacementGroupFactory([coordinator_bundle, stage_worker_bundle])
    
    trainable_with_config = tune.with_parameters(_run_hpo_trial, base_config=raw_config)
    
    tuner = tune.Tuner(trainable=tune.with_resources(trainable=trainable_with_config, 
                                                     resources=trial_placement_group
                                                     ),
                       tune_config=tune.TuneConfig(metric=metric,
                                                   mode=mode,
                                                   scheduler=scheduler,
                                                   num_samples=hpo_config.get("num_samples", 10),
                                                   search_alg=search_alg
                                                   ),
                       run_config=RunConfig(name=run_config_name,
                                            storage_path=storage_path,
                                            callbacks=[TBXLoggerCallback()],
                                            ),
                       param_space=param_search_space
        
                        )
    
    print("Launching Distributed DAG Hyperparameter Sweep...")
    results = tuner.fit()
    
    best_result = results.get_best_result(metric=metric, mode=mode)
    print(f"Best {metric}: {best_result.metrics.get(metric)}")
    print(f"Best Metrics Summary: {best_result.metrics}")
    return results, best_result
    
    
    
    
    
    
    
def _parse_tune_search_space(node: Any):
    """
    Recursively walks the YAML config and converts any {'_tune_': ...} block
    into the corresponding ray.tune search space primitive.
    """
    if isinstance(node, dict):
        if "_tune_" in node:
            dist_type = node["_tune_"]
            if dist_type == "choice":
                return tune.choice(node["values"])
            elif dist_type == "uniform":
                return tune.uniform(float(node["min"]), float(node["max"]))
            elif dist_type == "loguniform":
                return tune.loguniform(float(node["min"]), float(node["max"]))
            elif dist_type == "randint":
                return tune.randint(int(node["min"]), int(node["max"]))
            elif dist_type == "quniform":
                return tune.quniform(float(node["min"]), float(node["max"]))
            elif dist_type == "grid_search":
                return tune.grid_search(node["values"])
            else:
                raise ValueError(f"Unsupported _tune_ distribution type: '{dist_type}'")
        return {k: _parse_tune_search_space(v) for k, v in node.items()}
    
    if isinstance(node, list):
        return [_parse_tune_search_space(item) for item in node]
    

    
def _extract_sample_config_for_validation(node: Any):
    """
    Replaces {'_tune_': ...} blocks with a concrete sample value so dry-run
    DAG validation can verify all signatures upfront on the driver.
    """
    if isinstance(node, dict):
        if "_tune_" in node:
            dist_type = node["_tune_"]
            if dist_type == "choice":
                return node["values"][0]
            return node["min"]
        return {k: _extract_sample_config_for_validation(v) for k,v in node.items()}
    
    if isinstance(node, list):
        return [_extract_sample_config_for_validation(item) for item in node]
    
    return node


def _run_hpo_trial(sampled_params: Dict[str, Any], base_config: Dict[str, Any]) -> None:
    """
    Ray Tune trainable wrapper. Executes the entire multi-stage DAG for one
    sampled hyperparameter configuration and reports final metrics back to Tune.
    """
    trial_config = _apply_sampled_params(base_config, sampled_params)
    orchestrator = PipelineOrchestrator(trial_config)
    final_metrics = orchestrator.run()
    tune.report(final_metrics)
            
            
def _init_ray_cluster(infra_config) -> None:
    working_dir = infra_config.get("working_dir")
    temp_dir = infra_config.get("temp_dir")
    spill_dir = infra_config.get("directory_path")
    
    ray.init(ignore_reinit_error=True,
             _temp_dir=temp_dir,
             object_spilling_directory=spill_dir,
             runtime_env={"working_dir": working_dir} if working_dir else None
             )
    


def run_single_train(config_path: str):
    config = load_config_file(config_path=config_path)   
    infra_config = config.get("infrastructure", {})
    
    _init_ray_cluster(infra_config)
    orchestrator = PipelineOrchestrator(config)
    final_metrics = orchestrator.run()
    
    print(f"Pipeline execution complete. Final metrics: {final_metrics}")
    return final_metrics
     

def resolve_search_spaces(node: Any, registry_name: str = "components"):
    """
    Runs on the Driver before tuner.fit().
    Instantiates any '_target_' block marked with '_search_space_: true'
    (or registered under 'distributions'), leaving runtime '_target_' blocks
    (models, optimizers, losses) untouched for the Ray workers.
    """
    if isinstance(node, dict):
        resolved = {k: resolve_search_spaces(v, registry_name=k) for k,v in node.items()}
        
        if "_target_" in resolved and resolved.get("_search_space_", False):
            target_name = resolved.pop("_target_")
            resolved.pop("_search_space_", None)
            registry_name = resolved.pop("_registry_", registry_name)
            
            dist_fn = registry.get(name_or_path=target_name, registry_name=registry_name)
            args, kwargs = get_expected_arguments(dist_fn, resolved)
            return dist_fn(*args, **kwargs)
        return resolved
    if isinstance(node, list):
        return [resolve_search_spaces(item) for item in node]
    return node


def _sample_resolved_search_spaces(node: Any) -> Any:
    """
    Walks a config tree where '_search_space_: true' blocks have been instantiated
    into Ray Tune Domain objects, calling '.sample()' to produce concrete values for
    upfront dry-run DAG validation.
    """
    if hasattr(node, "sample") and callable(node.sample):
        return node.sample()
    
    if isinstance(node, dict):
        if list(node.keys()) == ["grid_search"] and isinstance(node["grid_search"], list):
            return node["grid_search"]
        return {k: _sample_resolved_search_spaces(v) for k,v in node.items()}
    
    if isinstance(node, list):
        return [_sample_resolved_search_spaces(item) for item in node]
    return node

    
    
 
def extract_flat_search_space(node: Any, prefix: str = "", registry_name = None) -> dict:
    """
    Walks the config tree, instantiates any '_target_' + '_search_space_: true' block,
    and returns a flat dictionary keyed by dot-separated string paths (e.g., 'workflows.0.params.lr')
    so search algorithms like OptunaSearch always receive string-only path keys.
    """
    flat_space = {}
    #print(f"node: {node}")
   
    if isinstance(node, dict):
        if "_target_" in node and bool(node.get("_search_space_", False)):
            cfg_copy = dict(node)
            target_name = cfg_copy.pop("_target_")
            cfg_copy.pop("_search_space_", None)
            registry_name = cfg_copy.pop("_registry_", registry_name)
            
            dist_fn = registry.get(name_or_path=target_name, registry_name=registry_name)
            args, kwargs = get_expected_arguments(dist_fn, cfg_copy)
            flat_space[prefix] = dist_fn(*args, **kwargs)
            return flat_space
        
        for k, v in node.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            flat_space.update(
                extract_flat_search_space(v, prefix=child_prefix, registry_name=str(k))
            )
            
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            flat_space.update(
                extract_flat_search_space(item, prefix=child_prefix, registry_name=registry_name)
            )
    return flat_space
            
            
    

def _apply_sampled_params(base_config: dict, sampled_params: dict):
    """Injects flat dot-path sampled values back into a deep copy of base_config."""
    merged = copy.deepcopy(base_config)
    for dot_path, value in sampled_params.items():
        parts = dot_path.split(".")
        curr = merged
        for part in parts[:-1]:
            curr = curr[int(part)] if isinstance(curr, list) else curr[part]
        last = parts[-1]
        if isinstance(curr, list):
            curr[int(last)] = value
        else:
            curr[last] = value
    return merged
    
    
    
    
    
    
def main():
    args = parse_args()
    if args.command == "train":
        print(f"Running single train pipeline with config: {args.config_path}")
        run_single_train(config_path=args.config_path)
    elif args.command == "hpo":
        print(f"Start Hyperparameter Sweep with {args.config_path}")
        train_hyperparameter_search(config_path=args.config_path)
    
if __name__ == "__main__":
    res = main()



