from typing import Dict, Any, List, Callable, TypedDict, get_type_hints
import abc
import torch
import ray
from distrops.utils import get_expected_arguments, resolve_config_targets
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import inspect
from .plugin import load_plugins
import math
from distrops.validator import validate_pipeline_dag
from distrops.registry import registry
import time
import uuid
import hashlib


class DistropsWorkflow(abc.ABC):
    @abc.abstractmethod
    def execute_training_pipeline(self, *args, **kwargs):
        pass
    
    
class DecisionTreeWorkflow:
    def __init__(self, config):
        self.config = config
        
        
        
class ExecutionEngine:
    def __init__(self, config, listeners):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.listeners = listeners or []
        
        data_loader_fn = registry.get(config.get("data_loader"))  
        
        
        


@ray.remote
def _execute_isolated_stage(workflow_name: str, execution_context: dict, registry_name="workflows") -> Dict[str, Any]:
    """
    Executes a single workflow stage inside a dynamically sized ray worker process.
    Loads external plugins and recursively instantiates '_target_' configuration.
    """
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False
    
    NON_TARGET_ROOT_KEYS = {"workflows", "hpo_bounds", "infrastructure", "plugins", "metrics_queue"}
    load_plugins(execution_context.get("plugins", []))
    workflow_cls = registry.get(name_or_path=workflow_name, registry_name=registry_name)
    workflow_instance = workflow_cls() if inspect.isclass(workflow_cls) else workflow_cls
    filtered_context = {
        k: v for k,v in execution_context.items() if k not in NON_TARGET_ROOT_KEYS
    }
    #instantiated_context = resolve_config_targets(execution_context, registry_name=registry_name)
    instantiated_context = resolve_config_targets(filtered_context, registry_name=registry_name)
    
    if "metrics_queue" in execution_context:
        instantiated_context["metrics_queue"] = execution_context["metrics_queue"]
        
    args, kwargs = get_expected_arguments(workflow_instance, instantiated_context)
    output = workflow_instance(*args, **kwargs)
    
    if not isinstance(output, dict):
        raise TypeError(f"Workflow '{workflow_cls.__name__}' must return a dict, got {type(output)}")
    return output


WORKFLOW_REGISTRY = {}

def register_workflow(name: str):
    def decorator(cls):
        WORKFLOW_REGISTRY[name] = cls
        return cls
    return decorator


class DataPrepOutput(TypedDict):
    train_data_path: str
    test_data_path: str
    
    
@ray.remote
class _DatasetPrepCoordinator:
    """Cluster-wide singleton actor that serializes dataset prep across all nodes."""
    
    def ensure_prepared(self, dataset_name: str,
                        pretrained_model_name_or_path: str,
                        output_dir: str, max_length: int,
                        max_samples: int = None,
                        ) -> DataPrepOutput:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.parquet")
        test_path = os.path.join(output_dir, "test.parquet")
        #lock_path = os.path.join(output_dir, ".prep.lock")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"ALREADY EXIST - HuggingFaceDataWorkflow: train_path: {train_path} and test_path: {test_path}")
            return {"train_data_path": train_path, "test_data_path": test_path}
        
        raw_ds = load_dataset(dataset_name)
        if max_samples is not None:
            for split in raw_ds.keys():
                limit = min(max_samples, len(raw_ds[split]))
                raw_ds[split] = raw_ds[split].select(range(limit))
                
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, 
                                                use_fast=False
                                                )
        
        def tokenize_batch(batch):
            tokens = tokenizer(batch["text"], max_length=max_length, 
                            truncation=True,
                            padding="max_length"
                            )
            if "label" in batch:
                tokens["labels"] = batch["label"]
            elif "labels" in batch:
                tokens["labels"] = batch["labels"]
            return tokens
        
        
        tmp_id = uuid.uuid4().hex[:8]
        tmp_train = f"{train_path}.{tmp_id}.tmp"
        tmp_test = f"{test_path}.{tmp_id}.tmp"
        
        train_cols_to_remove = raw_ds["train"].column_names
        test_cols_to_remove = raw_ds["test"].column_names
        

        raw_ds["train"].map(tokenize_batch, 
                            batched=True, 
                            remove_columns=train_cols_to_remove
                            ).to_parquet(tmp_train)
        
        raw_ds["test"].map(tokenize_batch, 
                            batched=True, 
                            remove_columns=test_cols_to_remove
                            ).to_parquet(tmp_test)
        
        os.replace(tmp_train, train_path)
        os.replace(tmp_test, test_path)
        
        return {"train_data_path": train_path, "test_data_path": test_path}
    

@registry.register(name="prepare_huggingface_data", registry_name="workflows")
class HuggingFaceDataWorkflow:
    
    def __call__(self, dataset_name, pretrained_model_name_or_path, 
                 output_dir, 
                 max_length: int = 128,
                 max_samples: int = None,
                 ) -> DataPrepOutput:
        cache_key = f"{os.path.abspath(output_dir)}_{max_samples}"
        dir_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
        actor_name = f"prep_coordinator_{output_dir}"
        coordinator = _DatasetPrepCoordinator.options(name=actor_name,
                                                      namespace="distrops",
                                                      lifetime="detached",
                                                      get_if_exists=True,
                                                      
                                                      ).remote()
        return ray.get(coordinator.ensure_prepared.remote(dataset_name=dataset_name,
                                                          pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                          output_dir=output_dir,
                                                          max_length=max_length,
                                                          max_samples=max_samples,
                                                          )
                       )
        
        

class PipelineOrchestrator:
    def __init__(self, global_config: dict, registry_name: str = "workflows"):
        self.global_config = global_config
        self.registry_name = registry_name
        self.pipeline_state = {**global_config}
        
        load_plugins(self.global_config.get("plugins", []))
        
    def run(self, metrics_queue=None, report_fn=None):
        validate_pipeline_dag(self.global_config)
        
        workflows = self.global_config.get("workflows", [])
        final_output = {}
        
        for idx, step_cfg in enumerate(workflows):
            name = step_cfg.get("name")
            step_registry = step_cfg.get("_registry_", self.registry_name)
            params = step_cfg.get("params") or {}
            resources = step_cfg.get("resources") or {"cpu": 1, "gpu": 0.0}
            
            execution_context = {**self.pipeline_state, **params, "metrics_queue": metrics_queue}
            
            print(f"Launching Stage {idx + 1}/{len(workflows)}: '{name}'",
                  f"[Allocated: {resources.get('cpu', 1)} CPU, {resources.get('gpu', 0.0)} GPU]"
                  )
            
            configured_remote_task = _execute_isolated_stage.options(num_cpus=resources.get("cpu",1),
                                                                     num_gpus=resources.get("gpu", 0.0)
                                                                     )
            
            stage_ref = configured_remote_task.remote(name, execution_context, step_registry)
            
            if metrics_queue is not None and report_fn is not None:
                while True:
                    ready, _ = ray.wait([stage_ref], timeout=0.5)
                    while not metrics_queue.empty():
                        report_fn(metrics_queue.get_nowait())
                    if ready:
                        break
            stage_output = ray.get(stage_ref)            
            self.pipeline_state.update(stage_output)
            self.pipeline_state[f"stage_{idx}_{name}"] = stage_output
            final_output = stage_output
        return final_output
    
    


class DataPrepOutput(TypedDict):
    train_data_path: str
    test_data_path: str
    
    
class TrainingOutput(TypedDict):
    checkpoint_path: str
    eval_loss: float
    eval_accuracy: float
    
class OptimizationOutput(TypedDict):
    quantized_model_path: str
    compression_ratio: float

