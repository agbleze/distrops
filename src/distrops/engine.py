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
    NON_TARGET_ROOT_KEYS = {"workflows", "hpo_bounds", "infrastructure", "plugins"}
    load_plugins(execution_context.get("plugins", []))
    workflow_cls = registry.get(name_or_path=workflow_name, registry_name=registry_name)
    workflow_instance = workflow_cls() if inspect.isclass(workflow_cls) else workflow_cls
    filtered_context = {
        k: v for k,v in execution_context.items() if k not in NON_TARGET_ROOT_KEYS
    }
    #instantiated_context = resolve_config_targets(execution_context, registry_name=registry_name)
    instantiated_context = resolve_config_targets(filtered_context, registry_name=registry_name)
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
    
    

@registry.register(name="prepare_huggingface_data", registry_name="workflows")
class HuggingFaceDataWorkflow:
    
    def __call__(self, dataset_name, pretrained_model_name_or_path, output_dir, max_length: int = 128) -> DataPrepOutput:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.parquet")
        test_path = os.path.join(output_dir, "test.parquet")
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"HuggingFaceDataWorkflow: train_path: {train_path} and test_path: ")
            return {"train_data_path": train_path, "test_data_path": test_path}
            
        raw_ds = load_dataset(dataset_name)
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
        
        
        
        train_cols_to_remove = raw_ds["train"].column_names
        test_cols_to_remove = raw_ds["test"].column_names
        

        raw_ds["train"].map(tokenize_batch, batched=True, remove_columns=train_cols_to_remove).to_parquet(train_path)
        raw_ds["test"].map(tokenize_batch, batched=True, remove_columns=test_cols_to_remove).to_parquet(test_path)
        
        return {"train_data_path": train_path, "test_data_path": test_path}


@registry.register(name="gradient_descent", registry_name="workflows")
class DeepLearningWorkflow:
    
    def __call__(self, train_data_path: str, test_data_path: str, 
                 #model_registry_key: str, 
                 epochs: int =3,
                 batch_size=16,
                 drop_last=False,
                 *args, **kwargs
                 ):
        train_ds = ray.data.read_parquet(train_data_path)
        train_stream = train_ds.iter_torch_batches(batch_size=batch_size)
        
        test_ds = ray.data.read_parquet(test_data_path)
        test_stream = test_ds.iter_torch_batches(batch_size=batch_size)
        
        total_samples = train_ds.count()
        if drop_last:
            max_steps_per_epoch = total_samples // batch_size
        else:
            max_steps_per_epoch = math.ceil(total_samples / batch_size)
            
        max_steps = max_steps_per_epoch * epochs
        
        ...
        best_checkpoint = ""
        return {"checkpoint_point": best_checkpoint, "eval_loss": 0.3}
        
        

class PipelineOrchestrator:
    def __init__(self, global_config: dict, registry_name: str = "workflows"):
        self.global_config = global_config
        self.registry_name = registry_name
        self.pipeline_state = {**global_config}
        
        load_plugins(self.global_config.get("plugins", []))
        
    def run(self):
        validate_pipeline_dag(self.global_config)
        
        workflows = self.global_config.get("workflows", [])
        final_output = {}
        
        for idx, step_cfg in enumerate(workflows):
            name = step_cfg.get("name")
            step_registry = step_cfg.get("_registry_", self.registry_name)
            params = step_cfg.get("params") or {}
            resources = step_cfg.get("resources") or {"cpu": 1, "gpu": 0.0}
            
            execution_context = {**self.pipeline_state, **params}
            
            print(f"Launching Stage {idx + 1}/{len(workflows)}: '{name}'",
                  f"[Allocated: {resources.get('cpu', 1)} CPU, {resources.get('gpu', 0.0)} GPU]"
                  )
            
            configured_remote_task = _execute_isolated_stage.options(num_cpus=resources.get("cpu",1),
                                                                     num_gpus=resources.get("gpu", 0.0)
                                                                     )
            stage_output = ray.get(configured_remote_task.remote(name, execution_context, step_registry))
            
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

