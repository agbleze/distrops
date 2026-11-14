from typing import Dict, Any, List, Callable
import abc
import torch
import ray
from utils import get_expected_arguments
import os
from datasets import load_dataset
from transformers import AutoTokenizer


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
        
        data_loader_fn = DATA_REGISTRY.get(config.get("data_loader"))  
        
        
        


@ray.remote
def _execute_isolated_stage(workflow_cls: Any, execution_context):
    workflow_instance = workflow_cls()
    args, kwargs = get_expected_arguments(workflow_instance, execution_context)
    output = workflow_instance(*args, **kwargs)
    
    if not isinstance(output, dict):
        raise TypeError(f"Worflow '{workflow_cls.__name__}' must return a dict, got {type(output)}")
    return output


WORKFLOW_REGISTRY = {}

def register_workflow(name: str):
    def decorator(cls):
        WORKFLOW_REGISTRY[name] = cls
        return cls
    return decorator


@register_worflow("prepare_huggingface_data")
class HuggingFaceDataWorkflow:
    
    def __call__(self, dataset_name, tokenizer_name, output_dir, max_length: int = 128):
        os.makedirs(output_dir, exist_ok=True)
        raw_ds = load_dataset(dataset_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        
        def tokenize_batch(batch):
            tokens = tokenizer(batch["text"], max_length=max_length, 
                               truncation=True,
                               padding="max_length"
                               )
            tokens["labels"] = batch["label"]
            return tokens
        
        train_path = os.path.join(output_dir, "train.parquet")
        test_path = os.path.join(output_dir, "test.parquet")
        
        raw_ds["train"].map(tokenize_batch, batched=True).to_parquet(train_path)
        raw_ds["test"].map(tokenize_batch, batched=True).to_parquet(test_path)
        
        return {"train_data_path": train_path, "test_data_path": test_path}


@register_workflow("gradient_descent")
class DeepLearningWorkflow:
    
    def __call__(self, train_data_path: str, test_data_path: str, 
                 model_registry_key: str, 
                 epochs: int =3,
                 batch_size=16,
                 *args, **kwargs
                 ):
        train_stream = ray.data.read_parquet(train_data_path).iter_torch_batches(batch_size=batch_size)
        test_stream = ray.data.read_parquet(test_data_path).iter_torch_batches(batch_size=batch_size)
        
        
        ...
        best_checkpoint = ""
        return {"checkpoint_point": best_checkpoint, "eval_loss": inf}
        
        

class PipelineOrchestrator:
    def __init__(self, global_config: dict, workflow_registry: dict, ):
        self.global_config = global_config
        self.workflow_registry = workflow_registry
        self.pipeline_state = {**global_config}
        
    def run(self):
        validate_pipeline_dag(self.global_config, self.workflow_registry)
        
        workflows = self.global_config.get("workflows", [])
        final_output = {}
        
        for idx, step_cfg in enumerate(workflows):
            name = step_cfg.get("name")
            params = step_cfg.get("params")
            resources = step_cfg.get("resources", {"cpu": 1, "gpu": 0.0})
            
            workflow_cls = self.workflow_registry.get("name")
            execution_context = {**self.pipeline_state, **params}
            
            print(f"Launching Stage {idx + 1}/{len(workflows)}: '{name}'",
                  f"[Allocated: {resources.get('cpu'), 1} CPU, {resources.get('gpu', 0.0)} GPU]"
                  )
            
            configured_remote_task = _execute_isolated_stage.options(num_cpus=resources.get("cpu",1),
                                                                     num_gpus=resources.get("gpu", 0.0)
                                                                     )
            stage_output = ray.get(configured_remote_task.remote(workflow_cls, execution_context))
            
            self.pipeline_state.update(stage_output)
            self.pipeline_state[f"stage_{idx}_{name}"] = stage_output
            final_output = stage_output
        return final_output
    
    
 
    