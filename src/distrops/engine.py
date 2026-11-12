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
