from typing import Dict, Any, List, Callable
import abc
import torch
import ray
from utils import get_expected_arguments

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