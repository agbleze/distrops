from typing import Dict, Any, List, Callable
import abc
import torch


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
