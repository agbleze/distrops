from typing import Dict, Any, List, Callable
import abc


class DistropsWorkflow(abc.ABC):
    @abc.abstractmethod
    def execute_training_pipeline(self, *args, **kwargs):
        pass
class DecisionTreeWorkflow:
    def __init__(self, config):
        self.config = config
        
        
        
        
