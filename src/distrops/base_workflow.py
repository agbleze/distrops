
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from .registry import registry

class BaseWorkflow(ABC):
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        registry.register(cls.__name__)
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __call__method.")
    
    
    
    