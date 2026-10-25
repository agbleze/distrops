from typing import Callable, Dict, Any, List


class Registry:
    def __init__(self, registry_name: str = "DEFAULT"):
        self.registry_name = registry_name
        self._registry: Dict[str, Callable] = {}
        
    def register_component(self, name: str, cls: Any):
        """_summary_

        Args:
            name (str): name of the component -function or class to register
            cls (Any): Object to register - function or class

        Raises:
            ValueError: Raise when the name is already registered in the registry
        
        Returns:
            None: _description_
        """
        if name not in self._registry:
            raise ValueError(f"{name} is already in {self.registry_name}")
        self._registry[name] = cls
        
    def register(self, name: str = None) -> Callable:
        """@decorator hook interface

        Args:
            name (str, optional): Name of the component - function or class. Defaults to None.

        Returns:
            Callable: Function or class that is being registered in the registry.
        """
        def decorator(cls_or_func: Any) -> Callable:
            name = name or cls_or_func.__name__
            self.register_component(name, cls_or_func)
            return cls_or_func
        return decorator
    
    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise ValueError(f"{name} is not in {self.registry_name}")
        return self._registry[name]
    
    def list_available(self) -> List:
        return list(self._registry.keys())
        
        
DATASET_REGISTRY = Registry("DATASET_REGISTRY")
MODEL_REGISTRY = Registry("MODEL_REGISTRY")