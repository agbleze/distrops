import inspect
from typing import get_type_hints
from registry import registry
 
class PipelineValidationError(Exception):
    pass

FRAMEWORK_INJECTED_PARAMS = {"self", "config", "params", "model_parameters"}


def _check_callable_signature(callable_obj: Any, available_keys: set,
                              stage_label: str,
                              ignore_params: set = None
                              ) -> None:
    """Inspects a callable's required parameters against available configuration

    Args:
        callable_obj (Any): _description_
        available_keys (set): _description_
        stage_label (str): _description_
        ignore_params (set, optional): _description_. Defaults to None.
    """
    ignore_params = ignore_params or FRAMEWORK_INJECTED_PARAMS
    sig = inspect.signature(callable_obj)
    missing_params = []
    for param_name, param in sig.parameters.items():
        if param_name in ignore_params:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if param.default is inspect.Parameter.empty and param_name not in available_keys:
            missing_params.append(param_name)
            
    if missing_params:
        raise PipelineValidationError(
            f"Contract Breach in {stage_label}: Missing required arguments {missing_params}. "
            f"Neither defined in 'params' nor produced by upstream stages."
        )


    
def validate_pipeline_dag(global_config: dict, workflow_registry) -> bool:
    workflows = global_config.get("workflows", [])
    if not workflows:
        raise PipelineValidationError(f"Configuration manifest contains no 'workflows' steps")
        
    available_keys = set(global_config.keys())
    
    for idx, step in enumerate(workflows):
        step_name = step.get("name")
        step_params = step.get("params", {})
        
        if step_name not in workflow_registry:
            raise PipelineValidationError(
                f"Stage {idx + 1} references unregistered workflow '{step_name}'.",
                f"Available: {list(workflow_registry.keys())}"
            )
            
        workflow_cls = workflow_registry[step_name]
        sig = inspect.signature(workflow_cls.__call__)
        
        stage_input_keys = available_keys.union(set(step_params.keys()))
        missing_params = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty and param_name not in stage_input_keys:
                missing_params.append(param_name)
                
        if missing_params:
            raise PipelineValidationError(
                f"Contract Breach at Stage {idx + 1} ('{step_name}'): "
                f"Missing required arguments {missing_params}. Neither defined in 'params' "
                f"nor produced by upstream stages."
            )        
        
        type_hints = get_type_hints(workflow_cls.__call__)
        return_type = type_hints.get("return")
        
        if hasattr(return_type, "__annotations__"):
            produced_keys = set(return_type.__annotations__.keys())
            available_keys.update(produced_keys)  
            
        elif return_type is dict:
            available_keys.update(return_type.keys())
    print("Dry-Run DAG Validation Passed: All stage contracts and signatures are aligned.")
    return True

