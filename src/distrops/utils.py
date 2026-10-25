import evaluate
import numpy as np
import torch
import yaml
import inspect
from typing import Callable, Any, Dict, Tuple

def tokenize_batch(batch, tokenizer, device):
    text = batch["text"]
    label = batch["labels"]
    text = text.tolist() if isinstance(text, np.ndarray) else text
    token = tokenizer(text, max_length=128, truncation=True, 
                    padding="longest", 
                    return_tensors="pt"
                    )
    token["labels"] = torch.tensor(label, dtype=torch.long)
    token = {k: v.to(device) for k,v in token.items()}
    return token

def compute_metric(eval_pred, metric_name):
    metric = evaluate.load(metric_name)
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return metric.compute(predictions=preds, references=labels)


def load_config_file(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



def get_expected_arguments(func, config):
    signature = inspect.signature(func)
    
    positional_args = []
    keyword_kwargs = []
    
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    
    bound_args = signature.bind_partial(**{k: v for k, v in config.items() if k in signature.parameters or has_var_keyword})
    
    for param_name, param in signature.parameters.items():
        if param_name not in bound_args.arguments:
            if param.default != inspect.Parameter.empty:
                bound_args.arguments[param_name] = param.default
                
    for param_name, value in bound_args.arguments.items():
        param = signature.parameters.get(param_name)
        if param is None:
            keyword_kwargs[param_name] = value

        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional_args.append(value)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            if isinstance(value, (list, tuple)):
                positional_args.extend(value)
            else:
                positional_args.append(value)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if isinstance(value, dict):
                keyword_kwargs.update(value)
        else:
            keyword_kwargs[param_name] = value
            
    return tuple(positional_args), keyword_kwargs




