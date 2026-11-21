
#%%
#from distrops import distrops


import pytest
import ray
from typing import TypedDict
import textwrap
from distrops.utils import get_expected_arguments, resolve_config_targets
from distrops.engine import PipelineOrchestrator
from distrops.validator import PipelineValidationError, validate_pipeline_dag
from distrops.registry import registry
import torch
import functools


print(f"CHECKING")

class MockPrepOutput(TypedDict):
    processed_uri: str
    input_dim: int
    num_classes: int
    
class MockTrainOutput(TypedDict):
    final_score: float
    optimizer_lr: float


@registry.register(name="adam", registry_name="optimizer")
class MockAdam:
    def __call__():
        torch.optim.adam()
        return torch.optim.Adam



@registry.register(name="mock_prep", registry_name="workflows")
class MockPrepWorkflow:
    def __call__(self, raw_uri: str, raw_features: int = 8, num_classes: int = 3, scale: int = 4) -> MockPrepOutput:
        return {"processed_uri": f"{raw_uri}_scaled_{scale}",
                "input_dim": raw_features,
                "num_classes": num_classes
                }   
    
@registry.register(name="mock_train", registry_name="workflows")
class MockTrainWorkflow:
    def __call__(self, processed_uri: str, input_dim: int,
                 num_classes: int, loss: torch.nn.Module,
                 optimizer: Callable, lr=0.01
                 ) -> MockTrainOutput:
        
        #loss_cls = torch.nn.CrossEntropyLoss
        #opt_cls = registry.get(optimizer, registry_name="optimizer")
        
        assert isinstance(loss, torch.nn.CrossEntropyLoss)
        assert isinstance(optimizer, functools.partial)
        
        model = torch.nn.Linear(input_dim, num_classes)
        opt = optimizer(model.parameters())
        assert isinstance(opt, torch.optim.SGD)
        
        x = torch.randn(4, input_dim)
        y = torch.randint(0, num_classes, (4,))
        opt.zero_grad()
        out = model(x)
        loss_val = loss(out, y)
        loss_val.backward()
        opt.step()
        
        assert processed_uri.endswith("_scaled_4")
        
        return {"final_loss": float(loss_val.item()),
                "optimizer_lr": float(opt.param_groups[0]["lr"])
                }


def test_recursive_resolve_config_targets_nested_and_partial():
    nested_cfg = {
        "criterion": {"_target_": "torch.nn.CrossEntropyLoss",
                      "label_smoothing": 0.15
                      },
        "opt_factory": {"_target_": "torch.optim.AdamW",
                        "_partial_": True,
                        "lr": 1e-4,
                        "weight_decay": 0.05
                        },
        "plain_string": "data.csv"
    }
    resolved = resolve_config_targets(nested_cfg)
    
    assert isinstance(resolved["criterion"], torch.nn.CrossEntropyLoss)
    assert resolved["criterion"].label_smoothing == 0.15
    assert isinstance(resolved["opt_factory"], functools.partial)
    assert resolved["plain_string"] == "data.csv"
    
    dummy_model = torch.nn.Linear(4, 2)
    opt = resolved["opt_factory"](dummy_model.parameters())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 1e-4
    
    
    
def test_dot_path_component_resolution():
    ce_cls = registry.get(name_or_path="torch.nn.CrossEntropyLoss",
                           registry_name="loss"
                           )
    assert ce_cls is torch.nn.CrossEntropyLoss
    
def test_registered_component_resolution():
    opt_cls = registry.get(name_or_path="adam", registry_name="optimizer")
    assert opt_cls is torch.optim.Adam
    
def test_dag_validation_passes_with_dot_paths_and_typed_dict():
    valid_config = {"workflows": [
                            {"name": "mock_prep", "params": {"raw_uri": "s3://bucket/data"}},
                            {"name": "mock_train",
                            "params": {"optimizer": {"_target_": "torch.optim.SGD",
                                                     "_partial_": True,
                                                     "lr": 0.05
                                                     }, 
                                       
                                       }
                            }
                        ]
                    }
    assert validate_pipeline_dag(valid_config) is True
    
def test_dag_validation_catches_missing_args_on_dot_path_class():
    broken_dot_path_config = {
        "workflows": [
            {"name": "mock_prep", "params": {"raw_uri": "s3://bucket/data"}},
            {"name": "mock_train", "params": {"loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                                                "optimizer": {"_target_": "torch.optim.SGD",
                                                              "_partial_": True,
                                                              }
                                              # lr intentionally omitted
                                            }
             
            }
        ]
    }    
    
    with pytest.raises(PipelineValidationError, match="Missing required arguments \\['lr'\\]"):
        validate_pipeline_dag(broken_dot_path_config)


#def test_pipeline_orchestrator_nested_targets_and_partial_end_to_end():
            
def test_dynamic_plugin_file_with_unified_registry_end_to_end(tmp_path):
    plugin_file = tmp_path / "custom_research_plugin.py"
    plugin_code = textwrap.dedent("""
                                  from typing import TypedDict
                                  from distrops.registry import registry
                                  
                                  @registry.registry(registry_name="loss", name="custom_focal_loss")
                                  class CustomFocalLoss:
                                      def __init__(self, focal_gamma: float, focal_alpha: float = 0.25):
                                          self.gamma = focal_gamma
                                          self.alpha = focal_alpha
                                          
                                      def compute(self, base_val: float) -> float:
                                          return base_val * self.gamma * self.alpha
                                          
                                          
                                  class PluginStageOutput(TypedDict):
                                      computed_loss: float
                                  
                                  
                                  @registry.register(registry_name="workflows", name="plugin_eval_workflow"):
                                  class PluginEvalWorkflow:
                                      def __call__(self, loss: str, focal_gamma: float, focal_alpha: float = 0.25) -> PluginStageOutput:
                                          loss_cls = registry.get(registry_name="loss", name=loss)
                                          loss_fn = loss_cls(focal_gamma=focal_gamma, focal_alpha=focal_alpha)
                                          return {"computed_loss": loss_fn.compute(10.0)}
                                     
                                    
                                
                                  """
                                  ) 
    plugin_file.write_text(plugin_code)
    ray.init(ignore_reinit_error=True, num_cpus=2)
    try:
        invalid_config = {
            "plugins": [str(plugin_file)],
            "workflows": [
                {"name": "plugin_eval_workflow",
                 "resources": {"cpu": 1, "gpu": 0.0},
                 "params": {"loss": "custom_focal_loss"},
                 }
            ]
        }
        orchestrator_fail = PipelineOrchestrator(invalid_config) 
        with pytest.raises(PipelineValidationError, match="Missing required arguments \\ ['focal_gamma'\\]"):
            orchestrator_fail.run()
            
        valid_config = {
            "plugins": [str(plugin_file)],
            "workflows": [
                {"name": "plugin_eval_workflow",
                 "resources": {"cpu": 1, "gpu": 0.0},
                 "params": {"loss": {"_target_": "custom_focal_loss",
                                    },
                            "focal_gamma": 2.0,
                            "focal_alpha": 0.5
                            },
                 }
            ]
        }
        orchestrator_pass = PipelineOrchestrator(valid_config)
        result = orchestrator_pass.run()
        assert result == {"computed_loss": 10.0}    
    finally:
        ray.shutdown()
        
        
@pytest.fixture
def mock_registry():
    return {"mock_prep": MockPrepWorkflow, "mock_train": MockTrainWorkflow}


def test_signature_get_expected_arguments():
    def target_fn(a: int, /, b: str, c: float = 1.5, *, d: bool = True):
        return (a,b,c,d)
    
    config = {"a": 10, "b": "hello",
              "d": False,
              "unused_infra_key": "/tmp/ray",
              "gpu": 0.5
              }
    args, kwargs = get_expected_arguments(target_fn, config)
    assert args == (10,)
    assert kwargs == {"b": "hello", "c": 1.5, "d": False}
    assert target_fn(*args, **kwargs) == (10, "hello", 1.5, False)


# def test_dag_validation_passes_with_valid_typed_dict_flow():
#     valid_config = {
#         "workflows": [
#             {"name": "mock_prep", "params": {"raw_uri": "s3://bucket/data"}},
#             {"name": "mock_train", "params": {"epochs": 5}},
#         ]
#     }
#     assert validate_pipeline_dag(valid_config) is True
    
    
def test_dag_validation_fails_on_missing_upstream_key():
    broken_config = {
        "workflows": [
            {"name": "mock_train", "params": {"loss": {"_target_": "torch.nn.CrossEntropyLoss",
                                                        "label_smoothing": 0.1,
                                                        },
                                            "optimizer": {"_target_": "torch.optim.SGD",
                                                            "_partial_": True,
                                                            "lr": 0.025,
                                                            "momentum": 0.9
                                                            }
                                            }
             }
        ]
    }
    
    with pytest.raises(PipelineValidationError, match="Missing required argument \\['processed_uri'\\]"):
        validate_pipeline_dag(broken_config)
        
        
def test_dag_validation_fails_on_unregistered_workflow():
    unregistered_config = {
        "workflows": [
            {"name": "non_existent_step", "params": {}},
        ]
    }
    with pytest.raises(PipelineValidationError, match="unregistered workflow 'non_existent_step'"):
        validate_pipeline_dag(unregistered_config)
        
        
def test_pipeline_orchestrator_nested_targets_and_partial_end_to_end_execution():
    ray.init(ignore_reinit_error=True, num_cpus=2)
    try:
        config = {
            "workflows": [
                {
                    "name": "mock_prep",
                    "resources": {"cpu": 1, "gpu": 0.0},
                    "params": {"raw_uri": "dataset_v1", "scale": 4, 
                               "raw_features": 6, "num_classes": 4
                               },
                },
                {
                    "name": "mock_train",
                    "resources": {"cpu": 1, "gpu": 0.0},
                    "params": {"loss": {"_target_": "torch.nn.CrossEntropyLoss",
                                        "label_smoothing": 0.1,
                                        },
                               "optimizer": {"_target_": "torch.optim.SGD",
                                             "_partial_": True,
                                             "lr": 0.025,
                                             "momentum": 0.9
                                             }
                               },
                }
            ]
        }
        orchestrator = PipelineValidationError(config)
        result = orchestrator.run()
        
        assert result["optimizer_lr"] == 0.025
        assert isinstance(result["final_loss"], float)
        assert orchestrator.pipeline_state["input_dim"] == 6
    finally:
        ray.shutdown()
        
