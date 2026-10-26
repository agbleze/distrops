from .registry import MODEL_REGISTRY
from .utils import get_expected_arguments
import torchvision.models as models

@MODEL_REGISTRY.register(name="huggingface_sequence_classification")
def hf_reflective_factory(config: dict):
    from transformers import AutoModelForSequenceClassification
    
    args, kwargs = get_expected_arguments(AutoModelForSequenceClassification.from_pretrained, config)
    
    model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
    return model



@MODEL_REGISTRY.register(name="torchvision_image_classification")
def torchvision_image_classification_factory(config: dict):
    model_name = config.get("model_name", "resnet18")
    if not hasattr(models, model_name):
        raise AttributeError(f"{model_name} is not a valid torchvision model in torchvision.models. Please check the model name.")
    
    model_builder = getattr(models, model_name)
    args, kwargs = get_expected_arguments(model_builder, config)
    model = model_builder(*args, **kwargs)
    return model