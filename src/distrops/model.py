from .registry import MODEL_REGISTRY
from .utils import get_expected_arguments

@MODEL_REGISTRY.register(name="huggingface_sequence_classification")
def hf_reflective_factory(config: dict):
    from transformers import AutoModelForSequenceClassification
    
    args, kwargs = get_expected_arguments(AutoModelForSequenceClassification.from_pretrained, config)
    
    model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
    return model