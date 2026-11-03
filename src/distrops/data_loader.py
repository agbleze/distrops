
import ray
from datasets import load_dataset
from .utils import tokenize_batch
from functools import partial
from transformers import AutoTokenizer
from .registry import DATASET_REGISTRY
import sklearn.datasets as sk_datasets


def get_data(config):
    dataset_name = config.get("dataset_name")
    batch_size = config.get("batch_size")
    model_name = config.get("model_name")
    device = config.get("device")
    
    dataset = load_dataset(dataset_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    collate_fn = partial(tokenize_batch, tokenizer=tokenizer, device=device)
    
    train_data = ray.data.from_items(dataset["train"].to_list()).rename_columns({"label": "labels"})
    test_data = ray.data.from_items(dataset["test"].to_list()).rename_columns({"label": "labels"})
    train_data_iterable = train_data.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    test_data_iterable = test_data.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    max_steps_per_epoch = train_data.count() // batch_size
    return train_data_iterable, test_data_iterable, max_steps_per_epoch
        
        
@DATASET_REGISTRY.register("huggingface_dataset")
def load_huggingface_dataset(config):
    dataset_name = config.get("dataset_name")
    dataset = load_dataset(dataset_name)
    return dataset


@DATASET_REGISTRY.register("sklearn_dataset")
def load_sklearn_dataset(config):
    dataset_name = config.get("dataset_name")
    if not hasattr(sk_datasets, dataset_name):
        raise AttributeError(f"{dataset_name} is not a valid sklearn dataset in sklearn.datasets. Please check the dataset name.")
    
    dataset_builder = getattr(sk_datasets, dataset_name)
    dataset = dataset_builder()
    return dataset