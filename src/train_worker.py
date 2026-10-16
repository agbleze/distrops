from .data_loader import data_loader
from utils import compute_metric
from functools import partial
import torch


def train_model(config):
    from transformers import (AutoModelForSequenceClassification
                            )
    import ray
    
    model_name = config.get("model_name")
    batch_size = config.get("batch_size")
    metric_name = config.get("metric_name", "accuracy")
    epochs = config.get("epochs")
    learning_rate = config.get("learning_rate")
    
    train_data_iterable, test_data_iterable, max_steps_per_epoch = data_loader(config=config)
    
    metric = partial(compute_metric, metric_name=metric_name)


    print("Getting Model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                use_safetensors=True,
                                                                num_labels=77
                                                                )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    max_step = max_steps_per_epoch * epochs
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in train_data_iterable:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        avg_train_loss = train_loss / train_steps if train_loss > 0 else 0.0
        
    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    
    with torch.no_grad():
        for batch in test_data_iterable:
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            references = batch["labels"].cpu().numpy()
            
            metric.add_batch(predictions=preds, references=references)
            eval_steps += 1

    avg_eval_loss = eval_loss / eval_steps if eval_loss > 0 else 0.0
    accuracy = metric.compute()["accuracy"]

    metric_result = {"eval_loss": avg_eval_loss,
                     "train_loss": avg_train_loss,
                     "accuracy": accuracy,
                     "epoch": epoch
                     }

    
    ray.tune.report(metric_result)
    
