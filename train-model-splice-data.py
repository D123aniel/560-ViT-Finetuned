from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import BeitForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
import os
from typing import Any, Dict, cast
from transformers.trainer_utils import EvalPrediction

login(os.getenv("HF_TOKEN"))

def main():

    # Load base model & dataset
    model_name = "microsoft/beit-base-patch16-224"
    # this was nonsliced beforehand so do a bit of slicing
    # this is with a 60:20:20 split with 10K items, change as necessary
    dataset_name = "D123aniel/560_ViT_dataset"
    train_fake = load_dataset(dataset_name, split="train[:3000]") #3000
    train_real = load_dataset(dataset_name, split="train[-3000:]")
    train_subset = concatenate_datasets([train_fake, train_real])

    val_fake = load_dataset(dataset_name, split="validation[:1000]") #1000
    val_real = load_dataset(dataset_name, split="validation[-1000:]")
    val_subset = concatenate_datasets([val_fake, val_real])

    test_fake = load_dataset(dataset_name, split="test[:1000]") #1000
    test_real = load_dataset(dataset_name, split="test[-1000:]")
    test_subset = concatenate_datasets([test_fake, test_real])

    # Optionally shuffle to mix front/back examples
    train_subset = train_subset.shuffle(seed=42)
    val_subset = val_subset.shuffle(seed=42)
    test_subset = test_subset.shuffle(seed=42)

    # Combine into one dataset dict
    dataset = DatasetDict({
        "train": train_subset,
        "validation": val_subset,
        "test": test_subset
    })

    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    num_labels = len(dataset["train"].features["label"].names)

    model = BeitForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(dataset["train"].features["label"].names)},
        label2id={label: i for i, label in enumerate(dataset["train"].features["label"].names)},
        ignore_mismatched_sizes=True
    )

    # Convert to beit tensor
    def transform(example):
        inputs = feature_extractor(images=example["image"], return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"][0], "label": example["label"]}

    encoded_dataset = dataset.map(transform, batched=False)

    # Load training arguments (optimized for A100)
    training_args = TrainingArguments(
        output_dir="./beit_finetuned",
        per_device_train_batch_size=64, #64
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        learning_rate=3e-5,
        warmup_steps=0.05,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=True,
        dataloader_num_workers=4, #8
        seed=42,
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        preds = np.argmax(p.predictions, axis=1)
        metrics = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        return metrics if metrics is not None else {}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    # Make sure you have a GPU
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    trainer.train()
    trainer.save_model("./beit_finetuned")
    
    results = trainer.evaluate(eval_dataset=cast(Any, encoded_dataset["test"]))
    print("Test accuracy:", results["eval_accuracy"])

if __name__ == "__main__":
    main()