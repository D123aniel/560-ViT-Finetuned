import torch
import numpy as np
import evaluate
from datasets import load_dataset, concatenate_datasets
from transformers import BeitForImageClassification, AutoImageProcessor, Trainer, TrainingArguments

def evaluate_test_set():
    model_path = "./beit_finetuned/checkpoint-470"
    dataset_name = "D123aniel/560_ViT_dataset"
    
    # Load the saved model and processor
    processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    model = BeitForImageClassification.from_pretrained(model_path)

    # Re-create test slice
    test_fake = load_dataset(dataset_name, split="test[:1000]")
    test_real = load_dataset(dataset_name, split="test[-1000:]")
    test_subset = concatenate_datasets([test_fake, test_real]).shuffle(seed=42)

    
    def transform(example):
        inputs = processor(images=example["image"], return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"][0], "label": example["label"]}

    encoded_test = test_subset.map(transform, batched=False)

    # Define metrics
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return accuracy_metric.compute(predictions=preds, references=p.label_ids)

    
    eval_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=64,
        bf16=True, 
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=encoded_test,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation on test set...")
    results = trainer.evaluate()
    
    print("------------------------------------------------------------------")
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    evaluate_test_set()