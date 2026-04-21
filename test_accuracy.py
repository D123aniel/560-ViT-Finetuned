import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import evaluate
from datasets import load_dataset, concatenate_datasets
from transformers import BeitForImageClassification, AutoImageProcessor, Trainer, TrainingArguments

from huggingface_hub import login
import os
login(os.getenv("HF_TOKEN"))

def evaluate_test_set():
    model_path = "D123aniel/560_ViT_Finetuned"
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

    output = trainer.predict(encoded_test)
    
    # 1. Get predictions and true labels
    y_preds = np.argmax(output.predictions, axis=1)
    y_true = output.label_ids
    
    # 2. Compute Confusion Matrix
    # Use the 'id2label' from your model config if available for better labeling
    labels = list(model.config.id2label.values()) if hasattr(model.config, 'id2label') else None
    cm = confusion_matrix(y_true, y_preds)

    print("------------------------------------------------------------------")
    print(f"Test Accuracy: {output.metrics['test_accuracy']:.4f}")
    print("------------------------------------------------------------------")
    print("Confusion Matrix:")
    print(cm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix: Fake vs Real")
    plt.show()

if __name__ == "__main__":
    evaluate_test_set()