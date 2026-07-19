
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(self.labels[idx])
        }

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["review_clean", "label_id"])
    return df

def main():
    df = load_data("data/processed/df_focused.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["review_clean"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(df["label_id"].unique())
    )

    training_args = TrainingArguments(
        output_dir="models/distilbert_prelim",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_json("models/distilbert_prelim/metrics.json")

    model.save_pretrained("models/distilbert_prelim")
    tokenizer.save_pretrained("models/distilbert_prelim")

if __name__ == "__main__":
    main()
