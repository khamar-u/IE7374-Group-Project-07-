
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json
import numpy as np

MODEL_PATH = "models/distilbert_prelim"

class DistilBERTClassifier:
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    def predict(self, text):
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).numpy()[0]
            label = np.argmax(probs)

        return {
            "input": text,
            "predicted_label": int(label),
            "probabilities": probs.tolist()
        }

if __name__ == "__main__":
    clf = DistilBERTClassifier()
    sample = "This medication made my symptoms worse and caused headaches."
    print(clf.predict(sample))
