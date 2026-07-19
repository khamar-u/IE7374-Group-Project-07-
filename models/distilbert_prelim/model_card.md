
# DistilBERT Preliminary Model — IE7374 Group 7

## Model Summary
A fine‑tuned DistilBERT model trained on patient drug reviews to classify adverse‑event sentiment categories (negative, neutral, positive).  
Uses minimally cleaned natural text (`review_clean`) to preserve semantic richness.

## Intended Use
- Detect adverse events in patient reviews  
- Support downstream NER + symptom clustering  
- Provide structured labels for summarization pipelines (BART)

## Dataset
Source: Drugs.com patient reviews  
Processed dataset: df_focused.csv  
Includes:
- cleaned text  
- processed text  
- VADER sentiment  
- topic modeling  
- label IDs  
- metadata (drugName, condition, rating)

## Training
- Base model: distilbert-base-uncased  
- Epochs: 2  
- Batch size: 16  
- Learning rate: 5e‑5  
- Optimizer: AdamW  

## Metrics
Final Accuracy: 0.8120  
Final Macro F1: 0.5484  
Epoch Losses: 0.8638 → 0.7283  

## Limitations
- Preliminary model trained on 500‑sample subset  
- Not intended for clinical decision‑making  
- Sensitive to noisy or sarcastic text  
- Does not detect multi‑label adverse events

## Ethical Considerations
- Outputs must be interpreted cautiously  
- Not a substitute for medical advice  
- Should not be used to classify real patient risk without expert review
