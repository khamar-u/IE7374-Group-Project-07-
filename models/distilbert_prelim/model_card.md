
# DistilBERT Preliminary Model — IE7374 Group 7

## Model Summary
A fine‑tuned DistilBERT model trained on patient drug reviews to classify adverse‑event sentiment categories (`negative`, `neutral`, `positive`).  
The model uses minimally cleaned natural text (`review_clean`) to preserve semantic richness.

## Intended Use
- Detect adverse events in patient reviews  
- Support downstream NER + symptom clustering  
- Provide structured labels for summarization pipelines (BART)

## Dataset
Source: Drugs.com patient reviews  
Processed dataset: `df_focused.csv`  
Includes:
- cleaned text  
- processed text  
- VADER sentiment  
- topic modeling  
- label IDs  
- metadata (drugName, condition, rating)

## Training
- Base model: `distilbert-base-uncased`  
- Epochs: 3  
- Batch size: 8  
- Learning rate: 5e‑5  
- Train/test split: 80/20  
- Loss: Cross‑entropy  
- Optimizer: AdamW  

## Metrics
Stored in `metrics.json`:
- precision  
- recall  
- F1  
- support  

## Limitations
- Preliminary model trained on 500‑sample subset  
- Not intended for clinical decision‑making  
- Sensitive to noisy or sarcastic text  
- Does not detect multi‑label adverse events

## Ethical Considerations
- Outputs must be interpreted cautiously  
- Not a substitute for medical advice  
- Should not be used to classify real patient risk without expert review
