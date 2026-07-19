# AI-Powered Adverse Event Summarizer from Patient Reviews

**Team 7 — IE7374: Generative AI**

## Overview

This project builds an NLP pipeline to detect and summarize adverse events described in patient reviews. We focus on extracting symptom mentions and producing conservative, source-grounded summaries of adverse events using pretrained transformer models.

## Models

- Summarization: BART (`facebook/bart-large-cnn`) for abstractive, source-grounded summaries.
- Adverse event classification: DistilBERT (`distilbert-base-uncased`) for efficient classification.

## Dataset

- UCI Drug Reviews Dataset (Kaggle: jessicali9530/kuc-hackathon-winter-2018), scraped from drugs.com.
- ~215k reviews (CSV with columns: uniqueID, drugName, condition, review, rating, date, usefulCount).
- Project will focus on Depression and Anxiety subsets (~15k reviews).

## Research Questions

1. How accurately can fine-tuned DistilBERT identify adverse event mentions versus a keyword baseline (precision/recall/F1)?
2. What symptom clusters appear most often in low-rated reviews (≤4) for antidepressants and anxiolytics, and do they match known side effect profiles?
3. Can BART generate adverse event summaries grounded in source reviews without making causal claims, and which prompt designs enforce that constraint?
4. How well does VADER sentiment scoring correlate with patient ratings, and where do the signals diverge?

## Plan of Action & Timeline

- Literature review & benchmarking (July 6–12): evaluate VADER, DistilBERT, BART; run small feasibility tests.
- Data pipeline & repo setup (July 6–18): merge CSVs, decode HTML entities, produce `review_clean` and `review_processed`, split focused subset (70/15/15).
- Model implementation & evaluation (July 13–25): fine-tune DistilBERT, run spaCy NER, cluster symptoms, apply BART to flagged reviews.
- Deliverables: Milestone 3 (July 19) — data pipeline and repo; Milestone 4 (July 26) — end-to-end pipeline; Milestone 5 (Aug 9) — technical report and presentation.

Frameworks: Python 3.10, HuggingFace Transformers, PyTorch, spaCy, NLTK, scikit-learn.

## Team Contributions

- Jin-woo Hong — Data Engineer: dataset sourcing, repo init, initial EDA. Deliverables: `drugsComTrain_raw.csv`, `drugsComTest_raw.csv`, `project_team_7.ipynb`.
- Yosephine Tong — Data Analyst: cleaning, preprocessing, EDA, VADER scoring, topic modeling. Deliverables: `df_clean.csv`, `df_focused.csv`, EDA notebook.
- Umang Khamar — NLP Engineer: DistilBERT fine-tuning, spaCy NER, symptom clustering, metrics (precision/recall/F1). Deliverables: distilbert_prelim model artifacts, NER outputs, classification metrics.
- Min Chang — AI Summarization Lead: BART pipeline, prompt design, ROUGE evaluation, Streamlit demo, report and slides.

Note: `review_clean` (minimally cleaned natural text) will be used for transformer models; `review_processed` is for classical NLP.

## Evaluation

- Classification: precision, recall, F1 against labeled adverse-event flags.
- Summaries: ROUGE scores and qualitative review; ensure conservative, source-grounded outputs.

## References

- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text.
- Lewis, M., et al. (2020). BART: Denoising sequence-to-sequence pre-training for NLG.
- Sanh, V., et al. (2019). DistilBERT: smaller, faster BERT.
- Sarker, A., & Gonzalez, G. (2015). ADR detection from social media.
- Kaggle dataset: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
