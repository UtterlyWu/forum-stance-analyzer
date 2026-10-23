# Forum Stance Analyzer

An NLP system for identifying agreement and disagreement between comments in online discussions.

The project explores whether semantic embeddings can be fine-tuned to detect stance relationships between a parent comment and its reply, then applies the trained model to Reddit discussion threads.

## What It Does

Given two related comments, the system predicts whether the reply agrees with the original comment.

The project includes:

- Fine-tuning SentenceTransformer models on labelled comment-reply pairs
- Separate representation learning for agreement and disagreement relationships
- A PyTorch classification model built on top of the learned embeddings
- Model training and evaluation utilities
- A Reddit scraper for collecting discussion threads
- A desktop GUI for scraping a Reddit thread and displaying stance predictions

## Model Architecture

The system uses SentenceTransformer models based on `all-mpnet-base-v2`.

Agreement and disagreement representations are learned using contrastive training. The embeddings generated for each pair of comments are then combined and passed through a neural-network classifier.

High-level flow:

```text
Parent Comment ──┐
                 ├─> Agreement / Disagreement Embeddings ─> Classifier ─> Stance
Reply Comment ───┘
