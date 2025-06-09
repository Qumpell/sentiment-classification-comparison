# Sentiment Analysis — Comparison of Different Methods

This project presents a comparison of three different approaches to binary sentiment classification:

1. **Naive Bayes with TF-IDF features**
2. **LSTM network with pre-trained GloVe embeddings**
3. **Transformer-based model (DistilBERT from Hugging Face)**

The goal is to evaluate how different NLP modeling strategies perform on the same sentiment analysis task.

---

## Dataset

We use a standard binary sentiment dataset (IMDB), consisting of text samples labeled as either **positive** or **negative**.

The dataset is split into:
- **Training set**
- **Test set**

---

## Methods Overview

### 1. Naive Bayes + TF-IDF
- Feature extraction using scikit-learn's `TfidfVectorizer`
- Multinomial Naive Bayes classifier
- Serves as a lightweight baseline

### 2. LSTM + GloVe Embeddings
- Each word in a sentence is mapped to its pre-trained GloVe embedding (300D)
- The sentence is passed through an LSTM-based neural network built in PyTorch
- Final classification is done using a fully connected output layer

### 3. Transformer (DistilBERT)
- Pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face
- Tokenization and fine-tuning handled using the `transformers` library and the `Trainer` API
- Leverages attention mechanisms and contextual embeddings

---

## Evaluation

Each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
