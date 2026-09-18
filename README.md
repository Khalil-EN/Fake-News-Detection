# Fake News Detection with Machine Learning

This repository was developed as part of a Machine Learning course assignment (GIF-7005). Its purpose is academic: to apply and compare supervised text-classification techniques taught in the course, and to practice building a full ML pipeline from raw data to model evaluation. The dataset, methodology, and results below should be read with that context in mind.

A supervised text-classification project that compares four ML models — Logistic Regression, SVM, Naive Bayes, and XGBoost — on the task of distinguishing real news articles from fake ones.

## Dataset

- `True.csv` — 21,417 real news articles (Reuters)
- `Fake.csv` — 23,481 fake news articles

This is the well-known ["Fake and Real News Dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle. Each row has a `title`, `text`, `subject`, and `date`.

## Pipeline

1. **Cleaning** — drop duplicates, merge `title` + `text` into a single `Content` field, remove special characters/URLs, tokenize, remove stopwords, lemmatize with spaCy.
2. **Split** — 80% train / 20% test, stratified on the label.
3. **Vectorization** — TF-IDF.
4. **Models** — Logistic Regression, Naive Bayes, Linear SVM, and XGBoost, each tuned with `GridSearchCV`.
5. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrices.

## Results

| Model | Test Accuracy |
|---|---|
| XGBoost | ~99.9% |
| SVM | ~99.4% |
| Logistic Regression | ~99.4% |
| Naive Bayes | ~94.5% |

## Understanding the ~99% Accuracy

The high scores reached here (up to ~99.9% for XGBoost) are a good illustration of how well TF-IDF combined with classic ML models can separate two text sources — but part of this performance comes from characteristics of the dataset itself rather than from the models learning to reason about truthfulness in a general sense. It's worth analyzing where this signal comes from:

- **Source fingerprint**: most `True.csv` articles begin with an agency dateline, e.g. `"WASHINGTON (Reuters) - ..."`, which doesn't appear in `Fake.csv`. This gives the model an easy, reliable cue tied to the article's origin rather than its content.
- **Disjoint `subject` categories**: `True.csv` only contains `politicsNews` and `worldnews`, while `Fake.csv` uses a different set entirely (`News`, `politics`, `left-news`, `Government News`, `US_News`, `Middle-east`). Since the two classes never share a subject label, this field (and the vocabulary tied to it) becomes a strong, almost direct predictor once vectorized.
- **Different collection process**: the two files were gathered from different sources and periods, which naturally introduces stylistic differences (formatting, punctuation, article length) that TF-IDF picks up on easily.

In other words, the models are very effective at distinguishing *these two specific sources of text*, and that's exactly what the dataset makes easiest to learn. This is a common and well-documented characteristic of this particular dataset, and it doesn't take away from the value of the exercise — building the full pipeline, comparing models, and tuning hyperparameters. It does mean the ~99% figure should be read as "near-perfect separation of these two sources," not "near-perfect fake-news detection in general."

## Requirements

```
pandas
numpy
nltk
spacy (+ en_core_web_sm)
wordcloud
matplotlib
seaborn
plotly
scikit-learn
xgboost
```

## Usage

Place `True.csv` and `Fake.csv` in a `data/` folder at the project root, then run the notebook:

```
data/
  True.csv
  Fake.csv
projetCodeNotebook.ipynb
```

```bash
jupyter notebook projetCodeNotebook.ipynb
```
