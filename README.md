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

## ⚠️ Why the accuracy is ~99% — this is a dataset artifact, not a solved NLP problem

These scores look impressive, but they're inflated by systematic differences between the two CSV files that have nothing to do with whether an article is actually true or false:

- **Source fingerprint**: almost every article in `True.csv` starts with an agency dateline, e.g. `"WASHINGTON (Reuters) - ..."`, `"SEATTLE/WASHINGTON (Reuters) - ..."`. None of the `Fake.csv` articles have this pattern. A model can learn to spot the string `"(Reuters)"` alone and get most of the way to 99% — it's classifying *writing style / source*, not *truthfulness*.
- **Disjoint `subject` categories**: `True.csv` only contains `politicsNews` and `worldnews`, while `Fake.csv` only contains `News`, `politics`, `left-news`, `Government News`, `US_News`, and `Middle-east`. The two classes never share a subject label, so `subject` alone is close to a perfect predictor and leaks directly into the TF-IDF features (subject-related vocabulary, formatting, etc.).
- **Different collection process**: the two files were scraped from different sources/time periods, which introduces other stylistic and formatting artifacts (punctuation habits, capitalization, article length, boilerplate) that a bag-of-words/TF-IDF model latches onto easily.

In short, the models aren't learning to detect *misinformation* — they're learning to detect *which of the two source files a snippet of text came from*. That's a much easier task than genuine fake-news detection, which is why even simple linear models cross 99%, and it's also why this result wouldn't generalize to real-world, out-of-distribution news text.

**To get a more honest measure of performance**, this pipeline could be extended to:
- Strip source-identifying boilerplate (agency datelines) before training.
- Drop or balance the `subject` column so it can't act as a shortcut.
- Test on a separate, independently collected fake/real news dataset (out-of-distribution evaluation).

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