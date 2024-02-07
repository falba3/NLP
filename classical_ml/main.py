#!/usr/bin/env python3

import pandas as pd
from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


# Loading the dataset and additional data (from imdb)
df = load_dataset('rotten_tomatoes')
imdb = load_dataset('imdb')

# Splitting train data
X_train = df['train']['text']
y_train = df['train']['label']

# Additional IMDB data
X_train += imdb['train']['text']
y_train += imdb['train']['label']

# Splitting test data
X_test = df['test']['text']
y_test = df['test']['label']


# Fitting the model with best parameters that achieved a high score
tfidf_lr = Pipeline(
    steps=[
        ("vectorizer", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            smooth_idf=True)),

        ("model", LogisticRegression(
            C=1,
            penalty='l2',
            solver='saga'))
    ])

tfidf_lr.fit(X_train, y_train)


def get_score_from_model(fitted_model):
    global df, X_test, y_test

    preds = fitted_model.predict(X_test)
    score = round(f1_score(y_test, preds), 4)

    return score


def make_preds(fitted_model):
    global df, X_test, y_test

    preds = fitted_model.predict(X_test)
    preds_df = pd.DataFrame(preds, columns = ['pred'])
    index_df = pd.DataFrame(list(range(0, len(preds))), columns=['index'])
    preds = pd.concat([index_df, preds_df], axis = 1)

    return preds


def main():
    global df, X_test, y_test, tfidf_lr

    # Getting and printing the f1 score
    score = get_score_from_model(tfidf_lr)
    print(score)

    # Getting predictions and saving into a csv
    preds = make_preds(fitted_model=tfidf_lr)
    preds.to_csv('results.csv')


if __name__ == "__main__":
    main()
