
import pandas as pd
from sklearn.metrics import f1_score
import pickle

def get_score_from_model(model):

    # Loading the test data
    test = pd.read_csv('data/test.parquet')

    # Splitting test data
    X_test = test['text']
    y_test = test['label']

    preds = model.predict(X_test)
    score = round(f1_score(y_test, preds), 4)

    return score


def main():
    score = get_score_from_model('models/tfidf_lr_81.pkl')
    print(score)