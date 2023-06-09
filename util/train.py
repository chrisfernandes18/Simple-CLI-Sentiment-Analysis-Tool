''' This file handles the training of the model given a dataset. '''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model(dataset):
    '''
    Trains the model given a dataset.

    Parameters
    ----------
    dataset:
        pandas dataframe in the shape (X, 2) where X is any number of rows.

    Returns
    -------
    classifier:
        MultinomialNB
    vectorizer:
        TfidfVectorizer
    '''

    # Load the data
    data = pd.read_csv(f'datasets/{dataset}')

    # Get keys
    keys = data.keys()

    # Extract features from the text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data[keys[0]].values.astype('U'))
    y = data[keys[1]]

    # Create a Naive Bayes classifier and train it on the training data
    classifier = MultinomialNB()
    classifier.fit(X, y)

    return classifier, vectorizer
