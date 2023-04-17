''' This file prints the expected sentiment of a given sentence. '''
from sklearn.metrics import accuracy_score, classification_report

def test_model(classifier, vectorizer, test_input: list, input_sentiment: list):
    '''
    Tests the given sentence on a model and outputs its predicted
    sentiment

    Parameters
    ----------
    classifier:
        Naive Bayes classifier unpickled.
    vectorizer:
        TfidfVectorizer that was used for the trained model.
    test_input:
        list of str which is the sentence given from cli.
    input_sentiment:
        list of str which is optionally given but the expected sentiment result of the sentence.

    Returns
    -------
    None
    '''
    X = vectorizer.transform(test_input)

    # Use the classifier to predict the sentiment of the test data
    y_pred = classifier.predict(X)

    # Print predicted sentiment
    print('--------------------------------')
    print(f'Predicted seniment: {y_pred[0]}')

    # Evaluate the model
    if input_sentiment:
        accuracy = accuracy_score(input_sentiment, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        print(classification_report(input_sentiment, y_pred))