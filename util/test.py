from sklearn.metrics import accuracy_score, classification_report

def test_model(classifier, vectorizer, test_input: list, input_sentiment: list):

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