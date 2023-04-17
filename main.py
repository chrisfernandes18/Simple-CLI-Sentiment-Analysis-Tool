from util import terminal, test_model, train_model, export_clf, import_clf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    args = terminal.get_input()
    if args.dataset:
        model_name = args.dataset.split(".")[0]
        print(f'Going to train model which will be called: {model_name}')
        classifier, vectorizer = train_model(args.dataset)
        export_clf(model_name, classifier, vectorizer)
    
    if args.sentence and args.model:
        print('Going to predict sentiment of given sentence.')
        classifier, vectorizer = import_clf(args.model)
        test_model(classifier, vectorizer, [" ".join(args.sentence)], args.sentiment)

if __name__ == '__main__':
    main()
