''' This is the main file which handles the overall logic of the repo. '''
import sys
from util import terminal, test_model, train_model, export_clf, import_clf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser, args = terminal.get_input()

    # if no arguments were given on cli print help menu
    if not len(sys.argv) > 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    print('----------------------------------------------------------\n')
    
    # if we are going to train a dataset
    if args.dataset:
        model_name = args.dataset.split(".")[0]
        print(f'Going to train model which will be called: {model_name}...\n')
        classifier, vectorizer = train_model(args.dataset)
        export_clf(model_name, classifier, vectorizer)
    
    # if we are going to try to predict the sentiment of something
    if args.sentence and args.model:
        print('Going to predict sentiment of given sentence...\n')
        print('"' + ' '.join(args.sentence) + '"\n')
        classifier, vectorizer = import_clf(args.model)
        test_model(classifier, vectorizer, [" ".join(args.sentence)], args.sentiment)

    print('----------------------------------------------------------')

if __name__ == '__main__':
    main()
