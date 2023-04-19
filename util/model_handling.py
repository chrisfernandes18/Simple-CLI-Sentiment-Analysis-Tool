''' This file handles the locally saving and loading of the models and their vectorizers. '''
import pickle

def export_clf(file_name, classifier, vectorizer):
    '''
    Compresses the model and vectorizer, saving it locally.

    Parameters
    ----------
    file_name:
        str which is the name of the model.
    classifier:
        MultinomialNB
    vectorizer:
        TfidfVectorizer

    Returns
    -------
    None
    '''
    with open(f'models/{file_name}.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    
    with open(f'models/{file_name}_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print('Model and Vectorizer exported successfully!\n')
    

def import_clf(file_name):
    '''
    Uncompresses the saved model and vectorizer, loading it in.

    Parameters
    ----------
    file_name:
        str which is the name of the model.

    Returns
    -------
    clf:
        MultinomialNB
    vectorizer:
        TfidfVectorizer
    '''
    with open(f'models/{file_name}.pickle', 'rb') as f:
        classifier = pickle.load(f)
    
    with open(f'models/{file_name}_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    return classifier, vectorizer