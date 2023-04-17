import pickle

def export_clf(file_name, classifier, vectorizer):
    with open(f'models/{file_name}.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    
    with open(f'models/{file_name}_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print('Model and Vectorizer exported successfully!')
    

def import_clf(file_name):
    with open(f'models/{file_name}.pickle', 'rb') as f:
        clf = pickle.load(f)
    
    with open(f'models/{file_name}_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print('Imported and loaded classifier and vectorizer successfully!')

    return clf, vectorizer