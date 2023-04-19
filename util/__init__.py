import os
from .model_handling import export_clf, import_clf
from .train import train_model
from .test import test_model

MODELS = filter(lambda x: x != '', [ '' if 'vectorizer.pickle' in file or '.md' in file else file.replace('.pickle', '') for file in os.listdir('models')])