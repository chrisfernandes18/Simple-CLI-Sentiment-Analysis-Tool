''' This file handles the CLI arguments. '''
import argparse
from . import MODELS

def get_input() -> argparse.Namespace: 
    '''
    Returns a namespace argparse object containing all the command line arguments
    given.

    Example
    -------
    python3 main.py S -m MODEL_NAME

    Parameters
    ----------
    None

    Returns
    -------
    args
        Namespace

    '''
    parser = argparse.ArgumentParser(description='Predict sentiment of a given sentence on a trained model.')
    parser.add_argument('sentence', type=str, metavar='S', nargs='*', help='Sentence to give')
    parser.add_argument('-s', dest='sentiment', type=str, help='Optional: check correctness of the sentiment model.')
    parser.add_argument('-m', dest='model', type=str, help=f'Model name - {", ".join(MODELS)}')
    parser.add_argument('-td', dest='dataset', type=str, help='Train dataset using given dataset.')
    args = parser.parse_args()
    return args