# Datasets

Here are where one should put their datasets and make sure that the name of the dataset is what you want the name of the model to be plus file type. Example:

```
$ IMDB.csv
```

will give a model that can be called from the command line as

```
$ IMDB
```

Below are linked some example datasets which can be downloaded to use for training the model.

- [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Amazon - Electronics](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- [Amazon - Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- [Twitter](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Dataset Setup

Make sure that the datasets put in here are in the following 2 column format and file type `.csv`:

| text       | sentiment |
| ---------- | --------- |
| I love you | positive  |
| I hate you | negative  |

- `text` will be the strings that we are going to give a sentiment to.

- `sentiment` will have whatever outputs you want them to be I just have `positive` and `negative` for all my datasets for simplicity.