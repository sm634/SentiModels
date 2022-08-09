from typing import Union, List
from data.labeled_reviews_data import BaseReviewsDataset
import pandas as pd


class ImdbReviewsDataset(BaseReviewsDataset):
    """Imdb movies reviews dataset."""

    def __init__(self, file='imdb_reviews_data.csv', split=Union[bool, List[float]],
                 review_col='review', label_col='sentiment'):
        """
        :param file: the path to the dataset. Default to the data folder.
        :param split: whether to split the dataset to train, val and test. When only provided bool, the default split
        ration of train:val:test is 0.8:0.1:0.1.
        :param review_col: name of column with reviews.
        :param label_col: name of column with labels/sentiment.
        """
        super().__init__(file, split, review_col, label_col)
        self.dataset = pd.read_csv(file)
        self.labels_dict = {'negative': 0, 'positive': 1}

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset.iloc[index, :]


class AmazonReviewPolarity(BaseReviewsDataset):
    """Amazon Products reviews polarity dataset."""

    def __init__(self, file='amazon_review_polarity.csv', split=Union[bool, List[float]],
                 review_col='review', label_col='sentiment'):
        """
        :param file: the path to the dataset. Default to the data folder.
        :param split: whether to split the dataset to train, val and test. When only provided bool, the default split
        ration of train:val:test is 0.8:0.1:0.1.
        :param review_col: name of column with reviews.
        :param label_col: name of column with labels/sentiment.
        """
        super().__init__(file, split, review_col, label_col)
        self.dataset = pd.read_csv(file, names=[label_col, 'title', review_col])
        self.dataset = self.dataset[[label_col, review_col]]
        self.labels_dict = {1: 0, 2: 1}

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset.iloc[index, :]


# Use case example of the dataset class
""" 
imdb_reviews = ImdbReviewsDataset()
imdb_reviews_data = imdb_reviews.dataset
imdb_reviews.recode_labels()
imdb_reviews.train_val_test_split()
train_data = imdb_reviews.train
"""
