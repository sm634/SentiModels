"""A Module that will create a general Base class for labeled reviews text datasets.
The class is fit for reviews datasets such as Imdb, amazon, yelp and tweets commonly used
for sentiment analysis models."""
from torch.utils.data import Dataset
from typing import Union, List, Optional
import pandas as pd


class BaseReviewsDataset(Dataset):
    """Base labeled reviews dataset class."""

    def __init__(self,
                 file,
                 split=Union[bool, List[float]],
                 review_col='review',
                 label_col='sentiment'
                 ):
        """
        :param file: the path to the dataset. Default to the data folder.
        :param split: whether to split the dataset to train, val and test. When only provided bool, the default split
        ration of train:val:test is 0.7:0.1:0.2.
        :param review_col: name of column with reviews.
        :param label_col: name of column with labels/sentiment.
        """
        self.dataset = pd.read_csv(file)
        self.review_col = review_col
        self.label_col = label_col
        self.split = split
        self.labels_dict = {}
        self.train = None
        self.val = None
        self.test = None

    def recode_labels(self, labels_dict=Optional[dict]):

        if len(self.labels_dict) != 0:
            labels_dict = self.labels_dict

        if self.dataset[self.label_col].dtype == str:
            self.dataset[self.label_col] = self.dataset[self.label_col].str.lower()

        self.dataset = self.dataset.replace({self.label_col: labels_dict})

    def train_val_test_split(self):

        if isinstance(self.split, List):

            # first recode labels of positive and negative sentiments are 1 and 0.
            self.recode_labels(self.labels_dict)

            # get a balanced dataset
            positive = self.dataset.loc[self.dataset[self.label_col] == 1]
            negative = self.dataset.loc[self.dataset[self.label_col] == 0]

            assert positive.shape[0] == negative.shape[0]

            # to get a train:val:test split of 0.7:0.1:0.2
            train_n = int(positive.shape[0] * self.split[0])
            valid_n = int(positive.shape[0] * self.split[1])

            self.train = positive.iloc[:train_n, :].append(negative.iloc[:train_n, :]).sample(frac=1)
            self.val = positive.iloc[train_n:(train_n + valid_n), :].append(
                negative.iloc[train_n:(train_n + valid_n), :]).sample(frac=1)
            self.test = positive.iloc[(train_n + valid_n):, :].append(
                negative.iloc[(train_n + valid_n):, :]).sample(frac=1)

        else:

            # first recode labels of positive and negative sentiments are 1 and 0.
            self.recode_labels(self.labels_dict)

            # get a balanced dataset
            positive = self.dataset.loc[self.dataset[self.label_col] == 1]
            negative = self.dataset.loc[self.dataset[self.label_col] == 0]

            assert positive.shape[0] == negative.shape[0]

            # to get a train:val:test split of 0.7:0.1:0.2
            train_n = int(positive.shape[0] * 0.7)
            valid_n = int(positive.shape[0] * 0.1)

            self.train = positive.iloc[:train_n, :].append(negative.iloc[:train_n, :]).sample(frac=1)
            self.val = positive.iloc[train_n:(train_n + valid_n), :].append(
                negative.iloc[train_n:(train_n + valid_n), :]).sample(frac=1)
            self.test = positive.iloc[(train_n + valid_n):, :].append(
                negative.iloc[(train_n + valid_n):, :]).sample(frac=1)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset.iloc[index, :]
