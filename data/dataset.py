from torch.utils.data import Dataset
from typing import Union, List
import pandas as pd
import os
import re


class TextDataset(Dataset):
    def __init__(self,
                 path,
                 labels: Union[int, List[int]],
                 split: str = "train"
                 ):

        self.labels = labels
        self.filename = os.listdir(path)
        if split.lower() == "train":
            self.train_set = pd.read_csv(path + [file for file in self.filename if re.search('train', file)][0])
        if split.lower() == "test":
            self.test_set = pd.read_csv(path + [file for file in self.filename if re.search('test', file)][0])
        if split.lower() == "val":
            self.val_set = pd.read_csv(path + [file for file in self.filename if re.search('val', file)][0])

        """
        : param path: the path to the data directory
        : param labels: expects type int or list of ints as label
        : param split: get either the train, test or val set. Currently expects a directory for a dataset with three
        separate files for train.csv, test.csv and valid.csv.
        """

    def __getitem__(self, index):
        return self.train_set[index], self.labels[index]

    def __len__(self):
        return len(self.train_set)