"""
Text preprocessor class.
- This will contain 
"""
from typing import List, Union
import numpy as np
from nltk.tokenize import word_tokenize
from string import punctuation
from collections import Counter
import random
from torch.utils.data import TensorDataset, DataLoader
import torch
import itertools


class TextVectorizer:
    """
    TextVectorizer class used for converting text to vectors to prepare for model input.
    """

    def __init__(self):
        self.numbers = [str(n) for n in range(0, 10)]
        self.vocabulary = {"": 0, "<UNK>": 1}
        self.inverse_vocabulary = dict(
            (v, k) for k, v in self.vocabulary.items()
        )
        self.padding_length = 40  # an attribute to pad vectors to a fixed length.
        self.tensor_dataset = None
        self.word_count_dict = {}
        self.total_word_count = 0
        self.unique_word_count = 0
        self.tokenizer = 'nltk'

    def standardize(self, text):
        """
        :Standardize text method: 1. Remove numbers, remove punctuation and lower case text.
        :return: str, standardized text.
        """
        text = ''.join([c for c in text if c not in punctuation])
        text = ''.join([c for c in text if c not in self.numbers])
        text = text.lower()
        return text

    def tokenize(self, text, tokenizer='standard'):
        """
        :Tokenize text into list of words.
        :param text: the input text to be tokenized.
        :param tokenizer: the method of tokenizing the text or sentence. Current options: ['standard', 'nltk']
        :return: List of words.
        """
        text = self.standardize(text)
        if tokenizer == 'standard':
            return text.split()
        elif tokenizer == 'nltk':
            return word_tokenize(text)

    def get_sorted_word_count(self, dataset: Union[List[str], List[List[str]]]):
        """
        Get the word and the number of times it appears in the dataset in descending count order.
        :param dataset: The list of text from the dataset.
        e.g. ['this is text1', 'this is text2']
        """

        all_words = ' '.join(text for text in dataset)
        all_words = self.tokenize(all_words, tokenizer=self.tokenizer)

        word_counts = Counter(all_words)

        self.total_word_count = len(all_words)
        # limit the sub_sampled vocab size to max.
        self.word_count_dict = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))

    def subsample(self, dataset: List[str],
                  threshold=1e-5,
                  max_vocab_size: Union[int, bool] = None) -> List:
        """
        :param dataset: The list of text from the dataset.
        e.g. ['this is text1', 'this is text2',...]
        :Subsample to get rid of most frequent words that add noise to the data and restrict to max_vocab_size.
        :param threshold: The threshold used for the subsampling equation from https://arxiv.org/pdf/1301.3781.pdf.
        :param max_vocab_size: if not None, then restrict the vocab size to the int value supplied.
        :return: List[int]: of words with less noise.
        """
        self.get_sorted_word_count(dataset)

        word_counts = self.word_count_dict
        total_count = self.total_word_count

        # word frequency dict
        frequencies = {
            word: count / total_count
            for word, count in word_counts.items()
        }

        # probability a word will be dropped.
        p_drop = {
            word: 1 - np.sqrt(threshold / frequencies[word])
            for word in word_counts
        }
        sub_sampled_dataset = []
        for text in dataset:
            tokens = self.tokenize(text, tokenizer=self.tokenizer)
            sub_sampled_dataset.append(' '.join([token for token in tokens
                                                 if random.random() < (1 - p_drop[token])]))

        if max_vocab_size is not None and type(max_vocab_size) == int:
            self.get_sorted_word_count(sub_sampled_dataset)
            vocab_count = self.word_count_dict
            vocab = dict(itertools.islice(vocab_count.items(), max_vocab_size))
            self.unique_word_count = len(vocab)

            max_restrict_vocab = []
            for text in sub_sampled_dataset:
                tokens = self.tokenize(text, tokenizer=self.tokenizer)
                max_restrict_vocab.append(' '.join([token for token in tokens if token in vocab])) # need to make
                # more efficient.

            return max_restrict_vocab

        else:
            return sub_sampled_dataset

    def make_vocabulary(self, dataset: List[str],
                        subsample=False,
                        threshold=1e-5,
                        max_vocab_size: Union[int, bool] = None):
        """
        :param dataset: expect dataset to be a list of strings.
        e.g. ['this is text1', 'this is text2',...]
        :param subsample: boolean, indicate whether to subsample dataset vocab or not to a restricted vocab size.
        :Create lookup table to store vocabulary from text data as int.
        :param threshold: threhold to use for subsampling, if enabled.
        :param max_vocab_size: if subsampling, restrict vocab to max size when max_vocab_size is not None.
        :return: dictionary {str: int}, lookup table with word-int mapping.
        """
        if subsample:
            dataset = self.subsample(dataset, threshold=threshold, max_vocab_size=max_vocab_size)

        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text, tokenizer='nltk')
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
            self.inverse_vocabulary = self.inverse_vocabulary

    def encode(self, text):
        """
        Encode the text
        :param text:
        :return:
        """
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence: List[int]):
        return " ".join(
            self.inverse_vocabulary.get(i, "<UNK>") for i in int_sequence
        )

    def pad_features(self, dataset: List[List[int]],
                     sequence_length: int = 40) -> np.array:
        """
        :param dataset: expect dataset to be a list of strings.
        e.g. ['this is text1', 'this is text2',...].
        : Pad the vectors to a fixed length.
        :param sequence_length: the length of the vector to be padded to.
        :return: np.Array, of the integer encodings of words.
        """
        self.padding_length = sequence_length
        features = np.zeros((len(dataset), self.padding_length), dtype=int)
        # for each words vector, pad the vector to sequence_length.
        for i, row in enumerate(dataset):
            features[i, -len(row):] = np.array(row)[:sequence_length]
        return features

    def create_tensor_dataset(self, dataset_x: np.array, dataset_y: np.array):
        """
        :param dataset_x: the dataset, text.
        :param dataset_y: the dataset labels
        :return:
        """
        dataset_x = torch.from_numpy(dataset_x)
        dataset_y = torch.from_numpy(dataset_y)
        self.tensor_dataset = TensorDataset(dataset_x, dataset_y)
        return self.tensor_dataset

    def load_tensor_data(self):
        return DataLoader(self.tensor_dataset)


def recode_sentiment_label(label: str, neg_label='negative', pos_label='positive'):
    if label.lower() == neg_label:
        return 0
    elif label.lower() == pos_label:
        return 1
