"""
Run this to prepare a sentiment analysis dataset to feed into model.
: Input will be the text data and its associated labels.
: Returns vector encoded tensors of the data to feed into the model.
    - Future versions: will also include the option to train the embeddings using gensim.
"""
import pandas as pd
import torch
from Utils.text_preprocessor import TextVectorizer
import os

root_dir = os.getcwd().replace('\\', '/') + '/'


def prepare_tensor_data(from_path, to_path, batch_size, text_col='review', label_col='sentiment', subsample=True,
                        max_vocab_size=20000, sequence_length=40, tokenizer='nltk'):
    """
    Function to run pipeline that turns text, label dataset -> tensor dataset.
    :param from_path: the path and file to get the input dataset. Currently only reads from .csv file
    extensions.
    :param to_path: the path and file_name to save the output tensor dataset. Save as a .pt file to load using
    torch.utils.data DataLoader.
    :param batch_size: batch size of the tensor data to be loaded.
    :param text_col: the name of the column where the text data is stored.
    :param label_col: the name of the column where the labels are stored. These have to be numerical (int) and the
    pipeline expects any non-numerical encoded labels to be encoded.
    :param subsample: Indicate if subsampling the vocab size to get rid of frequent but noisy words (e.g. 'and', 'is')
    :param max_vocab_size: The maximum number of unique word tokens to be used, dependent on subsample=True.
    :param sequence_length: The length of the vector, once converted from text.
    :param tokenizer: the type of tokenizer to be used. Currently two options: ['standard', 'nltk'].
    :return: save the output torch.TensorDataset in the provided to_path.
    """
    data = pd.read_csv(from_path, sep=',').dropna(how="any")

    # extract text and label data to work with.
    text_data = data[text_col].to_list()
    labels = data[label_col].to_numpy()

    vectorizer = TextVectorizer()
    # When creating vocab out of tokenizing, use nltk word_tokenizer.
    vectorizer.tokenizer = tokenizer
    # create vocabulary using the text data.
    vectorizer.make_vocabulary(text_data, subsample=subsample, max_vocab_size=max_vocab_size)

    # encode the text data into ints from the created vocabulary lookup table.
    text_int_encoding = [vectorizer.encode(text) for text in text_data]

    # pad the text_int_encoding.
    text_int_encoding = vectorizer.pad_features(text_int_encoding, sequence_length=sequence_length)

    # create the tensor dataset and save it in to_path.
    torch.save(vectorizer.create_tensor_dataset(text_int_encoding, labels), to_path)

    return vectorizer.load_tensor_data(batch_size=batch_size)
