"""
Run this to prepare a sentiment analysis dataset to feed into model.
: Input will be the text data and its associated labels.
: Returns vector encoded tensors of the data to feed into the model.
    - Future versions: will also include the option to train the embeddings using gensim.
"""
import pandas as pd
import torch
from text_preprocessor import TextVectorizer


def prepare_tensor_data(from_path, to_path, text_col='review', label_col='sentiment'):
    """
    Function to run pipeline that turns text, label dataset -> tensor dataset.
    :param from_path: the path and file to get the input dataset. Currently only reads from .csv file
    extensions.
    :param to_path: the path and file_name to save the output tensor dataset. Save as a .pt file to load using
    torch.utils.data DataLoader.
    :param text_col: the name of the column where the text data is stored.
    :param label_col: the name of the column where the labels are stored. These have to be numerical (int) and the
    pipeline expects any non-numerical encoded labels to be encoded.
    :return: save the output torch.TensorDataset in the provided to_path.
    """
    data = pd.read_csv(from_path, sep=',').dropna(how="any")

    # extract text and label data to work with.
    text_data = data[text_col].to_list()
    labels = data[label_col].to_numpy()

    vectorizer = TextVectorizer()
    # When creating vocab out of tokenizing, use nltk word_tokenizer.
    vectorizer.tokenizer = 'nltk'
    # create vocabulary using the text data.
    vectorizer.make_vocabulary(text_data, subsample=True, max_vocab_size=40000)

    # encode the text data into ints from the created vocabulary lookup table.
    text_int_encoding = [vectorizer.encode(text) for text in text_data]

    # pad the text_int_encoding.
    text_int_encoding = vectorizer.pad_features(text_int_encoding, sequence_length=40)

    # create the tensor dataset and save it in to_path.
    torch.save(vectorizer.create_tensor_dataset(text_int_encoding, labels), to_path)

    return vectorizer.load_tensor_data()
