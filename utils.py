import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from string import punctuation
from collections import Counter
import random


# a function that takes in raw text and retuns tokenized and lemmatized list of words after applying a number
# preprocessing steps mentioned in paper:
def preprocess(text: str) -> str:
    """
    Preprocessing with the following:
    1. remove punctuation, 2. remove numbers 3. lower case
    : return: a list of tokenized and lemmatized words.
    """
    numbers = [str(n) for n in range(0, 10)]
    text = ''.join([c for c in text if c not in punctuation])  # remove punctuation
    text = ''.join([c for c in text if c not in numbers])  # remove numbers
    preprocessed_text = text.lower()  # lower case the text.

    return preprocessed_text


def tokenize_lemmatize(text: str) -> list:
    """
    :Tokenize text into list of words.
    :Lemmatize these words to standardise them into their roots.
    :Remove any spaces within a stored word lemma for standardisation.
    Return: a standardised, lemmatized list of tokenized words.
    """
    ps = PorterStemmer()

    tokenized_text = word_tokenize(text)  # returns a tokenized list of words from text.
    tokenized_text = [ps.stem(word) for word in
                      tokenized_text]  # returns a lemmatization of the tokenized text (effectively reducing vocab)
    tokenized_text = [word.replace(' ', '') for word in tokenized_text]
    return tokenized_text


def recode_sentiment(val):
    if val == 1:
        return 0
    else:
        return 1


def subsample(words: iter, threshold=1e-5) -> iter:
    """Subsampling in order to get rid of most frequent words that add noise to the data.
    :words: [iter] a list-like structure of words"""

    word_counts = Counter(words)
    total_count = len(words)

    freqs = {word: count / total_count for word, count in word_counts.items()}
    # the probability that a word will be dropped from the paper,
    # 'Efficient Estimation of Word representation in vector space':-
    """https://arxiv.org/pdf/1301.3781.pdf"""

    p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
    # discard some frequent words, according to the subsampling equation
    # create a new list of words for training
    train_words = [word for word in words if random.random() < (1 - p_drop[word])]
    return train_words


def create_lookup_tables(words: iter, subsampling=False) -> tuple:
    """
    Create lookup words for vocabulary.
    :the 'words' argument or parameter: takes in a list of words
    :subsampling - if true, apply subsampling to the word list to remove some 'noisy' words.
    Return: Three dictionaries, vocab_int and int_vocab
    """
    word_count = Counter(words)
    # sorting the frequency of words from highest to lowest in occurrence.
    sorted_word_count = sorted(word_count, key=word_count.get, reverse=True)
    # creating dicts that has key-value pairs of word-count and count-word.
    vocab_int = {word: (ii + 2) for ii, word in
                 enumerate(sorted_word_count)}  # plus two to index to allow for 'unk' and padding features with 0.
    int_vocab = {(ii + 2): word for ii, word in enumerate(sorted_word_count)}

    # add the unknown word to dict
    vocab_int['<unk>'] = 1
    int_vocab[1] = '<unk>'

    return word_count, vocab_int, int_vocab


def word_to_int(input_text: iter, vocab_to_int: dict, token_lem=False) -> iter:
    """
    A function to be used for encoding text to integers for prediction (assigning unknown words).
    Return: list of integers representing words in text
    """
    if token_lem == True:
        standardised_text = tokenize_lemmatize(input_text)
    else:
        standardised_text = input_text.split()
    # Convert words not in lookup to '<unk>'.

    word_ints = []
    for ii, word in enumerate(standardised_text):
        try:
            word_ints.append(vocab_to_int[word])
        except KeyError:
            word_ints.append(1)
        # assign the text integer values.
    return word_ints


def pad_features(reviews_ints: iter, seq_length: int):
    '''
    : Take in a list of words encoded as integers, list length parametarised by 'seq_length',
    then return them as input numpy array features for the model.
    Return: features of review_ints, where each review is padded with 0's
    or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def sentiment_predict(model, new_texts, vocab_to_int, seq_length=40):
    """
    Function that takes in text, preproceses and passes it to the model for forward pass.
    Args:
     - model to perform the inference
     - input text
     - word to integer mapping dict
     - sequence length the text is padded to
    :Returns a thresholded sigmoid collapsed output for positive, negative or neutral sentiment."""

    model.eval()

    # preprocess, tokenize and lemmatize review
    new_texts = preprocess(new_texts)
    new_texts_ints = word_to_int(new_texts, vocab_to_int, token_lem=True)

    # pad tokenized sequence
    features = np.zeros((seq_length), dtype=int)
    if features.shape[0] >= len(new_texts_ints):
        features[seq_length - len(new_texts_ints):] = np.array(new_texts_ints)[:seq_length]
    else:
        features[:] = np.array(new_texts_ints)[:seq_length]

    input_tensor = torch.from_numpy(features)

    # perform a forward pass from the model
    output = model(input_tensor)

    pred = output.detach().numpy()[0][0]
    if pred >= 0.55:
        return ("positive, {:.4f}".format(2 * pred - 1))
    elif pred < 0.45:
        return ("negative, {:.4f}".format(2 * pred - 1))
    else:
        return ("neutral, {:.4f}".format(2 * pred - 1))


def class_predict(model, new_texts, vocab_to_int, seq_length=40):
    """
    Function that takes in text, preproceses and passes it to the model for forward pass.
    Args:
     - model to perform the inference
     - input text
     - word to integer mapping dict
     - sequence length the text is padded to
    :Returns a score of positive (1) or negative (0) sentiment for class prediction."""

    model.eval()

    # preprocess, tokenize and lemmatize review
    new_texts = preprocess(new_texts)
    new_texts_ints = word_to_int(new_texts, vocab_to_int, token_lem=True)

    # pad tokenized sequence
    features = np.zeros((seq_length), dtype=int)
    if features.shape[0] >= len(new_texts_ints):
        features[seq_length - len(new_texts_ints):] = np.array(new_texts_ints)[:seq_length]
    else:
        features[:] = np.array(new_texts_ints)[:seq_length]

    input_tensor = torch.from_numpy(features)

    # perform a forward pass from the model
    output = model(input_tensor)

    pred = output.detach().numpy()[0][0]
    if pred >= 0.5:
        return 1
    elif pred < 0.5:
        return 0
