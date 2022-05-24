import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import json

from utils import preprocess, recode_sentiment, create_lookup_tables, pad_features
from model.SentimentCNN import SentimentCNN
from TrainEval import train_sentimentCNN, test_sentimentCNN
from helpers import logger


def main():
    # Some hyperparameters that will need to be configured.
    embedding_size = 300
    seq_length = 250
    output_size = 1
    batch_size = 4096

    reviews_data = pd.read_csv('Data/amazon_review_polarity/train.csv', encoding='latin-1')

    reviews_data.columns = ['sentiment', 'title', 'review']

    reviews_data['sentiment'] = reviews_data['sentiment'].apply(lambda x: recode_sentiment(x))

    pos_reviews = reviews_data[reviews_data['sentiment'] == 1].sample(1000000)
    neg_reviews = reviews_data[reviews_data['sentiment'] == 0].sample(1000000)

    reviews_data = pd.concat([pos_reviews, neg_reviews], axis=0).sample(frac=1)

    reviews_data = reviews_data.dropna()

    # get reviews in a list from the pd.Series
    reviews_list = reviews_data['review'].to_list()

    senti_list = reviews_data['sentiment'].to_list()

    # preprocess the reviews in accordance to the preprocess function.
    preprocessed_reviews_list = [preprocess(review) for review in reviews_list]

    # tokenize text and without stemmatizing it to preprocess more
    tokenized_text = ' '.join(
        preprocessed_reviews_list).split()  # stem the tokenized words and replace any extra white spaces.

    ################
    # Encoding words
    ################

    word_count, vocab_int, int_vocab = create_lookup_tables(tokenized_text)

    with open('Data/amazon_review_polarity/Amazon_polarity_subset2m_vocab_to_int.json', 'w') as f:
        json.dump(vocab_int, f)

    with open('Data/amazon_review_polarity/Amazon_polarity_subset2m_int_to_vocab.json', 'w') as fp:
        json.dump(int_vocab, fp)

    # NO LEMMATIZATION: numerical encoding
    reviews_ints = []
    for review in preprocessed_reviews_list:
        reviews_ints.append([vocab_int[word] for word in review.split()])

    indices_to_drop = [i for i, ints in enumerate(reviews_ints) if len(ints) == 0]

    reviews_ints = [review for i, review in enumerate(reviews_ints) if i not in indices_to_drop]
    senti_list = [senti for i, senti in enumerate(senti_list) if i not in indices_to_drop]

    ##########################################
    # hyperparameter config, Padding Sequences
    ##########################################

    review_lens = [len(review) for review in reviews_ints]
    review_len_mean = np.array(review_lens).mean()
    review_len_std = np.array(review_lens).std()

    # Parameters
    try:
        vocab_size = len(vocab_int) + 1  # for the 0 padding + our word tokens
    except:
        vocab_size = vocab_size

    features = pad_features(reviews_ints, seq_length)
    sentiments = np.array(senti_list)

    ####################################################################################
    # split data into training, validation, and test data (features and labels, x and y)
    ####################################################################################

    split_frac = 0.95

    split_idx = int(len(features) * split_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = sentiments[:split_idx], sentiments[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders: make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # save the tensors for later use (if not going to preprocess).

    torch.save(torch.from_numpy(train_x), "Data/amazon_review_polarity/Amazon_polarity_subset2m_trainX.pt")
    torch.save(torch.from_numpy(train_y), "Data/amazon_review_polarity/Amazon_polarity_subset2m_trainy.pt")
    torch.save(torch.from_numpy(val_x), "Data/amazon_review_polarity/Amazon_polarity_subset2m_valX.pt")
    torch.save(torch.from_numpy(val_y), "Data/amazon_review_polarity/Amazon_polarity_subset2m_valy.pt")
    torch.save(torch.from_numpy(test_x), "Data/amazon_review_polarity/Amazon_polarity_subset2m_testX.pt")
    torch.save(torch.from_numpy(test_y), "Data/amazon_review_polarity/Amazon_polarity_subset2m_testy.pt")

    #######################
    # Instantiate the model
    #######################

    model = SentimentCNN(vocab_size, output_size, embedding_size, batch_size, seq_length)
    logger.info(model)

    #################
    # Train the model
    #################

    # loss and optimization functions
    lr = 0.001  # learning rate to be used for the optimizer.

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # #### Import the train model function from TrainTestSentimentCNN ipynb

    train_sentimentCNN(model, train_loader, valid_loader, criterion, optimizer,
                       save_model_as='Sentiment_CNN_subset2m_amazon_pol', n_epochs=5)

    # load the model with the trained parameters/weight that performed best in validation.
    model.load_state_dict(torch.load('Sentiment_CNN_subset2m_amazon_pol.pt'))

    ################
    # TEST THE MODEL
    ################

    test_sentimentCNN(model, test_loader, criterion)


if __name__ == "__main__":
    main()
