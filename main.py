import TrainEval
from Utils.text_preprocess_pipeline import prepare_tensor_data
from Utils.text_preprocessor import TextVectorizer, PrepareTensor
from data.dataset import ImdbReviewsDataset, AmazonReviewPolarity
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from model.SentimentCNN import BaseSentimentCNN
from model.DPCNN import DPCNN
from TrainEval import train_model
import ast
from datetime import datetime
# from helpers import logger
from config import Config


def main():
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--channel_size', type=int, default=250)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--preprocess', type=str, choices=['true', 'false'])
    parser.add_argument('--skip_train', type=str, choices=['true', 'false'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_set', type=str, default='test.csv')
    parser.add_argument('--model', type=str, choices=['BaseSentimentCNN', 'dpcnn'],
                        default='BaseSentimentCNN')
    parser.add_argument('--sequence_length', type=int, default=300)
    parser.add_argument('--linear_out', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--word_embedding_dimension', type=int, default=300)
    parser.add_argument('--dataset', type=str, choices=['imdb_reviews', 'amazon_polarity'], default='imdb_reviews')
    parser.add_argument('--load_specific_param', type=str, choices=['true', 'false'], default='false')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create the configuration for model hyperparameters
    config = Config(sequence_length=args.sequence_length,
                    batch_size=args.batch_size,
                    vocab_size=args.vocab_size,
                    learning_rate=args.lr,
                    epoch=args.epoch,
                    linear_out=args.linear_out,
                    word_embedding_dimension=args.word_embedding_dimension,
                    dropout=args.dropout)

    preprocess = args.preprocess
    skip_train = args.skip_train

    ######################
    # Load desired Dataset
    ######################

    if args.dataset == 'imdb_reviews':
        dataset = ImdbReviewsDataset(file='data/imdb_reviews_data.csv')
    elif args.dataset == 'amazon_polarity':
        dataset = AmazonReviewPolarity(file='data/amazon_review_polarity.csv')
    else:
        raise 'Please choose a dataset from the choices available'

    dataset.train_val_test_split()
    print(args.dataset + 'dataset loaded')
    train_set = dataset.train.dropna(how='any')
    valid_set = dataset.val.dropna(how='any')
    print('train and validation dataset split complete')

    text_col = dataset.review_col
    label_col = dataset.label_col

    # temp condition for amazon polarity dataset
    sample_size_per_polarity = 500000
    if args.dataset == 'amazon_polarity':
        pos_train = train_set.loc[train_set[label_col] == 1].sample(sample_size_per_polarity)
        neg_train = train_set.loc[train_set[label_col] == 0].sample(sample_size_per_polarity)
        train_set = pos_train.append(neg_train).sample(frac=1)

        pos_val = valid_set.loc[valid_set[label_col] == 1].sample(frac=0.5)
        neg_val = valid_set.loc[valid_set[label_col] == 0].sample(frac=0.5)
        valid_set = pos_val.append(neg_val).sample(frac=1)
        print("ama_pol sub train set shape: ", train_set.shape)
        print("ama_pol sub valid set shape: ", valid_set.shape)

    # save model parameters as after training.
    time_stamp = str(datetime.now()).replace(' ', '_t_').replace('.', '_').replace(':', '-')
    save_parameters_as = args.model + '_' + args.dataset + '_' + time_stamp

    # the parameters to load after training.
    load_parameters = save_parameters_as
    # vocabulary lookup dictionary file to save and use
    vocab_lookup_file = 'Utils/vocabulary_lookup' + '_' + args.dataset + '_' + time_stamp + '.json'

    # If training model
    if not skip_train == 'true':

        ###########################################
        # Vectorize text data and get tensor format
        ###########################################

        if preprocess == 'true':

            print("Preprocessing Started")
            train_loader = prepare_tensor_data(input_data=train_set,
                                               to_path='data/train.pt',
                                               path_to_save_vocab=vocab_lookup_file,
                                               batch_size=config.batch_size,
                                               text_col=text_col,
                                               label_col=label_col,
                                               subsample=True,
                                               max_vocab_size=config.vocab_size,
                                               sequence_length=config.sequence_length,
                                               drop_last=True)

            # Preparing Valid set.
            vectorizer = TextVectorizer()

            with open(vocab_lookup_file, 'r') as f:
                vocab_look_up = ast.literal_eval(f.read())

            vectorizer.vocabulary = vocab_look_up

            # encode the model input text col.
            valid_text_encoded = valid_set[text_col].apply(lambda text: vectorizer.encode(text))
            # to have word to int encoding and tokenized into a list.
            valid_text_encoded_list = valid_text_encoded.to_list()  # returns List[List[int]] after above operation.
            # remove empty features.
            valid_label_col = valid_set[label_col].to_list()
            idx_to_drop = [i for i, text in enumerate(valid_text_encoded_list) if len(text) == 0]
            print("no. of text samples prior to dropping empty features: ", len(valid_text_encoded_list),
                  "\nindexes to drop: ", idx_to_drop)
            valid_text_int_encoding = [text for i, text in enumerate(valid_text_encoded_list) if i not in idx_to_drop]
            valid_labels = [label for i, label in enumerate(valid_label_col) if i not in idx_to_drop]
            assert len(valid_text_int_encoding) == len(valid_labels)
            valid_feature_padded = vectorizer.pad_features(valid_text_int_encoding, sequence_length=args.sequence_length)
            valid_labels = np.array(valid_labels)
            print("Preprocessing Complete")

            # initialise TensorPrepare to convert to tensor.
            tensor_prep = PrepareTensor()
            tensor_prep.create_tensor_dataset(valid_feature_padded, valid_labels)
            valid_loader = tensor_prep.load_tensor_data(batch_size=args.batch_size, drop_last=True)

        else:
            # otherwise the file_name passed in the argument should be a .pt file with saved tensors.
            train_loader = DataLoader(torch.load('data/train.pt'), batch_size=config.batch_size, drop_last=True)
            valid_loader = DataLoader(torch.load("data/valid.pt"), batch_size=config.batch_size, drop_last=True)

        print("tensor data loaded")

        #######################
        # Instantiate the model
        #######################

        if args.model.lower() == 'dpcnn':
            model = DPCNN(config)
            print(model)
        else:
            model = BaseSentimentCNN(config)
            print(model)

        ############################
        # loss, optimizer, embedding
        ############################

        # loss and optimization functions
        lr = config.lr  # learning rate to be used for the optimizer.

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #################
        # Train the model
        #################

        train_model(model, train_loader, valid_loader, criterion, optimizer,
                    save_model_as=save_parameters_as, n_epochs=config.epoch)

    ####################
    # Evaluate the model
    ####################

    # Instantiate the model
    if args.model.lower() == 'dpcnn':
        model = DPCNN(config)
        print(model)
    else:
        model = BaseSentimentCNN(config)
        print(model)

    if args.load_specific_param == 'true':
        load_parameters_version = 'model_parameters/dpcnn_amazon_polarity_2022-08-05_t_09-08-23_505992'
        vocab_lookup_file = 'Utils/vocabulary_lookup_amazon_polarity_2022-08-05_t_09-08-23_505992.json'
        model.load_state_dict(torch.load(load_parameters_version))
        print("hard coded version of parameters {} loaded.".format(load_parameters_version))
        with open(vocab_lookup_file, 'r') as f:
            vocab_look_up = ast.literal_eval(f.read())
        print("vocab look up {} loaded".format(vocab_lookup_file))
    else:
        # load the model parameters/weights that minimised validation set loss.
        model.load_state_dict(torch.load('model_parameters/' + load_parameters))
        print("model parameters from {} loaded".format(load_parameters))
        with open(vocab_lookup_file, 'r') as f:
            vocab_look_up = ast.literal_eval(f.read())
        print("vocab look up {} loaded".format(vocab_lookup_file))

    # test data to evaluate model performance against.
    test_set = dataset.test
    test_set = test_set.dropna(axis=0, how='any')

    test_sample = 10000
    if args.dataset == 'amazon_polarity':
        test_set = test_set.iloc[:test_sample, :]
        print(" ama_pol sub test set shape: ", test_set.shape)
    else:
        print(" test_set shape: ", test_set.shape)

    vectorizer = TextVectorizer()
    # get the correct vocabulary look up
    vectorizer.vocabulary = vocab_look_up

    # encode the model input text col.
    test_text_encoded = test_set[text_col].apply(lambda text: vectorizer.encode(text))  # get test set review text
    # to have word to int encoding and tokenized into a list.
    test_text_list = test_text_encoded.to_list()  # returns List[List[int]] after above operation.
    test_feature_padded = vectorizer.pad_features(test_text_list, sequence_length=args.sequence_length)

    # initialise TensorPrepare to convert to tensor.
    tensor_prep = PrepareTensor()
    tensor_prep.create_tensor_dataset(test_feature_padded, test_set[label_col])
    test_loader = tensor_prep.load_tensor_data(batch_size=args.batch_size, drop_last=True)
    print("test set tensor created.")

    eval_metric = TrainEval.test_model(model, test_loader)  # rewrite this function.
    print(eval_metric)  # save the evaluation score along with logger storing model hyperparameters.

    # save model details for reproducibility.
    with open('evaluation/evaluation_metrics_' + args.model + '_' +
              str(datetime.now()).replace(' ', '_t_').replace(':', '-').replace('.', '-') + '.csv', 'w') as f:
        f.write("evaluation metric," + str(eval_metric) + '\n')
        f.write("dataset," + args.dataset + '\n')
        f.write("model," + str(model) + '\n')
        f.write("config," + str({'--lr': args.lr,
                                 '--batch_size': args.batch_size,
                                 '--epoch': args.epoch,
                                 '--vocab_size': args.vocab_size,
                                 '--vocab_lookup': vocab_lookup_file,
                                 '--params': load_parameters,
                                 '--sequence_length': args.sequence_length,
                                 '--linear_out': args.linear_out,
                                 '--word_embedding_dimension': args.word_embedding_dimension,
                                 '--dropout': args.dropout
                                 }
                                ))


if __name__ == "__main__":
    main()
