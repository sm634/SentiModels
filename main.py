from Utils.text_preprocess_pipeline import prepare_tensor_data
from torch import nn
import torch
from torch.utils.data import DataLoader
import argparse

from model.SentimentCNN import BaseSentimentCNN
from TrainEval import train_sentimentCNN
# from helpers import logger
from config import Config


def main():
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--preprocess', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_set', type=str, default='train.csv')
    parser.add_argument('--valid_set', type=str, default='valid.csv')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create the configuration
    config = Config(sequence_length=40,
                    batch_size=args.batch_size,
                    vocab_size=args.vocab_size,
                    learning_rate=args.lr,
                    epoch=args.epoch)

    #####################
    # load in tensor data
    #####################

    # preprocess csv to prepare tensors for model input
    train_set = args.train_set
    valid_set = args.valid_set

    if args.preprocess:
        train_loader = prepare_tensor_data(from_path='data/' + train_set, to_path='data/train.pt',
                                           batch_size=config.batch_size, text_col='review', label_col='sentiment',
                                           subsample=True, max_vocab_size=config.vocab_size,
                                           sequence_length=config.sequence_length)
        valid_loader = prepare_tensor_data(from_path='data/' + valid_set, to_path='data/valid.pt',
                                           batch_size=config.batch_size, text_col='review',
                                           label_col='sentiment', subsample=True, max_vocab_size=config.vocab_size,
                                           sequence_length=config.sequence_length)
    else:
        # otherwise the file_name passed in the argument should be a .pt file with saved tensors.
        train_loader = DataLoader(args.file_name, batch_size=config.batch_size, drop_last=True)
        valid_loader = DataLoader(args.valid_set, batch_size=config.batch_size, drop_last=True)

    print('data loaded')

    #######################
    # Instantiate the model
    #######################

    model = BaseSentimentCNN(config)
    print(model)

    # loss and optimization functions
    lr = config.lr  # learning rate to be used for the optimizer.

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #################
    # Train the model
    #################

    train_sentimentCNN(model, train_loader, valid_loader, criterion, optimizer,
                       save_model_as='imdb_subset_params', n_epochs=config.epoch)


if __name__ == "__main__":
    main()
