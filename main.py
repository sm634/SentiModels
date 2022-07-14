from Utils.text_preprocess_pipeline import prepare_tensor_data
from torch import nn
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import time

from model.SentimentCNN import BaseSentimentCNN
from model.DPCNN import DPCNN
from TrainEval import train_sentimentCNN
# from helpers import logger
from config import Config


def main():
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--channel_size', type=int, default=250)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--preprocess', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_set', type=str, default='train.csv')
    parser.add_argument('--valid_set', type=str, default='valid.csv')
    parser.add_argument('--model', type=str, default='BaseSentimentCNN')
    parser.add_argument('--sequence_length', type=int, default=40)
    parser.add_argument('--linear_out', type=int, default=1)
    parser.add_argument('--word_embedding_dimension', type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create the configuration
    config = Config(sequence_length=args.sequence_length,
                    batch_size=args.batch_size,
                    vocab_size=args.vocab_size,
                    learning_rate=args.lr,
                    epoch=args.epoch,
                    linear_out=args.linear_out,
                    word_embedding_dimension=args.word_embedding_dimension)

    #####################
    # load in tensor data
    #####################

    # preprocess csv to prepare tensors for model input
    train_set = args.train_set
    valid_set = args.valid_set

    if args.preprocess:
        print("Preprocessing Started")
        train_loader = prepare_tensor_data(from_path='data/' + train_set, to_path='data/train.pt',
                                           batch_size=config.batch_size, text_col='review', label_col='sentiment',
                                           subsample=True, max_vocab_size=config.vocab_size,
                                           sequence_length=config.sequence_length, drop_last=True)
        valid_loader = prepare_tensor_data(from_path='data/' + valid_set, to_path='data/valid.pt',
                                           batch_size=config.batch_size, text_col='review',
                                           label_col='sentiment', subsample=True, max_vocab_size=config.vocab_size,
                                           sequence_length=config.sequence_length, drop_last=True)
    else:
        # otherwise the file_name passed in the argument should be a .pt file with saved tensors.
        train_loader = DataLoader('data/train.pt', batch_size=config.batch_size, drop_last=True)
        valid_loader = DataLoader("data/valid.pt", batch_size=config.batch_size, drop_last=True)

    #######################
    # Instantiate the model
    #######################

    if args.model.lower() == 'dpcnn':
        model = DPCNN(config)
        print(model)
    else:
        model = BaseSentimentCNN(config)

    # loss and optimization functions
    lr = config.lr  # learning rate to be used for the optimizer.

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #################
    # Train the model
    #################

    if args.model.lower() == 'dpcnn':

        embeds = nn.Embedding(config.vocab_size, config.word_embedding_dimension, padding_idx=0)

        train_loss = 0
        valid_loss = 0
        for epoch in tqdm(range(config.epoch)):
            t1 = time.time()
            # train mode for model to update weights after each backward pass (backprop update)
            model.train()
            try:
                count = 0
                for X, label in train_loader:
                    count += 1
                    # clear the gradients of all optimized variables
                    optimizer.zero_grad()

                    input_data = embeds(X)
                    out = model(input_data)

                    if out.shape != label.shape:
                        out = out.reshape(label.shape[0], )

                    loss = criterion(out, label.float())
                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # perform a single optimization step (parameter update)
                    optimizer.step()

                    train_loss += loss.item() * X.size(0)
                    count += 1
            except IndexError:
                pass
            # change to eval mode for model to freeze weights
            model.eval()

            try:
                for X, label in valid_loader:

                    input_data = embeds(X)
                    out = model(input_data)

                    if out.shape != label.shape:
                        out = out.reshape(label.shape[0], )

                    loss = criterion(out, label.float())
                    # store valid loss.
                    valid_loss += loss.item() * X.size(0)

                # Calculate average losses.
                train_loss = train_loss / (len(train_loader.sampler) * 100)
                valid_loss = valid_loss / (len(valid_loader.sampler) * 100)
                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    epoch, train_loss, valid_loss))

                t2 = (time.time() - t1) / 60
                print("Epoch {}".format(epoch) + " completed in: {:.3f}".format(t2), " minutes")
            except IndexError:
                pass

        print("Done Training")

    else:
        train_sentimentCNN(model, train_loader, valid_loader, criterion, optimizer,
                           save_model_as='imdb_subset_params', n_epochs=config.epoch)


if __name__ == "__main__":
    main()
