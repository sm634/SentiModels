import numpy as np
import torch
from tqdm import tqdm
from time import time
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def train_sentimentCNN(model, train_loader, valid_loader, criterion, optimizer,
                       save_model_as: str, n_epochs=5, valid_loss_min=np.Inf):
    """:A simple training procedure for the SentimentCNN model, checking against validation set.
    Return: A set of trained weight/parameter values for for the model
    """

    valid_loss_min = valid_loss_min

    for epoch in tqdm(range(1, n_epochs + 1)):
        t1 = time()

        # keep track of training loss
        train_loss = 0.0
        valid_loss = 0.0

        try:
            ###################
            # train the model #
            ###################
            model.train()
            counter = 0
            for data, target in train_loader:
                # move tensors to GPU if CUDA is available
                counter += 1

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data).reshape(target.shape[0], )
                target = target.to(torch.float32)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)

        except:
            print("train fail index: ", counter)

        t2 = (time() - t1) / 60
        print("Epoch {}".format(epoch) + " completed in: {:.3f}".format(t2), " minutes")

        ######################
        # validate the model #
        ######################
        try:
            model.eval()
            counter = 0
            for data, target in valid_loader:
                counter += 1

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data).reshape(target.shape[0], )
                target = target.to(torch.float32)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)
        except:
            print("valid fail index: ", counter)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), save_model_as + '.pt')
            valid_loss_min = valid_loss

    return None


#############################################
# Test the model against evaluation metrics #
#############################################

def test_sentimentCNN(model, test_loader, criterion):
    """:A simple test procedure for the SentimentCNN model.
    Return: Accuracy score of the model.
    """

    all_preds = torch.empty(0)
    all_targets = torch.empty(0)

    model.eval()
    # iterate over test data
    try:
        counter = 0
        for inputs, target in test_loader:
            counter += 1

            # get predicted outputs
            output = model(inputs).reshape(target.shape[0], )

            target = target.to(torch.float32)  # convert target to float

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer

            # concatenate the preds and labels from the test_loader to get them all
            all_preds = torch.cat((all_preds, pred.cpu()))
            all_targets = torch.cat((all_targets, target.cpu()))

    except:
        print(counter)

    all_targets = all_targets.detach().numpy()
    all_preds = all_preds.detach().numpy()

    # compute accuracy, precision, recall and f1 score by comparing predictions to true label
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro")
    recall = recall_score(all_targets, all_preds, average="macro")
    f1 = f1_score(all_targets, all_preds, average="macro")

    return ("Test Accuracy: {:.4f}".format(accuracy),
            "Precision: {:.4f}".format(precision),
            "Recall: {:.4f}".format(recall),
            "F1: {:.4f}".format(f1)
            )
