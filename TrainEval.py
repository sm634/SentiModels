import numpy as np
import torch
from tqdm import tqdm
from time import time
import os
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score
)

# added to enable directory navigation when running the main.py script.
main_script_path = os.getcwd().replace('\\', '/') + '/'


def train_model(model, train_loader, valid_loader, criterion, optimizer,
                save_model_as: str, embeds=None, n_epochs=5, valid_loss_min=np.Inf):
    """:A simple training procedure for the SentimentCNN model, checking against validation set.
    Return: A set of trained weight/parameter values for for the model
    """

    valid_loss_min = valid_loss_min

    for epoch in tqdm(range(1, n_epochs + 1)):
        t1 = time()

        # keep track of training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        counter = 0

        for X, target in train_loader:
            try:
                # move tensors to GPU if CUDA is available
                counter += 1

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                if embeds is not None:
                    # specify the embedding matrix as input.
                    input_data = embeds(X)
                    output = model(input_data)
                else:
                    output = model(X)

                if output.shape != target.shape:
                    output = output.reshape(target.shape[0], )

                # calculate the batch loss
                loss = criterion(output, target.float())
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * X.size(0)
            except IndexError as x:
                print("{} error occurred at batch {}".format(x, counter))
                pass

        ######################
        # validate the model #
        ######################
        model.eval()
        counter = 0

        for X, target in valid_loader:
            try:
                counter += 1

                if embeds is not None:
                    # specify the embedding matrix as input.
                    input_data = embeds(X)
                    output = model(input_data)
                else:
                    output = model(X)

                if output.shape != target.shape:
                    output = output.reshape(target.shape[0], )

                # calculate the batch loss
                loss = criterion(output, target.float())
                # update average validation loss
                valid_loss += loss.item() * X.size(0)
            except IndexError as x:
                print("{} error occurred in batch {}".format(x, counter))
                pass

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        # logger.info training/validation statistics
        print(' Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), main_script_path + 'model_parameters/' + save_model_as)
            valid_loss_min = valid_loss

        t2 = (time() - t1) / 60
        print("Epoch {}".format(epoch) + " completed in: {:.3f}".format(t2), " minutes")
    return None


#############################################
# Test the model against evaluation metrics #
#############################################

def test_model(model, test_loader, embeds=None):
    """:A simple test procedure for the SentimentCNN model.
    Return: Accuracy score of the model.
    """

    all_predictions = torch.empty(0)
    all_targets = torch.empty(0)

    model.eval()
    # iterate over test data
    counter = 0

    for X, target in test_loader:
        try:
            counter += 1

            if embeds is not None:
                # get predicted outputs
                input_data = embeds(X)
                output = model(input_data)
            else:
                output = model(X)

            if output.shape != target.shape:
                output = output.reshape(target.shape[0], )

            target = target.float()  # convert target to float

            # convert output probabilities to predicted class (0 or 1)
            # prediction = torch.round(output.squeeze())  # rounds to the nearest integer
            prediction = torch.round(output)

            # concatenate the predictions and labels from the test_loader to get them all
            all_predictions = torch.cat((all_predictions, prediction.cpu()))
            all_targets = torch.cat((all_targets, target.cpu()))
        except IndexError as x:
            print("{} at batch {}".format(x, counter))
            pass

    all_targets = all_targets.detach().numpy()
    all_predictions = all_predictions.detach().numpy()

    # compute accuracy, precision, recall and f1 score by comparing predictions to true label
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_predictions, average="macro")
    f1 = f1_score(all_targets, all_predictions, average="macro")

    return ("Test Accuracy: {:.4f}".format(accuracy),
            "Precision: {:.4f}".format(precision),
            "Recall: {:.4f}".format(recall),
            "F1: {:.4f}".format(f1)
            )
