import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime

from utils import results

def dump_dict_list(history):
    """
    Convert history to a pandas dataframe
    and store it in the results folder
    """
    filename = results("history.csv")

    print("Saving history {}".format(filename))
    print("")

    df = pd.DataFrame(history)
    df.to_csv(filename)


def print_progress(fold, epoch, n_epochs, avg_train_accuracy, avg_train_loss, avg_test_accuracy, avg_test_loss):
    """
    Print training and testing performance
    per epoch
    """
    print("Fold: %d, Epoch: %d/%d" % (fold + 1, epoch + 1, n_epochs))
    print("Train accuracy: %.2f%%" % (avg_train_accuracy * 100))
    print("Train loss: %.3f" % (avg_train_loss))
    print("Test accuracy: %.2f%%" % (avg_test_accuracy * 100))
    print("Test loss: %.3f" % (avg_test_loss))
    print("")


def save_progress(fold, epoch, avg_test_accuracy, model, model_optimizer):
    """
    Save a model checkpoint every epoch
    """
    checkpoint = "{:d}-{:.2f}.tar".format(epoch + 1, avg_test_accuracy)

    print("Saving checkpoint {}".format(results(checkpoint)))
    print("")

    #```python
    # save in a dictionary
    torch.save(
        {
            "fold": fold + 1,
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "model_opt": model_optimizer.state_dict(),
        },
        results(checkpoint),
    )

def train(x_batch, y_batch, model, criterion, model_optimizer):
    """
    Perform a single forward and back propagation step
    on the training data
    """

    model_optimizer.zero_grad()  # remove any existing gradients

    # forward propagate
    output, _ = model(x_batch)
    _, y_pred = torch.max(output.data, 1)
    _, y_truth = torch.max(y_batch, 1)

    # compute model loss
    loss = criterion(output, y_truth)

    # backpropagate the gradients
    loss.backward()

    # update parameters based on backprop
    model_optimizer.step()

    # accuracy
    correct_counts = y_pred.eq(y_truth.data.view_as(y_pred))

    # average accuracy
    accuracy = torch.mean(correct_counts.type(torch.FloatTensor))

    return accuracy, loss


def test(x_batch, y_batch, model, criterion):
    """
    Perform a single forward propagation step
    on the testing data
    """
    # forward propagate
    output, _ = model(x_batch)
    _, y_pred = torch.max(output.data, 1)
    _, y_truth = torch.max(y_batch, 1)

    # compute model loss
    loss = criterion(output, y_truth)

    # compute validation accuracy
    correct_counts = y_pred.eq(y_truth.data.view_as(y_pred))

    # mean validation accuracy
    accuracy = torch.mean(correct_counts.type(torch.FloatTensor))

    # predicted and ground truth values converted to a list
    y_pred = y_pred.tolist()
    y_truth = y_truth.tolist()

    return accuracy, loss, y_pred, y_truth


def run_model(fold, model, x_train, y_train, x_test, y_test, batch_size=32, n_epochs=2, learning_rate=0.001):

    global history  # for model checkpoints

    # loss function and optimizer
    criterion = nn.NLLLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        running_train_accuracy = 0
        running_train_loss = 0

        running_test_accuracy = 0
        running_test_loss = 0

        n_iters_train = len(x_train) / batch_size
        n_iters_test = len(x_test) / batch_size

        # train the model (for each mini-batch)
        model.train()
        for index in range(0, len(x_train), batch_size):
            x_batch = x_train[index : index + batch_size]
            y_batch = y_train[index : index + batch_size]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            train_accuracy, train_loss = train(
                x_batch, y_batch, model, criterion, model_optimizer
            )

            # update metrics
            running_train_accuracy += train_accuracy.item()
            running_train_loss += train_loss.item()

        # test the model
        with torch.no_grad():
            model.eval()

            test_pred, test_truth = [], []

            for index in range(0, len(x_test), batch_size):
                x_batch = x_test[index : index + batch_size]
                y_batch = y_test[index : index + batch_size]
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                test_accuracy, test_loss, pred, truth = test(
                    x_batch, y_batch, model, criterion
                )

                # update metrics
                running_test_accuracy += test_accuracy.item()
                running_test_loss += test_loss.item()

                # append to the end of the prediction and ground truth lists
                test_pred.extend(pred)
                test_truth.extend(truth)

        # mean metrics
        avg_train_accuracy = running_train_accuracy / n_iters_train
        avg_train_loss = running_train_loss / n_iters_train

        avg_test_accuracy = running_test_accuracy / n_iters_test
        avg_test_loss = running_test_loss / n_iters_test

        # append checkpoints to model history
        history.append(
            {
                "fold": fold + 1,
                "epoch": epoch + 1,
                "avg_train_accuracy": avg_train_accuracy * 100,
                "avg_test_accuracy": avg_test_accuracy * 100,
                "avg_train_loss": avg_train_loss,
                "avg_test_loss": avg_test_loss,
                "test_pred": test_pred,
                "test_truth": test_truth,
            }
        )

        # print progress
        print_progress(
            fold,
            epoch,
            n_epochs,
            avg_train_accuracy,
            avg_train_loss,
            avg_test_accuracy,
            avg_test_loss,
        )

        # save progress
        save_progress(fold, epoch, avg_test_accuracy, model, model_optimizer)
