import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from matplotlib import image as im
import ast
import seaborn as sb
import matplotlib.pyplot as plt

from data import data, read_file

RESULTS_FOLDER = '/content/gdrive/MyDrive/DeXpression/results'


def load_from_array():
    """
    Load dataset from a specified folder
    """
    x = np.load(data("x.npy")).reshape(-1, 1, 224, 224)
    y = np.load(data("y.npy"))

    return x, y

def save_to_array(x, y):
    """
    Save dataset to a specified folder
    """
    with open(data("x.npy"), "wb") as file:
        np.save(file, x)

    with open(data("y.npy"), "wb") as file:
        np.save(file, y)

def load_dataset(use_existing=True):
    """
    Return input and output variables from the
    dataset
    """
    if use_existing:
        x, y = load_from_array()
    else:
        x = []
        y = []

        data_dir = data("images")

        for image_file in sorted(os.listdir(data_dir)):
            image_path = os.path.join(data_dir, image_file)
            label_path = data("labels", f"{image_file[:-4]}.txt")

            if os.path.exists(label_path):
                image, label = read_file(image_path, label_path)
                x.append(image)
                y.append(label)

        x = np.stack(x, axis=0).reshape(-1, 1, 224, 224)
        y = np.stack(y, axis=0)

        save_to_array(x, y)

    print("Loaded datasets {} and {}".format(x.shape, y.shape))
    print("")

    return x, y

def kfold(x, y, splits=5, shuffle=True):
    """
    Perform a K-fold split on the dataset
    x -> Input variables from the dataset
    y -> Output variables from the dataset
    """
    if shuffle:
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(x))
        x = x[shuffled_indices]
        y = y[shuffled_indices]

    fold_size = len(x) // splits

    for i in range(splits):
        start = i * fold_size
        end = (i + 1) * fold_size

        x_train = np.concatenate((x[:start], x[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        x_test = x[start:end]
        y_test = y[start:end]

        yield x_train, y_train, x_test, y_test

def convert_to_torch(x_train, y_train, x_test, y_test):
    """
    Convert the train and test data to torch tensors
    """
    # Convert training images to a torch tensor
    x_train = torch.from_numpy(x_train)
    x_train = x_train.type(torch.FloatTensor)

    # Convert training labels to a torch tensor
    y_train = y_train.astype(int)
    y_train = torch.from_numpy(y_train)

    # Convert test images to torch tensor
    x_test = torch.from_numpy(x_test)
    x_test = x_test.type(torch.FloatTensor)

    # Convert testing labels to torch tensor
    y_test = y_test.astype(int)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test

def initialize():
    """
    Initialize the model and move it to the device
    """
    model = DeXpression()
    model = model.to(device)
    summary(model, (1, 224, 224))

    return model

def plot_confusion_matrix(fold):
    """
    Plot the confusion matrix for a given fold
    """
    filename = results("confusion_matrix_fold_{}.png".format(fold))

    data = pd.read_csv(results("history.csv"))

    test_pred = ast.literal_eval(data.loc[data["fold"] == fold, "test_pred"].values[0])
    test_truth = ast.literal_eval(data.loc[data["fold"] == fold, "test_truth"].values[0])

    cmatrix = confusion_matrix(test_truth, test_pred)
    plot = plot_confusion_matrix(cmatrix, target_names, filename)

def get_test_pred(fold):
    """
    Get the predicted values from the test set
    """
    data = pd.read_csv(results("history.csv"))
    return ast.literal_eval(data.loc[data["fold"] == fold, "test_pred"].values[0])

def get_data(fold):
    """
    Get the test data from the given fold
    """
    data = pd.read_csv(results("history.csv"))
    return ast.literal_eval(data.loc[data["fold"] == fold, "test_data"].values[0])

def get_test_truth(fold):
    """
    Get the ground truth values from the test set
    """
    data = pd.read_csv(results("history.csv"))
    return ast.literal_eval(data.loc[data["fold"] == fold, "test_truth"].values[0])

def get_metric(metric, fold):
    """
    Get the value of a specific metric from the history
    """
    data = pd.read_csv(results("history.csv"))
    return data.loc[data["fold"] == fold, metric].values

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix
    """
    cmatrix = cm(y_true, y_pred)
    return cmatrix

def plot_confusion_matrix(cmatrix, target_names, filename):
    """
    Plot the confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sb.heatmap(cmatrix, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()

def plot_metrics():
    """
    Plot accuracy and loss metrics
    """
    data = pd.read_csv(results("history.csv"))

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(data["epoch"], data["avg_train_accuracy"], label="Train Accuracy")
    plt.plot(data["epoch"], data["avg_test_accuracy"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend(loc="best")
    plt.savefig(results("accuracy.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(data["epoch"], data["avg_train_loss"], label="Train Loss")
    plt.plot(data["epoch"], data["avg_test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend(loc="best")
    plt.savefig(results("loss.png"))
    plt.close()

def results(*paths):
    """
    Return the path to the results folder with the given file paths
    """
    return os.path.join(RESULTS_FOLDER, *paths)
