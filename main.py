import os
from google.colab import drive
import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as s
import pandas as pd
from datetime import datetime
from pathlib import Path
from matplotlib import image as im
import glob
import numpy as np
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, LogSoftmax
from torch.nn import LayerNorm, BatchNorm2d, Dropout
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix as cm
import ast
import seaborn as sb
import matplotlib.pyplot as plt

from utils import load_from_array, save_to_array, load_dataset, kfold, convert_to_torch, initialize, train, test, print_progress, save_progress, dump_dict_list, get_test_pred, get_data, get_test_truth, get_metric, confusion_matrix, plot_confusion_matrix, plot_metrics

from model import DeXpression

drive.mount('/content/gdrive')

path = '/content/gdrive/MyDrive/DeXpression'
os.chdir(path)

DATA_FOLDER = '/content/gdrive/MyDrive/DeXpression/data/'
RESULTS_FOLDER = '/content/gdrive/MyDrive/DeXpression/results'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    x, y = load_dataset()
    folds = kfold(x, y)

    for fold, (x_train, y_train, x_test, y_test) in enumerate(folds):
        x_train, y_train, x_test, y_test = convert_to_torch(x_train, y_train, x_test, y_test)

        model = initialize()
        run_model(fold, model, x_train, y_train, x_test, y_test)

        dump_dict_list(res)

        print("Start plotting")
        print("")

        for fold in range(5):
            plot_confusion_matrix(fold+1)
        plot_metrics()

if __name__ == "__main__":
    main()

