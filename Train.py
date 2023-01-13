#!/usr/bin/env python3

import argparse
import glob
import pickle
import random
from copy import deepcopy
from statistics import mean

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import Model
from dataset import Dataset
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from torchvision import transforms
from data_visualizer import DataVisualizer
from classification_visualizer import ClassificationVisualizer

def train():
    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    # Define hyper parameters
    resume_training = True
    model_path = 'model.pkl'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # cuda: 0 index of gpu

    model = Model()  # Instantiate model

    learning_rate = 0.001
    maximum_num_epochs = 50
    termination_loss_threshold = 0.0001
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    test_visualizer = ClassificationVisualizer('Test Images')

    # Resume training
    if resume_training:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        idx_epoch = checkpoint['epoch']
        epoch_train_losses = checkpoint['train_losses']
        epoch_test_losses = checkpoint['test_losses']

        # model.train()
    else:
        idx_epoch = 0
        epoch_train_losses = []
    # -----------

    model.to(device)  # move the model variable to the gpu if one exists
    while True:
        # Train batch by batch -----------------------------------------------
        train_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_train), total=len(loader_train),
                                                  desc=Fore.GREEN + 'Training batches for Epoch ' + str(
                                                      idx_epoch) + Style.RESET_ALL):
            image_t = image_t.to(device)
            label_t = label_t.to(device)

            # Apply the network to get the predicted ys
            label_t_predicted = model.forward(image_t)

            # Compute the error based on the predictions
            loss = loss_function(label_t_predicted, label_t)

            # Update the model, i.e. the neural network's weights
            optimizer.zero_grad()  # resets the weights to make sure we are not accumulating
            loss.backward()  # propagates the loss error into each neuron
            optimizer.step()  # update the weights

            train_losses.append(loss.data.item())

        # Compute the loss for the epoch
        epoch_train_loss = mean(train_losses)
        epoch_train_losses.append(epoch_train_loss)

        loss_visualizer = DataVisualizer('Loss')
        loss_visualizer.draw([0, maximum_num_epochs], [termination_loss_threshold, termination_loss_threshold],
                             layer='threshold', marker='--', markersize=1, color=[0.5, 0.5, 0.5], alpha=1,
                             label='threshold', x_label='Epochs', y_label='Loss')

        loss_visualizer.draw(list(range(0, len(epoch_train_losses))), epoch_train_losses, layer='train loss',
                             marker='-', markersize=1, color=[0, 0, 0.7], alpha=1, label='Train Loss', x_label='Epochs',
                             y_label='Loss')

if __name__ == "__train__":
    train()