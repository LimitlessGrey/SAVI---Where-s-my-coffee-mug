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

def test():


if __name__ == "__train__":
    train()