#!/usr/bin/env python3

import numpy as np
import os
import cv2
from model import RGBDClassifier
import torch
import glob
from sklearn.model_selection import train_test_split
import pickle
import random

def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    resume_training = True
    model_path = 'model.pkl'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model = RGBDClassifier()  # Instantiate model

    learning_rate = 0.001
    maximum_num_epochs = 50
    termination_loss_threshold = 0.01
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------

    # Load the dataset from disk
    dataset_path = '...' # TODO: Insert path
    image_filenames = glob.glob(dataset_path + '/*.jpg')

    # Sample only a few images to speed up development
    image_filenames = random.sample(image_filenames, k=300)

    # split images into train and test
    train_image_filenames, test_image_filenames = train_test_split(image_filenames, test_size=0.2)

    dataset_train = Dataset(train_image_filenames)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)

    dataset_test = Dataset(test_image_filenames)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=256, shuffle=True)


    # -----------------------------------------------------------------
    # Get Point Cloud
    # -----------------------------------------------------------------



    # -----------------------------------------------------------------
    #
    # -----------------------------------------------------------------

    color_images = []
    depth_images = []

    for filename in os.listdir(dataset_path):
        # Load the color and depth images
        color_image = cv2.imread(os.path.join(dataset_path, filename, 'color.jpg'))
        depth_image = cv2.imread(os.path.join(dataset_path, filename, 'depth.png'), cv2.IMREAD_ANYDEPTH)
        # Convert the depth image to meters
        depth_image = depth_image / 1000.0
        # Add the images to the list
        color_images.append(color_image)
        depth_images.append(depth_image)

    # -----------------------------------------------------------------
    # Training Model can be done in another file.py
    # -----------------------------------------------------------------


    # Process the images to detect and classify objects
    for color_image, depth_image in zip(color_images, depth_images):
        # Convert the color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect objects in the image using a pre-trained object detection model
        objects = RGBDClassifier.detectMultiScale(gray_image)

        # Loop through the detected objects and classify them using a pre-trained object classification model
        for (x, y, w, h) in objects:
            object_image = color_image[y:y + h, x:x + w]
            object_class = RGBDClassifier.predict(object_image)

            # Use the depth image to calculate the distance to the object
            object_depth = depth_image[y:y + h, x:x + w].mean()
            # distance_to_object = object_depth * focal_length / w

            # Print the object class and distance
            print(f'Detected object: {object_class}, distance: {distance_to_object:.2f} meters')


if __name__ == '__main__':
    main()