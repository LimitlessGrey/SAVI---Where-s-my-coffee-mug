

import torch
import numpy as np
from colorama import Fore, Style
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_filenames):

        super().__init__()

        self.image_filenames = image_filenames
        self.num_images = len(self.image_filenames)

        self.labels = []
        for image_filename in self.image_filenames:
            self.labels.append(self.getClassFromFilename(image_filename))

        # Create a set of transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def __getitem__(self, index):  # return a specific element x,y given the index, of the dataset

        # Load the image
        image_pil = Image.open(self.image_filenames[index])

        image_t = self.transforms(image_pil)

        return image_t, self.labels[index]

    def __len__(self):  # return the length of the dataset
        return self.num_images

    def getClassFromFilename(self, filename):

        parts = filename.split('/')
        part = parts[-1]
        parts = part.split('.')
        class_name = parts[0]

        # TODO: Define conditions for the diferent Labels (mug, banana, etc)

        if class_name == 'mug':
            label = 0  # use the idx of the outputs vector where the 1 should be
        elif class_name == '...':
            label = 1
        elif class_name == 'banana':
            label = 2
        else:
            raise ValueError('Unknown class')

        return label
