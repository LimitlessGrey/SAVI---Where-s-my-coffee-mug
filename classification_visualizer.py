#!/usr/bin/env python3

import random

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import nn
from torchvision import datasets, models, transforms



class ClassificationVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
        self.tensor_to_pil_image = transforms.ToPILImage()

    def draw(self, inputs, labels, outputs):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(12,10)
        plt.suptitle(self.title)
        # plt.legend(loc='best')
        label_dataset = labels
        inputs = inputs
       
 
        batch_size,_,_,_ = list(inputs.shape)
      
        output_probabilities = F.softmax(outputs, dim=1).tolist()

        random_idxs = random.sample(list(range(batch_size)), k=5*5)
        for plot_idx, image_idx in enumerate(random_idxs, start=1):

            output_probability = output_probabilities[image_idx]
         
            max_value=max(output_probability)
            label_test = output_probability.index(max_value)
            label_check = label_dataset[image_idx].data.item()
            # print(label_test, label_check)
            if label_test == 0:
                class_name = 'apple'
            elif label_test == 1:
                class_name = 'ball'
            elif label_test == 2:
                class_name = 'banana'
            elif label_test == 3:
                class_name = 'bell_pepper'
            elif label_test == 4:
                class_name = 'binder'
            elif label_test == 5:
                class_name = 'bowl'
            elif label_test == 6:
                class_name = 'calculator'
            elif label_test == 7:
                class_name = 'camera'
            elif label_test == 8:
                class_name = 'cap'
            elif label_test == 9:
                class_name = 'cell_phone'
            elif label_test == 10:
                class_name = 'cereal_box'
            elif label_test == 11:
                class_name = 'coffee_mug'
            elif label_test == 12:
                class_name = 'comb'
            elif label_test == 13:
                class_name = 'dry_battery'
            elif label_test == 14:
                class_name = 'flashlight'
            elif label_test == 15:
                class_name = 'food_bag'
            elif label_test == 16:
                class_name = 'food_box'
            elif label_test == 17:
                class_name = 'food_can'
            elif label_test == 18:
                class_name = 'food_cup'
            elif label_test == 19:
                class_name = 'food_jaar'
            elif label_test == 20:
                class_name = 'garlic'
            elif label_test == 21:
                class_name = 'glue_stick'
            elif label_test == 22:
                class_name = 'greens'
            elif label_test == 23:
                class_name = 'hand_towel'
            elif label_test == 24:
                class_name = 'instant_noodles'
            elif label_test == 25:
                class_name = 'keyboard'
            elif label_test == 26:
                class_name = 'kleenex'
            elif label_test == 27:
                class_name = 'lemon'
            elif label_test == 28:
                class_name = 'lightbulb'
            elif label_test == 29:
                class_name = 'lime'
            elif label_test == 30:
                class_name = 'marker'
            elif label_test == 31:
                class_name = 'mushroom'
            elif label_test == 32:
                class_name = 'notebook'
            elif label_test == 33:
                class_name = 'onion'
            elif label_test == 34:
                class_name = 'orange'
            elif label_test == 35:
                class_name = 'peach'
            elif label_test == 36:
                class_name = 'pear'
            elif label_test == 37:
                class_name = 'pitcher'
            elif label_test == 38:
                class_name = 'plate'
            elif label_test == 39:
                class_name = 'pliers'
            elif label_test == 40:
                class_name = 'potato'
            elif label_test == 41:
                class_name = 'rubber_eraser'
            elif label_test == 42:
                class_name = 'scissors'
            elif label_test == 43:
                class_name = 'shampoo'
            elif label_test == 44:
                class_name = 'soda_can'
            elif label_test == 45:
                class_name = 'sponge'
            elif label_test == 46:
                class_name = 'stapler'
            elif label_test == 47:
                class_name = 'tomato'
            elif label_test == 48:
                class_name = 'toothbrush'
            elif label_test == 49:
                class_name = 'toothpaste'
            elif label_test == 50:
                class_name = 'water_bottle'
            elif label_test == 48:
                class_name = 'toothbrush'
            else:
                raise ValueError('Unknown class')

            if label_dataset[image_idx].data.item() == label_test:
                color='green'
            else:
                color='red'

            
            image_t = inputs[image_idx,:,:,:]
            image_pil = self.tensor_to_pil_image(image_t)

            ax = self.figure.add_subplot(5,5,plot_idx) # define a 5 x 5 subplot matrix
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_xlabel(class_name, color=color, size=14)

           

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)