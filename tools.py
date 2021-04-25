# PROGRAMMER: Humberto Jimenez
# DATE CREATED: 21 Feb 2021
#Purpose: to create tools that aid the train and predict python program files.
#Parts of this model are inspired directly from the pytorch exercises taught in the Udacity sessions,
#I have added a note to indicate where parts of the code is similar to what was taught, any similarities with other student's work is purely coincidental
#as we have all learned from the same source (examples and videos).

#basic imports
import argparse
import copy
import os
import json

#torch and torchvision
import torch
from torchvision import transforms, datasets

#This section was created taking the part 6 - saving and loading models
# and this mentor shared article https://medium.com/udacity-pytorch-challengers/saving-loading-your-model-in-pytorch-741b80daf3c
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']

    return model

#This section was already contained in part one of the assessment
def load_cat_names(filename):
    with open (filename) as f:
        category_names = json.load(f)
    return category_names
