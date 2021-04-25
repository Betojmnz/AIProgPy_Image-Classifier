# PROGRAMMER: Humberto Jimenez
# DATE CREATED: 21 Feb 2021
#Purpose: to create a model training object that acts as an image classifier.
#Parts of this model are inspired directly from the pytorch exercises taught in the Udacity sessions,
#I have added a note to indicate where parts of the code is similar to what was taught, any similarities with other student's work is purely coincidental
#as we have all learned from the same source (examples and videos).

#basic imports
import argparse
import numpy as np
from PIL import Image
import json
import os
import random

#torch and torchvision import
import torch
from torchvision import transforms, models
import torch.nn.functional as F

from tools import load_checkpoint, load_cat_names

#Define arguments loader
#Learning source https://pymotw.com/3/argparse/
def parse_args ():
    parser = argparse.ArgumentParser(description = "Image classifier calculations")
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--topk', dest='topk', default='5', type=int)
    parser.add_argument('--filepath', dest='filepath', type=str)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu') #Added gpu/cpu argument
    return parser.parse_args()

#Define image processor
#This section was inspired by browsing the web and with the aid of this learnign source: https://pillow.readthedocs.io/en/latest/reference/Image.html
def process_image(image):
    #Open the image using PIL
    pil_image = Image.open(image)

    #perform transformations to make it suitable for the model
    adjust_image = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

    #adjust image using the transformations
    image = adjust_image(pil_image)

    return image

#Define predictor
def predict(image_path, model, topk):
    #Invoke the arguments
    args = parse_args() #Adding to fix args not defined issue
    #move the model to the gpu environment
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu' #readjusted the cuda statement
    model.to(device)

    # transform the image by running it through the process_image method,
    #unsqueeze to return the tensor with an extra dimension and convert it to float.
    # Reference for unsqueeze method: https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

    image = torch.unsqueeze(process_image(image_path),0).float()

    #Enable no grad to prevent a backward step and reduce memory consumption
    with torch.no_grad():
        #Send image and label to the gpu/device
        image = image.to(device)
        #push the image into the network
        logps = model.forward(image)
    #Calculate the probability using the softmax function
    probability = F.softmax(logps.data, dim=1)
    #Create the array of top probabilities
    probs = np.array(probability.topk(topk)[0][0])

    #Determine top classes and create an array with the top possible answers
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]

    return probs, top_classes

#Define main part of the program to run
def main():
    #invoke all arguments for the function
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)

    #Choose a random image from the test folder #10 if the user does not specify a file
    if args.filepath is None:
        test_image = random.choice(os.listdir('flowers/test/10/'))
        test_img_path= 'flowers/test/10/'+ test_image
    else:
        test_img_path = args.filepath

    #Calculate probability and classes using model
    probs, classes = predict(test_img_path, model, args.topk) #Added the topk to allow the user to choose their no. of classes
    labels = [cat_to_name[str(index)] for index in classes]

    #Print results to display file, class labels and probability
    print ('selected file:' + test_img_path)
    print('the image is likely to be a: {} with a confidence interval of: {:.3f} %'.format(cat_to_name[str(classes[0])] , probs[0]))
    print('\n other alternatives are: \n {} \n with {} probability'.format(labels, probs))


if __name__ == "__main__":
    main()
