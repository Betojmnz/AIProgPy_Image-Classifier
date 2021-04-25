# PROGRAMMER: Humberto Jimenez
# DATE CREATED: 21 Feb 2021
#Purpose: to create a model training object that acts as an image classifier.
#Parts of this model are inspired directly from the pytorch exercises taught in the Udacity sessions,
#I have added a note to indicate where parts of the code is similar to what was taught, any similarities with other student's work is purely coincidental
#as we have all learned from the same source (examples and videos).

#basic imports
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse

from tools import load_checkpoint

#torch and torchvision imports
import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder

#Define arguments loader
#Learning source https://pymotw.com/3/argparse/
def parse_args():
    parser = argparse.ArgumentParser(description="Model trainer backend")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16','alexnet'])
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=1024) #Modified the default hidden units.
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001', type =float)
    parser.add_argument('--epochs', dest='epochs', default='5', type=int)
    parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
    parser.add_argument('--gpu', action='store', default='gpu') #Added gpu/cpu argument
    return parser.parse_args()

#Create model trainer
#The model trainer section was inspired by the Part 8 - transfer learning activity (Please refer to the training videos/materials to check for similarities)
def model_trainer(model, criterion, optimizer, dataloaders, epochs):

    steps=0
    training_loss = 0
    print_every = 100
    args = parse_args() #Parsing arguments into the model


    #move the model to the gpu environment
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu' #readjusted the cuda statement
    model.to(device)

    #Message to inform the user that the model is about to be trained
    print('training starts in..')
    print('3..')
    print('2..')
    print('1')

    for e in range(epochs):
        for images, labels in dataloaders[0]:
            steps += 1

            #Move the images and labels into the device/cuda
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            #forward and backward steps
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if steps % print_every ==0:
                vloss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in dataloaders[1]:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        optimizer.zero_grad()

                        vloss += batch_loss.item()

                        #Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                print(f"Epoch {e+1}/{epochs}.. "
                      f"Training loss: {training_loss/print_every:.3f}.. "
                      f"Valid loss: {vloss/len(dataloaders[1]):.3f}.. "
                      f"Accuracy: {accuracy/len(dataloaders[1]):.3f}")

                training_loss = 0

# data loads and model preparation
#This section was inspired by the Part 8 - transfer learning activity (please check training materials to identify similarities)
def main ():
    #invoke the arguments into the function
    args = parse_args()
    #load the data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    image_datasets = [datasets.ImageFolder(train_dir, transform = training_transforms),
                      datasets.ImageFolder(valid_dir, transform = validation_transforms),
                      datasets.ImageFolder(test_dir, transform = testing_transforms)]

    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size = 64, shuffle= True), #adjusted batch size, with 32 the model training throws very small values.
                   torch.utils.data.DataLoader(image_datasets[1], batch_size = 64),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size = 64)]

    model = getattr(models, args.arch)(pretrained = True)

    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    #Determine number of features based on the selected architecture
    if args.arch == 'vgg16':
        fc1 = 25088 #input layer
        fc2 = args.hidden_units #Use hidden units values or default
        fc3 = 102 #output layer

    #Features based on the alexnet model
    else:
        fc1 = 9216 #input layer
        fc2 = args.hidden_units #Use hidden units values or default
        fc3 = 102 #output layer

    #Define model/classifier
    model.classifier = nn.Sequential(nn.Linear(fc1, fc2),
                                     nn.ReLU(),
                                     nn.Dropout(0.6), #adjusted dropout rate to 0.6 to increase accuracy. Initially set at 0.5
                                     nn.Linear(fc2,fc3),
                                     nn.LogSoftmax(dim=1))


    #Define all arguments needed for running the model trainer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= args.learning_rate)
    model.class_to_idx = image_datasets[0].class_to_idx
    epochs = args.epochs

    #Run the model trainer
    model_trainer (model, criterion, optimizer, dataloaders, epochs)

    #Save the checkpoint
    path = args.save_dir

    checkpoint = {'model': model,
                  'input size': 25088,
                  'output size': 102,
                  'arch':args.arch,
                  'classifier': model.classifier,
                  'hidden_layers': args.hidden_units, #Save hidden units value
                  'learning_rate': args.learning_rate,
                  'epochs': args.epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, path)

    print ('model trained and checkpoint created')

if __name__ == "__main__":
    main()
  