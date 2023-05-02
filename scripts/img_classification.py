import os
import csv
import pandas as pd
import torch
import random
import torchvision
import cv2
import torchvision.transforms as transforms
import numpy as np
import lpips

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GeneratedImagesDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class PredictImagesDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img_path = self.img_dir
        image = cv2.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(250000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on ' + str(device))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = GeneratedImagesDataset('/home/buens/Git/jan.buens/data/classification/annotations/train_annotations.csv',
                                           '/home/buens/Git/jan.buens/data/classification/train',
                                           transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    valset = GeneratedImagesDataset('/home/buens/Git/jan.buens/data/classification/annotations/val_annotations.csv',
                                           '/home/buens/Git/jan.buens/data/classification/val',
                                           transform=transform)

    valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                              shuffle=True, num_workers=4)

    testset = GeneratedImagesDataset('/home/buens/Git/jan.buens/data/classification/annotations/test_annotations.csv',
                                           '/home/buens/Git/jan.buens/data/classification/test')

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True, num_workers=4)

    # load VGG16 network and add a fully connect layer for binary classification
    net = models.vgg16(pretrained=True)
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, 2)

    net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.000001)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        correct = 0
        num_samples = len(valloader.dataset)
        for i, data in enumerate(valloader, 0):

            input, label = data[0].to(device), data[1].to(device)

            output = net(input)

            predicted = torch.argmax(F.softmax(output)).item()
            label = label.item()
            print('prediction: ' + str(predicted) + ', label: ' + str(label))

            if predicted == label:
                correct += 1

        accuracy = correct / num_samples
        print('Accuracy: ' + str(round(accuracy * 100, 2)) + '%')

    print('Finished Training')

    print('Testing...')
    correct = 0
    num_samples = len(valloader.dataset)

    for i, data in enumerate(valloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net(inputs)

        _, predicted = torch.max(outputs, 1)
        for i, j in zip(predicted, labels):
            v = i.item()
            j = j.item()
            print('prediction: ' + str(v) + ', label: ' + str(j))
            if v == j:
                correct += 1

    accuracy = correct / num_samples
    print('Accuracy: ' + str(round(accuracy * 100, 2)) + '%')

    PATH = './net.pth'
    torch.save(net.state_dict(), PATH)


def choose_image(save_dir, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load VGG16 network and add a fully connect layer for binary classification
    net = models.vgg16(pretrained=False)
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, 2)
    net.load_state_dict(torch.load(path))
    net.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1

    testset = GeneratedImagesDataset(save_dir + '/annotations/test_annotations.csv', save_dir + '/test',
                                         transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    correct = 0
    num_samples = len(testloader.dataset)

    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        predicted = torch.argmax(F.softmax(outputs)).item()
        label = labels.item()

        print('prediction: ' + str(predicted) + ', label: ' + str(label))
        if predicted == label:
            correct += 1

    accuracy = correct / num_samples
    print('Accuracy: ' + str(round(accuracy * 100, 2)) + '%')


def predict(img_path, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load VGG16 network and add a fully connect layer for binary classification
    net = models.vgg16(pretrained=False)
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, 2)
    net.load_state_dict(torch.load(path))
    net.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = PredictImagesDataset(img_path, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=4)

    for i, data in enumerate(testloader, 0):
        inputs = data.to(device),
        outputs = net(inputs[0])
        print(outputs)
        predicted = torch.argmax(outputs).item()

        # 0 = shadow, 1 = no shadow
        print('prediction: ' + str(predicted))

"""
directory = os.fsencode('/home/buens/Git/jan.buens/data/classification/test')

with open('/home/buens/Git/jan.buens/data/classification/annotations/test_annotations.csv', 'a') as f:
    c = 1
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        row = str(filename) + ', ' + str(c)

        f.write(row)
        f.write('\n')
"""
# train()
# choose_image('/home/buens/Git/jan.buens/data/classification/', '/home/buens/Git/jan.buens/GAN_Inversion/scripts/net.pth')
# predict('/home/buens/Git/RandomShadow/ffhq_no_shadow/shadowed_images/01568.png', '/home/buens/Git/jan.buens/GAN_Inversion/scripts/net.pth')