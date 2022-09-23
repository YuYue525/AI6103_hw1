import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

batch_size = 128

train_set_size = 40000
val_set_size = 10000

torch.manual_seed(0)

def class_count(dataset, classes):
    class_count = {}
    for _, index in dataset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count

def show_images(max_batch_num):

    dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle = True, num_workers = 2)
    
    batch_num = 0
    
    for images, _ in dataloader:
        batch_num += 1
        if batch_num > max_batch_num:
            break
        # print('images.shape:', images.shape)
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        plt.savefig("batch_"+str(batch_num)+"_images.png")

if __name__ == "__main__":

    # show_images(20)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)

    classes = dataset.classes

    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 1)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size * 2, shuffle = False, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size * 2, shuffle = False, num_workers = 1)

    print("train_set:", class_count(train_set, classes))
    # print("val_set:", class_count(val_set, classes))
    # print("test_set:", class_count(test_set, classes))


