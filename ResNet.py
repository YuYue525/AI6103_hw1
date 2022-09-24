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
    
def channel_means(data_loader):
    
    channels_sum, num_batches = 0,0

    for data, target in data_loader:
        
        channels_sum += torch.mean(data,dim=[0,2,3])
        # print(torch.mean(data,dim=[0,2,3]))
        
        num_batches +=1
        
    mean = channels_sum/num_batches
    return mean

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
        
def plot_loss(train_loss, test_loss, save_path = None):
    plt.cla()
    plt.plot(range(len(train_loss)), train_loss, 'b')
    plt.plot(range(len(test_loss)), test_loss, 'r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("ResNet: Loss vs Number of epochs")
    plt.legend(['train', 'test'])
    # plt.show()
    if save_path != None:
        plt.savefig(save_path)
    
def plot_acc(train_acc, test_acc, save_path = None):
    plt.cla()
    plt.plot(range(len(train_acc)), train_acc, 'b')
    plt.plot(range(len(test_acc)), test_acc, 'r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title("ResNet: Accuracy vs Number of epochs")
    plt.legend(['train', 'test'])
    # plt.show()
    if save_path != None:
        plt.savefig(save_path)

def data_preprocessing(randomerasing_value = None):
     
    if randomerasing_value == None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=randomerasing_value, inplace=False)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)

    classes = dataset.classes

    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 1)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 1)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size * 2, shuffle = False, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size * 2, shuffle = False, num_workers = 1)

    # print("train_set:", class_count(train_set, classes))
    # print("val_set:", class_count(val_set, classes))
    # print("test_set:", class_count(test_set, classes))
    
    return data_loader, train_loader, val_loader, test_loader

# Training
def train(epoch, net, criterion, trainloader, scheduler, optimizer):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % 50 == 0:
          print("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total))

    scheduler.step()
    return train_loss/(batch_idx+1), 100.*correct/total
    
# Testing
def test(epoch, net, criterion, testloader):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx+1), 100.*correct/total
    
    
# save checkpoint
def save_checkpoint(net, acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    
# defining resnet models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # This is the "stem"
        # For CIFAR (32x32 images), it does not perform downsampling
        # It should downsample for ImageNet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # four stages with three downsampling
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test_resnet18():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# main body
config = {
    'epoch': 300,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4
}

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    data_loader, _, _, _ = data_preprocessing()

    means = tuple([float(x) for x in channel_means(data_loader)])
    # print(means)
    
    data_loader, train_loader, val_loader, test_loader = data_preprocessing(means)
    
    net = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_train_acc = []
    epoch_val_acc = []

    for epoch in range(1, config['epoch'] + 1):
        train_loss, train_acc = train(epoch, net, criterion, train_loader, scheduler, optimizer)
        val_loss, val_acc = test(epoch, net, criterion, val_loader)
        
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)
        epoch_train_acc.append(train_acc)
        epoch_val_acc.append(val_acc)
        
        print(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, valicate loss " + \
          ": %0.4f, valicate accuracy : %2.2f") % (epoch, train_loss, train_acc, val_loss, val_acc))

        
    plot_loss(epoch_train_loss, epoch_val_loss, "loss_epoch_300_cosine_annealing_weight_decay_random_erasing.jpg")
    plot_acc(epoch_train_acc, epoch_val_acc, "acc_epoch_300_cosine_annealing_weight_decay_random_erasing.jpg")
    
    test_loss, test_acc = test(epoch, net, criterion, test_loader)
    print(("test loss : %0.4f, test accuracy : %2.2f") % (test_loss, test_acc))
