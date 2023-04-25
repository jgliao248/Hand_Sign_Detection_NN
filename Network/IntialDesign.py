"""
Justin Liao
CS5330
Spring 2023
Final Project


This file contains the initial NN design using PyTorch to learn and test the data given on Kaggle.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

from constants import TEST_DATA_FILE_PATH, TRAIN_DATA_FILE_PATH


class GestureDataset(Dataset):
    """
    The data class that extracts the 28x28 image from the csv file to be used as a tensor object for
    PyTorch
    """
    def __init__(self, csv, train=True):
        self.csv = pd.read_csv(csv)
        self.img_size = 224
        # print(self.csv['image_names'][:5])
        self.train = train
        text = "pixel"
        self.images = torch.zeros((self.csv.shape[0], 1))
        for i in range(1, 785):
            temp_text = text + str(i)
            temp = self.csv[temp_text]
            temp = torch.FloatTensor(temp).unsqueeze(1)
            self.images = torch.cat((self.images, temp), 1)
        self.labels = self.csv['label']
        self.images = self.images[:, 1:]
        self.images = self.images.view(-1, 28, 28)

    def __getitem__(self, index):
        img = self.images[index]
        img = img.numpy()
        img = cv2.resize(img, (self.img_size, self.img_size))
        tensor_image = torch.FloatTensor(img)
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image /= 255.
        if self.train:
            return tensor_image, self.labels[index]
        else:
            return tensor_image

    def __len__(self):
        return self.images.shape[0]


class Classifier(nn.Module):
    """
    initial NN for learning features and classifying the ASL data from the csv file
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # 220, 220
            nn.MaxPool2d(2),  # 110, 110
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),  # 106, 106
            nn.MaxPool2d(2),  # 53,53
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),  # 51, 51
            nn.MaxPool2d(2),  # 25, 25
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),  # 23, 23
            nn.MaxPool2d(2),  # 11, 11
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),  # 9, 9
            nn.MaxPool2d(2),  # 4, 4
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.Linear1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.1)
        self.Linear3 = nn.Linear(256, 25)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.dropout(x)
        x = self.Conv5(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x

def validate(val_loader,model):
    """
    A test/validate function to test how the training of the model was performing against the train data set
    :param val_loader: the data loader containing the test data
    :param model: the model being tested
    :return: a tuple containing the accuracy from the testing, the f1 result and the test_loss from the testing
    """
    test_loss = 0


    model.eval()
    test_labels=[0]
    test_pred=[0]
    for i, (images,labels) in enumerate(val_loader):
        outputs=model(images)
        predicted = torch.softmax(outputs,dim=1)
        _,predicted=torch.max(predicted, 1)
        test_pred.extend(list(predicted.data.cpu().numpy()))
        test_labels.extend(list(labels.data.cpu().numpy()))
        test_loss += F.nll_loss(outputs, labels, size_average=False).item()

    test_pred=np.array(test_pred[1:])
    test_labels=np.array(test_labels[1:])
    correct=(test_pred==test_labels).sum()
    accuracy=correct/len(test_labels)
    f1_test=f1_score(test_labels,test_pred,average='weighted')
    model.train()

    test_loss /= len(val_loader.dataset)

    return accuracy,f1_test, test_loss


def graph_results(train_losses: list, train_counter: list, test_losses: list, test_counter: list):
    """
    Graphs the training and testing results from the creation of the network onto a window.
    :param train_losses: a list that contains the training loss data
    :param train_counter: a list that contains the train counter data
    :param test_losses: a list that contains the test loss data
    :param test_counter: a list that contains the test counter data
    :return: None
    """
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


def main():
    """
    The main driver for loading the training and testing datasets to train and test a NN
    :return: none
    """
    test_batch_size = 64
    train_batch_size = 128
    data = GestureDataset(TRAIN_DATA_FILE_PATH)
    data_val = GestureDataset(TEST_DATA_FILE_PATH)
    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=train_batch_size, num_workers=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=data_val, batch_size=test_batch_size, num_workers=0, shuffle=True)

    model = Classifier()

    model.train()
    checkpoint = None
    device = "cuda"
    learning_rate = 1e-3
    start_epoch = 0
    end_epoch = 10
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True, min_lr=1e-6)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(end_epoch + 1)]

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        start_epoch = torch.load(checkpoint)['epoch']
    for epoch in range(start_epoch, end_epoch + 1):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(predicted, 1)
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            train_losses.append(loss.item())
            train_counter.append(
                (i * train_batch_size) + ((epoch - 1) * len(train_loader.dataset)))
        val_accuracy, val_f1, val_loss = validate(val_loader, model, device)
        test_losses.append(1 - val_accuracy)
        print("------------------------------------------------------------------------------------------------------")
        print("Epoch [{}/{}], Training F1: {:.4f}, Validation Accuracy: {:.4f}, Validation F1: {:.4f}".format(epoch,
                                                                                                              end_epoch,
                                                                                                              f1,
                                                                                                              val_accuracy,
                                                                                                              val_f1))
        scheduler.step(val_accuracy)

    graph_results(train_losses, train_counter, test_losses, test_counter)

if __name__ == '__main__':
    main()