import numpy as np
import torch
from torch import nn, optim

import Network.network
from Network.network import MyNetwork
import SignLanguageDataset as dataset
from sklearn.metrics import f1_score
import constants as c
import torch.nn.functional as F

data = torch.utils.data.DataLoader(
    dataset.SignLanguageDataset(dataset.TRAIN_PATH, dataset.DATA_DIRECTORY_PATH), batch_size=128, num_workers=4,
    shuffle=True)

val_loader = torch.utils.data.DataLoader(
    dataset.SignLanguageDataset(dataset.TEST_PATH, dataset.DATA_DIRECTORY_PATH), batch_size=64, shuffle=True)
device = "cpu"

model_path = c.NETWORK_MODEL_PATH
optimizer_path = c.NETWORK_OPTIMIZER_PATH
# Validating the model against the validation dataset and generate the accuracy and F1-Score.
def validate(val_loader,model, test_losses):
    model.eval()
    test_labels=[0]
    test_pred=[0]
    test_loss = 0
    for i, (images,labels) in enumerate(val_loader):
        outputs=model(images.to(device))
        predicted = torch.softmax(outputs,dim=1)
        test_loss += F.nll_loss(predicted, labels, size_average=False).item()
        _,predicted=torch.max(predicted, 1)
        test_pred.extend(predicted)
        test_labels.extend(labels)

    test_loss /= len(val_loader.dataset)
    test_losses.append(test_loss)
    test_pred=np.array(test_pred[1:])
    test_labels=np.array(test_labels[1:])
    correct=(test_pred==test_labels).sum()
    accuracy=correct/len(test_labels)
    f1_test=f1_score(test_labels,test_pred,average='weighted')
    model.train()
    return accuracy,f1_test, test_losses

def main():
    model = MyNetwork()
    model = model.to("cpu")
    model.train()
    checkpoint = None


    learning_rate = 1e-3
    start_epoch = 0
    end_epoch = 5
    momentum = 0.5

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(val_loader.dataset) for i in range(end_epoch + 1)]


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True, min_lr=1e-6)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        start_epoch = torch.load(checkpoint)['epoch']


    for epoch in range(start_epoch, end_epoch + 1):
        for i, (images, labels) in enumerate(val_loader):
            outputs = model(images.to(device))
            loss = criterion(outputs.to(device), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(predicted, 1)
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')

            if i % c.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(data.dataset),
                           100. * i / len(data), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (i * 128) + ((epoch - 1) * len(data.dataset)))
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optimizer_path)
        val_accuracy, val_f1, test_losses = validate(val_loader, model, test_losses)
        print("------------------------------------------------------------------------------------------------------")
        print("Epoch [{}/{}], Training F1: {:.4f}, Validation Accuracy: {:.4f}, Validation F1: {:.4f}".format(epoch,
                                                                                                              end_epoch,
                                                                                                              f1,
                                                                                                              val_accuracy,
                                                                                                              val_f1))
        scheduler.step(val_accuracy)
    Network.network.graph_results(train_losses, train_counter, test_losses, test_counter)

if __name__ == '__main__':
    main()