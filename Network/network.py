import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import SignLanguageDataset as data
import constants as c

class MyNetwork(nn.Module):
    """
    The base neural network used for assignment.
    """

    def __init__(self):
        """
        The constructor of MyNetwork class.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),   # 26 x 26
            nn.ReLU(),
            nn.MaxPool2d(2),        # 13 x 13
            nn.Dropout(0.2),
            nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),   # 11 x 11
            nn.ReLU(),
            nn.MaxPool2d(2),        # 5 x 5
            nn.Dropout(0.2),
            nn.BatchNorm2d(64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),  # 3 x 3
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm2d(128)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2)

        )
        self.fc2 = nn.Linear(256, 25)
        self.flatten = nn.Flatten()


    def forward(self, x):
        """
        Computes a forward pass for the network
        :param x: the input to the neural network.
        :return:
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)


def train(network_name: str, network: MyNetwork, train_losses: list, train_counts: list,
          train_loader: torch.utils.data.DataLoader, optimizer: optim.SGD, epoch: int,
          batch_size_train):
    """
    Trains the neural network with the given data. It will log the training counts and losses into a list.
    After the training, the network model is saved along with the optimizer
    :param batch_size_train: size of training batch
    :param network_name: the name of the network model to be saved
    :param network: the network to be trained
    :param train_losses: a list that stores the training loss data
    :param train_counts: a list that stores the training counts
    :param train_loader: data loader that holds the training data
    :param optimizer: the optimizer for the model
    :param epoch: the current epoch
    :return: updated training losses and train counts after training after the epoch
    """
    network.train()
    # get the names of the network and optimizer
    model_path = c.NETWORK_MODEL_PATH
    optimizer_path = c.NETWORK_OPTIMIZER_PATH

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % c.LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counts.append(
                (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

    return train_losses, train_counts


def test(network, test_loader, test_losses):
    """
    Tests the given network with the given test data and returns the test losses
    :param network: the network being tested
    :param test_loader: the data loader with the test data
    :param test_losses: a list that stores the test losses
    :return: the updated test losses
    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_losses


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
    n_epochs = 20
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.001
    momentum = 0.5

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )




    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        data.SignLanguageDataset(data.TRAIN_PATH, data.DATA_DIRECTORY_PATH), batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        data.SignLanguageDataset(data.TEST_PATH, data.DATA_DIRECTORY_PATH), batch_size=batch_size_test, shuffle=True)

    network = MyNetwork().to("cpu")
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test_losses = test(network, test_loader, test_losses)

    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter = train("model_name", network, train_losses, train_counter, train_loader, optimizer,
                                            epoch, batch_size_train)

        test_losses = test(network, test_loader, test_losses)

    graph_results(train_losses, train_counter, test_losses, test_counter)

def testing():
    batch_size_train = 64
    train_loader = torch.utils.data.DataLoader(
        data.SignLanguageDataset(data.TRAIN_PATH, data.DATA_DIRECTORY_PATH, torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])), batch_size=batch_size_train, shuffle=True)
    train_loader.dataset.type
if __name__ == '__main__':
    main()