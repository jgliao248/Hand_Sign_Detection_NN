import torch
from matplotlib import pyplot as plt

import constants as c
from Network.network import MyNetwork
import SignLanguageDataset as dataset

label_maps = dataset.create_labels_dict()

def main():
    model_path = c.NETWORK_MODEL_PATH
    optimizer_path = c.NETWORK_OPTIMIZER_PATH
    learning_rate = 1e-3

    model = MyNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))

    model.eval()

    test_loader = torch.utils.data.DataLoader(
        dataset.SignLanguageDataset(dataset.TEST_PATH, dataset.DATA_DIRECTORY_PATH), batch_size=64, shuffle=True)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    #print(example_data[0])

    with torch.no_grad():
        output = model(example_data)


    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            label_maps.get(output.data.max(1, keepdim=True)[1][i].item())))
        plt.xticks([])
        plt.yticks([])

    plt.show()

if __name__ == '__main__':
    main()