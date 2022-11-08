from dataset import CitiesDataset
from classifier import NeuralNetworkClassifier
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def train(model, dataloader, epochs=1):
    """
    Trains a neural network on a dataset and returns the trained model

    Parameters:
    - model: a pytorch model
    - dataloader: a pytorch dataloader

    Returns:
    - model: a trained pytorch model
    """

    # components of a ml algortithms
    # 1. data
    #  2. model
    # 3. criterion (loss function)
    # 4. optimiser

    # initialise an optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):  # for each epoch
        for batch in dataloader:  # for each batch in the dataloader
            features, labels = batch
            prediction = model(features)  # make a prediction
            # compare the prediction to the label to calculate the loss (how bad is the model)
            loss = F.cross_entropy(prediction, labels)
            loss.backward()  # calculate the gradient of the loss with respect to each model parameter
            optimiser.step()  # use the optimiser to update the model parameters using those gradients
            print(loss)  # log the loss
            optimiser.zero_grad()  # zero grad

    # return trained model

# class DataLoader
#     def __iter__():
#         while True:
#             shuffle(dataset)
#             for batch in dataset:
#                 yield batch


if __name__ == "__main__":
    dataset = CitiesDataset()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=16)
    nn = NeuralNetworkClassifier()
    train(nn, train_loader)
