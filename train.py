from dataset import CitiesDataset
from classifier import NeuralNetworkClassifier, CNN
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import random_split


def train(model, dataloader, val_loader, test_loader, epochs=100):
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
    # 2. model
    # 3. criterion (loss function)
    # 4. optimiser

    writer = SummaryWriter()

    # initialise an optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=0.5)
    batch_idx = 0
    for epoch in range(epochs):  # for each epoch
        for batch in dataloader:  # for each batch in the dataloader
            features, labels = batch
            prediction = model(features)  # make a prediction
            # compare the prediction to the label to calculate the loss (how bad is the model)
            loss = F.cross_entropy(prediction, labels)
            loss.backward()  # calculate the gradient of the loss with respect to each model parameter
            optimiser.step()  # use the optimiser to update the model parameters using those gradients
            print("Epoch:", epoch, "Loss:", loss.item())  # log the loss
            optimiser.zero_grad()  # zero grad
            writer.add_scalar("Loss/Train", loss.item(), batch_idx)
            batch_idx += 1
        print('Evaluating on valiudation set')
        # evaluate the validation set performance
        val_loss = evaluate(model, val_loader)
        writer.add_scalar("Loss/Val", val_loss, batch_idx)
    # evaluate the final test set performance
    print('Evaluating on test set')
    test_loss = evaluate(model, test_loader)
    # writer.add_scalar("Loss/Test", test_loss, batch_idx)
    model.test_loss = test_loss
    return model   # return trained model


def evaluate(model, dataloader):
    losses = []
    for batch in dataloader:
        features, labels = batch
        prediction = model(features)
        loss = F.cross_entropy(prediction, labels)
        losses.append(loss.detach())
    avg_loss = np.mean(losses)
    print(avg_loss)
    return avg_loss


if __name__ == "__main__":
    dataset = CitiesDataset()
    train_set_len = round(0.7*len(dataset))
    val_set_len = round(0.15*len(dataset))
    test_set_len = len(dataset) - val_set_len - train_set_len
    split_lengths = [train_set_len, val_set_len, test_set_len]
    # split the data to get validation and test sets
    train_set, val_set, test_set = random_split(dataset, split_lengths)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)
    nn = NeuralNetworkClassifier()
    cnn = CNN()
    train(
        cnn,
        train_loader,
        val_loader,
        test_loader,
        epochs=1000
    )
