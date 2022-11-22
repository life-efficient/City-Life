import requests
from dataset import CitiesDataset
import torch
from torchvision.models import resnet50


# class Module:
#     def __call__(self):

#         # other interesting stuff
#         self.forward()

#     def forward():
#         raise NotImplementedError()

class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # initialise weights and biases (parameters)
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4096, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
            # torch.nn.Softmax()
        )

    def forward(self, features):
        """Takes in features and makes a prediction"""
        return self.layers(features)


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # initialise weights and biases (parameters)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            # torch.nn.Softmax()
        )

    def forward(self, features):
        """Takes in features and makes a prediction"""
        return self.layers(features)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TransferLearning(torch.nn.Module):
    def __init__(self):
        super().__init__()

        embedding_size = 2048

        self.layers = resnet50()
        for param in self.layers.parameters():
            param.grad_required = False

        self.embedding_extractor = resnet50()
        self.embedding_extractor.fc = torch.nn.Sequential(
            Identity(),
            # torch.nn.Linear(2048, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 128),
        )

        self.predict_from_extracted_features = torch.nn.Linear(2048, 10)

        self.layers = torch.nn.Sequential(
            self.embedding_extractor,
            self.predict_from_extracted_features
        )

    def forward(self, x):
        return self.layers(x)

    def embed(self, x):
        return self.embedding_extractor(x)


if __name__ == "__main__":
    model = TransferLearning()
    # nn = NeuralNetworkClassifier()
    # citiesDataset = CitiesDataset()
    # example = citiesDataset[0]
    # print(example)
    # features, label = example
    prediction = model(features)
    print('Prediction:', prediction)
    print('Label:', label)
