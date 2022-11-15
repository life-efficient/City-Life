from dataset import CitiesDataset
import torch


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


if __name__ == "__main__":
    citiesDataset = CitiesDataset()
    example = citiesDataset[0]
    print(example)
    features, label = example
    nn = NeuralNetworkClassifier()
    prediction = nn(features)
    print('Prediction:', prediction)
    print('Label:', label)
