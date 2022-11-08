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

    # def __call__(self, features):
    #     print('hello')


if __name__ == "__main__":
    citiesDataset = CitiesDataset()
    example = citiesDataset[0]
    print(example)
    features, label = example
    nn = NeuralNetworkClassifier()
    prediction = nn(features)
    print('Prediction:', prediction)
    print('Label:', label)
