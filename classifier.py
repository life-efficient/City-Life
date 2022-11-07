from dataset import CitiesDataset
import torch


class NeuralNetworkClassifier(torch.nn.Module):
    pass
    # TODO


if __name__ == "__main__":
    citiesDataset = CitiesDataset()
    example = citiesDataset[0]
    print(example)
    features, label = example
    nn = NeuralNetworkClassifier()
    prediction = nn(features)
    print('Prediction:', prediction)
    print('Label:', label)
