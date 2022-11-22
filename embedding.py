from dataset import CitiesDataset
from torch.utils.data import DataLoader
from classifier import TransferLearning
import torchvision
from torchvision import transforms
import torch
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter


def embed_all_images():

    size = 128
    n_images = 64
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop((size, size)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])
    dataset = CitiesDataset(transform=transform)
    image_loader = DataLoader(dataset, batch_size=n_images)
    model = TransferLearning()

    # load in trained parameters
    try:
        state_dict = torch.load("models/latest_model.pt")

    except:
        raise FileNotFoundError(
            "Model weights not found, run `train.py` first")

    model.load_state_dict(state_dict)
    for batch in image_loader:
        features, _ = batch
        embeddings = model.embed(features)
        print(embeddings)
        break

    return features, embeddings


def compare_embeddings(embeddings):
    """
    Inputs: Tensor of embeddings (rows = examples, columns = embeddings)

    Output: Square matrix (or upper triangular) containing cosine similarity between each

    """
    writer = SummaryWriter()
    features, embeddings = embeddings
    features = features.detach()
    embeddings = embeddings.detach()
    label_img = features
    print(label_img.shape)
    writer.add_embedding(
        embeddings,
        label_img=label_img,
        tag="embeddings"
    )
    img_grid = torchvision.utils.make_grid(features)
    writer.add_image("batch", img_grid)
    similarities = torchmetrics.functional.pairwise_cosine_similarity(
        embeddings)
    print(similarities)
    # for idx1, idx2 in range(len(embeddings)):
    #     e1 = embeddings[idx1]
    # torch.cosine_similarity(e1, e2)

    disp = ConfusionMatrixDisplay(similarities.numpy()).plot(cmap="Purples")
    # disp.figure_.savefig("")
    # plt.show()
    # plt.imshow(similarities, cmap='hot', interpolation='nearest')
    # # plt.savefig('matrix.png')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()


if __name__ == "__main__":
    embeddings = embed_all_images()
    compare_embeddings(embeddings=embeddings)
