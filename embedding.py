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

# montage images/img_*.jpg - tile 50x50 - geometry 50x50! sprite.jpg


def embed_all_images():

    size = 128
    batch_size = 256

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop((size, size)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])
    dataset = CitiesDataset(transform=transform)
    image_loader = DataLoader(dataset, batch_size=batch_size)
    model = TransferLearning()

    # load in trained parameters
    try:
        state_dict = torch.load("models/latest_model.pt")

    except:
        raise FileNotFoundError(
            "Model weights not found, run `train.py` first")

    writer = SummaryWriter()

    model.load_state_dict(state_dict)
    model.eval()

    print('Embedding Images')
    for batch_idx, batch in enumerate(image_loader):
        features, labels = batch
        embeddings = model.embed(features)
        # print(embeddings)
        break
        print(f"Embedding batch {batch_idx}")

    writer.add_embedding(
        embeddings,
        label_img=features,
        tag="embeddings",
        metadata=[dataset.idx_to_city_name[int(label)] for label in labels]
        # global_step=batch_idx
    )

    return features, embeddings


def compare_embeddings(embeddings):
    """
    Inputs: Tensor of embeddings (rows = examples, columns = embeddings)

    Output: Square matrix (or upper triangular) containing cosine similarity between each

    """
    features, embeddings = embeddings
    features = features.detach()
    embeddings = embeddings.detach()

    # img_grid = torchvision.utils.make_grid(features)
    # writer.add_image("batch", img_grid)
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
