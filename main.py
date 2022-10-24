# %%
import os
from pprint import pprint
import random
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import RidgeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class City:
    def __init__(self, city_name):
        self.images = os.listdir(os.path.join("images", city_name))
        self.images = [os.path.join("images", city_name, fp)
                       for fp in self.images]
        self.name = city_name

    def __repr__(self):
        return f"{self.name} ({len(self.images)} images)"

    def show(self):
        img_fp = random.choice(self.images)
        img = Image.open(img_fp)
        img.show()


class CitiesDataset:
    """Many examples of images from different cities"""

    def __init__(self):
        self.cities = self.get_cities().values()
        self.all_imgs = []
        for city in self.cities:
            self.all_imgs.extend(city.images)
        # list of all of the image exmaples
        # self.all_imgs = [*city.images for city in self.cities]

        size = 64
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop((size, size)),
            transforms.Grayscale(),
            transforms.Normalize(self.get_normalisation_parameters())
        ])

    def get_normalisation_parameters(self):
        images, _ = self.get_X_y()
        print(images)
        sdsdc

    def get_cities(self):
        city_map = {}
        city_fps = os.listdir("images")
        # print(cities)
        for city_name in city_fps:
            city_map[city_name] = City(city_name)
        return city_map

    def __repr__(self):
        return "hello"  # str(self.cities)

    # TODO define how to index this class with a city name
    def __getitem__(self, example_idx):
        """Get me the image of example 10 (example_idx)"""
        img_fp = self.all_imgs[example_idx]
        return self.get_X_y_from_fp(img_fp)

    def get_X_y_from_fp(self, img_fp):
        city_name = img_fp.split("/")[1]
        # print(img_fp)
        img = Image.open(img_fp)
        if self.transform:
            img = self.transform(img)
        return img, city_name

    def get_X_y(self):

        images = [
            np.array(
                self.transform(
                    Image.open(img_fp)
                )
            ).flatten()
            for img_fp in self.all_imgs
        ]
        city_names = [img_fp.split("/")[1] for img_fp in self.all_imgs]
        # extra transform

        return images, city_names

    # TODO show a random image from a random city


def evaluate_sklearn_model(model, features, labels):
    f1_score = model.score(features, labels)
    y_pred = model.predict(features)
    # print('F1:', f1_score)
    print('Accuracy:', accuracy_score(labels, y_pred))


if __name__ == "__main__":
    cities = CitiesDataset()
    # for example in cities:
    #     print(example)
    classifier = RidgeClassifier()
    X, y = cities.get_X_y()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=round(0.1*len(X)))
    classifier.fit(X_train, y_train)
    print('Training')
    evaluate_sklearn_model(classifier, X_train, y_train)
    print('Testing')
    evaluate_sklearn_model(classifier, X_test, y_test)


# %%
