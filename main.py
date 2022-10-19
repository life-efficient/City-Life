# %%
import os
from pprint import pprint
import random
from PIL import Image


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

    def get_cities(self):
        city_map = {}
        city_fps = os.listdir("images")
        # print(cities)
        for city_name in city_fps:
            city_map[city_name] = City(city_name)
        return city_map

    def __repr__(self):
        return str(self.cities)

    # TODO define how to index this class with a city name
    def __getitem__(self, example_idx):
        """Get me the image of example 10 (example_idx)"""
        img_fp = self.all_imgs[example_idx]
        img = Image.open(img_fp)
        img.thumbnail((256, 256))

        return img

    # TODO show a random image from a random city


if __name__ == "__main__":
    cities = CitiesDataset()
    # pprint(cities)
    # cities.cities['Beijing, China'].show()
    example100 = cities[100]
    ex10 = cities[10]
    print(example100)
    example100.show()
    print(ex10)
    # cities['Beijing, China'].show()
# %%
