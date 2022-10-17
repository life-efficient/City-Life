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


class Cities:
    def __init__(self):
        self.cities = self.get_cities()

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

    # TODO show a random image from a random city


if __name__ == "__main__":
    cities = Cities()
    pprint(cities)
    cities.cities['Beijing, China'].show()
# %%
