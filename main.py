# %%
import os
from pprint import pprint


class City:
    pass


class Cities:
    def __init__(self):
        self.cities = self.get_cities()

    def get_cities(self):
        city_name_to_image_fps = {}
        cities = os.listdir("images")
        # print(cities)
        for city in cities:
            images = os.listdir(os.path.join("images", city))
            city_name_to_image_fps[city] = images
            # print(f"{city} ({len(images)} images)")
        return city_name_to_image_fps

    def __repr__(self):
        return cities.cities


if __name__ == "__main__":
    cities = Cities()
    print(cities)
# %%
