# %%
import os


class City:
    pass


class Cities:
    def __init__(self):
        self.cities = self.get_cities()

    def get_cities():
        city_name_to_image_fps = {}
        cities = os.listdir("images")
        print(cities)
        for city in cities:
            images = os.listdir(os.path.join("images", city))

            print(f"{city} ({len(images)} images)")


if __name__ == "__main__":
    get_cities()

# %%
