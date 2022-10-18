# %%
import os


def remove_non_windows_friendly_characters_from_image_names():
    city_fps = os.listdir("images")
    for city_name in city_fps:
        images = os.listdir(os.path.join("images", city_name))
        image_names = [os.path.join("images", city_name, fp)
                       for fp in images]
        for image_name in image_names:
            new_name = image_name.replace('|', '')
            print()
            print(image_name)
            print(new_name)
            os.rename(image_name, new_name)


if __name__ == "__main__":
    remove_non_windows_friendly_characters_from_image_names()

# %%
