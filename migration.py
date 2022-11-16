# %%
import os


def remove_non_windows_friendly_characters_from_image_names():
    city_fps = os.listdir("images")
    for city_name in city_fps:
        images = os.listdir(os.path.join("images", city_name))
        image_names = [os.path.join("images", city_name, fp)
                       for fp in images]
        for image_name in image_names:
            old_name = image_name
            image_name = image_name.replace('|', '')
            image_name = image_name.replace('&', '')
            image_name = image_name.replace(':', '')
            os.rename(old_name, image_name)


if __name__ == "__main__":
    remove_non_windows_friendly_characters_from_image_names()

# %%
