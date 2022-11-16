# %%
import os
import re
import string
from uuid import uuid4
from pathlib import Path, PureWindowsPath


def get_extension(fp):
    ext = fp.split('.')[-1]
    return f".{ext}"


def remove_non_windows_friendly_characters_from_image_names():
    city_fps = os.listdir("images")
    for city_name in city_fps:
        images = os.listdir(os.path.join("images", city_name))

        new_names = [os.path.join("images", city_name, str(uuid4()) + get_extension(fp))
                     for fp in images]
        original_names = [os.path.join("images", city_name, fp)
                          for fp in images]

        image_names = zip(new_names, original_names)
        for new_name, old_name in image_names:
            os.rename(old_name, new_name)


if __name__ == "__main__":
    remove_non_windows_friendly_characters_from_image_names()

# %%
