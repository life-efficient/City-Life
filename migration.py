# %%
import os
import re
import string
from pathlib import Path, PureWindowsPath


def remove_non_windows_friendly_characters_from_image_names():
    city_fps = os.listdir("images")
    for city_name in city_fps:
        images = os.listdir(os.path.join("images", city_name))

        new_names = [os.path.join("images", city_name, PureWindowsPath(fp))
                     for fp in images]
        original_names = [os.path.join("images", city_name, fp)
                          for fp in images]

        image_names = zip(new_names, original_names)
        for new_name, old_name in image_names:
            print(new_name)
            print(old_name)
            print()
            # old_name = image_name

            # filename = image_name.split('/')[-1]
            # filename = Path(image_name)
            # windows_friendly_path = PureWindowsPath(filename)

            # pattern = re.compile('[\W_]+')

            # filename = pattern.sub('', filename)
            # if old_name != windows_friendly_path:
            #     print('name changed')
            #     print(old_name)
            #     print(windows_friendly_path)
            #     print()

            # image_name = image_name.replace('|', '')
            # image_name = image_name.replace('&', '')
            # image_name = image_name.replace(':', '')
            # image_name = image_name.replace('?', '')
            # image_name = image_name.replace('>', '')
            # os.rename(old_name, image_name)


if __name__ == "__main__":
    remove_non_windows_friendly_characters_from_image_names()

# %%
