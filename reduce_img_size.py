from PIL import Image
import os
from tqdm import tqdm


def main():
    # take images from data/SIAR
    # resize them to 128x128
    # save them to data/SIAR_128

    def resize_image(image_path, size=(128, 128)):
        image = Image.open(image_path)
        image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    for dir in tqdm(os.listdir("data/SIAR")):
        if not os.path.isdir(f"data/SIAR/{dir}"):
            continue
        for file in os.listdir(f"data/SIAR/{dir}"):
            # if file is a directory or not a png file continue
            if os.path.isdir(file) or not file.endswith(".png"):
                continue
            image = resize_image(f"data/SIAR/{dir}/{file}")
            if not os.path.exists(f"data/SIAR_128/{dir}"):
                os.makedirs(f"data/SIAR_128/{dir}")
            image.save(f"data/SIAR_128/{dir}/{file}")
            # print(f"saved {file}")


if __name__ == "__main__":
    main()
