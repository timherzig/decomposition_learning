import numpy
from PIL import Image
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

def show_examples(img_folder):
    print('in')
    fig = plt.figure(figsize=(50, 10))
    for idx, img_name in enumerate(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, img_name)
        img = np.array(Image.open(img_path))
        print("Image shape: ", img.shape)
        fig.add_subplot(4, 11, idx+1)
        plt.imshow(img)

        hsv_img = np.array(Image.open(img_path).convert("HSV"))
        fig.add_subplot(4, 11, (idx+12))
        plt.imshow(hsv_img[:, :, 0])   

        hsv_img = np.array(Image.open(img_path).convert("HSV"))
        fig.add_subplot(4, 11, (idx+23))
        plt.imshow(hsv_img[:, :, 1])

        hsv_img = np.array(Image.open(img_path).convert("HSV"))
        fig.add_subplot(4, 11, (idx+34))
        plt.imshow(hsv_img[:, :, 2])
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Train variational auto encoder')
    parser.add_argument('--input', required=True, type=str, help='Path to input image folder')


    args = parser.parse_args()
    return args


def main():
    args = get_args()
    show_examples(args.input)

if __name__ == '__main__':
    main()
