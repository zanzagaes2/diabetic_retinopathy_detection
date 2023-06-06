import cv2
import os
import numpy as np

from functools import partial
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

from utilities.utilities import df_test

def gaussian_blur(image, scale, sigma = 10):
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma), -4, 128)
    b = np.zeros(image.shape)
    cv2.circle(b, (image.shape[1] // 2, image.shape[0] // 2), int(scale * 0.9),(1,1,1),-1,8,0)
    image = image * b + 128 * (1-b)
    return image

def scale_radius(image, scale):
    x = image[image.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(image, (0,0), fx=s, fy=s)

def crop_gray(image):
    img = np.array(image)
    foreground = (img != [128, 128, 128]).astype(np.uint8)
    bbox = Image.fromarray(foreground).getbbox()
    cropped = image.crop(bbox)
    return cropped

def preprocess(image):
    scale = 300
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = scale_radius(image, scale)
    image = gaussian_blur(image, scale)
    image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    image = crop_gray(image)
    return image

def process_image(image_name, operations, directory, convert_directory):
    try:
        with Image.open(os.path.join(directory, f'{image_name}')) as image:
            converted = image
            for f in operations:
                converted = f(converted)
            converted.save(os.path.join(convert_directory, f'{image_name}'))
    except FileNotFoundError:
        print(f"File not fund: {image_name}")

def main(operations, df, directory, convert_directory, n_process = 25):
    if not os.path.exists(convert_directory):
        os.makedirs(convert_directory)

    f = partial(process_image, operations = operations,
                directory = directory, convert_directory = convert_directory)
    with Pool(n_process) as pool:
        with tqdm(total = len(df)):
            pool.map(f, df.image)


if __name__ == '__main__':
    fs = [
        crop_gray
    ]
    main(fs, df_test(), '/home/jupyter-tfg2223retina/dataset/test',
         '/home/jupyter-tfg2223retina/dataset/test/cropped')


