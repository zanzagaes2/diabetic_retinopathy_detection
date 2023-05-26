import os

import cv2
import numpy as np

from functools import partial
from multiprocessing import Pool
from PIL import Image, ImageFilter
from pandas import DataFrame
from tqdm import tqdm

from utilities.utilities import df_test, PathConstant

def crop_image(image):
    STRIPE_WIDTH = 32
    BACKGROUND_OFFSET = 10
    MAX_LENGTH_WIDTH_PROPORTION = 0.8

    def get_bbox(image):
        blurred = image.filter(ImageFilter.BLUR)
        ba = np.array(blurred)
        height, width, _ = ba.shape

        bbox = None
        if width > 1.2 * height:
            left_max = ba[:, : width // STRIPE_WIDTH, :].max(axis=(0, 1)).astype(int)
            right_max = ba[:, - width // STRIPE_WIDTH:, :].max(axis=(0, 1)).astype(int)
            max_bg = np.maximum(left_max, right_max)

            foreground = (ba > max_bg + BACKGROUND_OFFSET).astype(np.uint8)
            bbox = Image.fromarray(foreground).getbbox()

            if bbox is not None:
                left, upper, right, lower = bbox
                if right - left < MAX_LENGTH_WIDTH_PROPORTION * height \
                        or lower - upper < MAX_LENGTH_WIDTH_PROPORTION * height:
                    bbox = None
        return bbox

    def square_bbox(img):
        width, heigth = img.size
        left = max((width - heigth) // 2, 0)
        upper = 0
        right = min(width - (width - heigth) // 2, width)
        lower = heigth
        return (left, upper, right, lower)

    bbox = bbox if (bbox := get_bbox(image)) is not None else square_bbox(image)
    return image.crop(bbox)

def resize(image, new_size):
    return image.resize((new_size, new_size))

def graham(image, sigma = 10, pillow_output = True):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma), -4, 128)
    if pillow_output:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

def scaleRadius(img, scale=300):
    x = img[img.shape[0]/2,:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def preprocess_prueba(img,scale=300):
    img_n = scaleRadius(img,scale)
    img_n = cv2.addWeighted(img_n,4,cv2.GaussianBlur(img_n,(0,0),scale/30),-4,128)
    b=np.zeros(img_n.shape)
    cv2.circle(b,(img_n.shape[1]/2,img_n.shape[0]/2),int(scale*0.9),(1,1,1),-1,8,0)
    img_n = img_n*b +128*(1-b)
    return img_n

def crop_gray(image):
    img = np.array(image)
    foreground = (img != [128, 128, 128]).astype(np.uint8)
    bbox = Image.fromarray(foreground).getbbox()
    cropped = image.crop(bbox)
    return cropped


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
