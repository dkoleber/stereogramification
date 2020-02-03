import numpy as np
import cv2


def flip_RB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def flip_BR(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def decimal_to_pixel(img):
    return (img*255.).astype(np.uint8)

def pixel_to_decimal(img):
    image = img
    if type(image) == 'list':
        image = np.array(image)
    return image.astype(np.float32)/255.

def image_3d_to_4d(img):
    image = None
    if not isinstance(img, np.ndarray):
        image = np.array(img)
    else:
        image = img
    if image.shape[0] != 1 and len(image.shape) == 3:
        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

    return image

def image_4d_to_3d(img):
    image = None
    if isinstance(img, np.ndarray) and img.shape[0] == 1 and len(img.shape) == 4:
        image = np.reshape(img, (img.shape[1], img.shape[2], img.shape[3]))
    else:
        image = img
    return image

def save_image(img, path):
    image = image_4d_to_3d(img)
    image = decimal_to_pixel(image)
    image = flip_RB(image)
    cv2.imwrite(path, image)

def open_image(path):
    image = cv2.imread(path)
    image = flip_BR(image)
    image = pixel_to_decimal(image)
    return image

def load_image(path):
    image = image_3d_to_4d(open_image(path))
    # image = np.moveaxis(image, 1, 2)
    return image