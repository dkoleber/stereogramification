import pyximport
pyximport.install()
from stereogram import get_stereogram

import numpy as np
import cv2
import time


def make_stereogram(filename):
    full_start_time = time.time()
    img = cv2.imread(filename)
    if img is None:
        print('Image load failed')
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.true_divide(img, 255.)

    rows, cols = img.shape

    padding_height = int(max(0, cols - rows) / 2)
    padding_width = int(max(0, rows - cols) / 2)
    img = cv2.copyMakeBorder(img, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, value=(0,0,0))
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    start_time = time.time()
    img, c1, c2 = get_stereogram(img)
    end_time = time.time() - start_time
    print(end_time)

    img = cv2.circle(img, c1, 10, (0, 0, 0), -1)
    img = cv2.circle(img, c2, 10, (0, 0, 0), -1)

    M = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    full_end_time = time.time() - full_start_time
    print(full_end_time)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    make_stereogram('res/test_10.png')
    # make_stereogram('res/squirrel.png')
    # make_stereogram('arrow.png')