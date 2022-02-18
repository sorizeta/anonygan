import os
import sys
import cv2
import dlib

import numpy as np
from PIL import Image


SCALE_FACTOR = 1
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__),
    "./shape_predictor_68_face_landmarks.dat")

predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    h, w, c = im.shape
    d = dlib.rectangle(0, 0, w, h)
    # Get the landmarks/parts for each face in box d.
    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, d).parts()])
    return landmarks


def read_im_and_landmarks(fname):
    im = open_oriented_im(fname)
        
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    l = get_landmarks(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im, l


def open_oriented_im(im_path):
    im = Image.open(im_path)
    if hasattr(im, '_getexif'):
        exif = im._getexif()
        if exif is not None and 274 in exif:
            orientation = exif[274]
            im = apply_orientation(im, orientation)
    img = np.asarray(im)#.astype(np.float32) / 255.
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    im, l = read_im_and_landmarks(sys.argv[1])
    plt.imshow(im)
    for p in l:
        print(p[0,0], p[0,1])
        plt.scatter(p[0,0], p[0,1], c='g')
    plt.show()