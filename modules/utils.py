import logging
import time
from time import strftime
from time import localtime
from skimage.draw import polygon_perimeter, set_color
import numpy as np
import matplotlib.pyplot as plt

def timeNow():
    return strftime('%Y-%M-%d %H:%m:%2ds', localtime())


def getLogger(className='', level=logging.INFO):
    logger = logging.getLogger(f'module/{className}')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level)
    consoleFormatter = logging.Formatter(f"%(filename)s/{className}[%(levelname)s]:\n %(message)s")
    consoleHandler.setFormatter(consoleFormatter)
    logger.addHandler(consoleHandler)
    return logger

# Global logger
Log = getLogger()


'''
    image: ndarray as RGB floats
    bbox: dlib.rectangle defining bounding box
    color: tuple of (R,G,B) floats
'''
def add_bbox(image, bbox, color):
    c = np.array([bbox.left(), bbox.left(), bbox.right(), bbox.right()], dtype=int)
    r = np.array([bbox.bottom(), bbox.top(), bbox.top(), bbox.bottom()], dtype=int)
    p = polygon_perimeter(r, c, shape=image.shape, clip=True)
    img = image.copy()
    set_color(img, p, color)
    return img


def imshow(image):
    plt.imshow(image)
    plt.show()
