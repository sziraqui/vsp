import logging
import time
from time import strftime
from time import localtime
from skimage.draw import polygon_perimeter, set_color
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
from os import path

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
    image: ndarray as RGB ints
    bbox: dlib.rectangle defining bounding box
    color: tuple of (R,G,B) ints
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


def parse_config(configFile):
    rootdir = path.abspath(path.dirname(configFile))
    filename = path.basename(configFile)
    config = {}
    with open(path.join(rootdir, filename), 'rt', encoding='utf-8') as f:
        config = json.load(f)
    try:
        modelFile = path.join(rootdir, config['model_file'])
        config['model_file'] = path.abspath(modelFile)
    except KeyError:
        pass
    try:
        modelFile = path.join(rootdir, config['model_file_checkpoint'])
        config['model_file_checkpoint'] = path.abspath(modelFile)
    except KeyError:
        pass
    try:
        logFile = path.join(rootdir, config['log_dir'])
        config['log_dir'] = path.abspath(logFile)
    except KeyError:
        pass
    try:
        detectorWt = path.join(rootdir, config['lip_detector_weights'])
        config['lip_detector_weights'] = path.abspath(detectorWt)
    except KeyError:
        pass
    try:
        config['hdf5_data_list'] = glob(config['hdf5_data_list'])
    except KeyError:
        pass
    try:
        if config['loss_func'] == 'ctc_loss':
            config['loss_func'] = {'ctc_loss': lambda y_true, y_pred: y_pred}
    except:
        pass
    return config