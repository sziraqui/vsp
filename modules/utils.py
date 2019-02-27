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
from PIL import Image
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing.text import Tokenizer


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


def add_rect(image, x1, y1, x3, y3, color):
    x2, y2 = x1, y3
    x4, y4 = x3, y1
    r = np.array([y1, y2, y3, y4], dtype=int)
    c = np.array([x1, x2, x3, x4], dtype=int)
    p = polygon_perimeter(r, c, shape=image.shape, clip=True)
    img = image.copy()
    set_color(img, p, color)
    return img

'''
    Returns top-left and bottom-right cordinates of dlib.rectangle
'''
def bbox2points(bbox):
    x1, y1 = bbox.tl_corner().x, bbox.tl_corner().y
    x3, y3 = x1 + bbox.width()-1, y1 + bbox.height()-1
    return x1,y1,x3,y3
'''
    image: ndarray as RGB ints
    bbox: dlib.rectangle defining bounding box
    color: tuple of (R,G,B) ints
'''
def add_bbox(image, bbox, color):
    x1, y1, x3, y3 = bbox2points(bbox)
    return add_rect(image, x1, y1, x3, y3, color)


def imshow(image):
    plt.imshow(image)
    plt.show()


def image_resize(img, h, w):
    return np.array(Image.fromarray(img).resize((w,h), resample=Image.BICUBIC))

'''
    Get filename part of path without extension
'''
def get_filename(filepath):
    return ''.join(path.basename(filepath).split('.')[:-1])


def shuffle_together(X, Y, seed=-1):
    if seed < 0:
        seed = np.random.randint(0, 2**(32 - 1) - 1)
    rstate = np.random.RandomState(seed)
    rstate.shuffle(X)
    rstate = np.random.RandomState(seed)
    rstate.shuffle(Y)


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
    try:
        config['video_list'] = path.abspath(path.join(rootdir, config['video_list']))
        config['video_list'] = glob(config['video_list'])
    except KeyError:
        pass
    try:
        config['transcript_list'] = path.abspath(path.join(rootdir, config['transcript_list']))
        config['transcript_list'] = glob(config['transcript_list'])
    except KeyError:
        pass
    try:
        config['cache_dir'] = path.abspath(path.join(rootdir, config['cache_dir']))
    except KeyError:
        pass

    try:
        config['tokenizer'] = path.abspath(path.join(rootdir, config['tokenizer']))
    except KeyError:
        pass
    return config


def load_tokenizer(file_path):
    with open(file_path) as f:
        js = json.dumps(json.load(f))
        tokenizer = tokenizer_from_json(js)
        return tokenizer


def build_grid_tokenizer():
    tokenizer = Tokenizer(oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    sp_tokens = ['<start>', '<end>', '<pad>', '<unk>']
    commands = ['bin', 'lay', 'place', 'set']
    colors = ['blue', 'green', 'red', 'white']
    prepos = ['at', 'by', 'in', 'with']
    letters = list('abcdefjhijklmnopqrstuvxyz') # 'w' is not present in grid
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    adverbs = ['again', 'now', 'please', 'soon']
    tokenizer.fit_on_texts(
        sp_tokens + commands + colors + prepos + letters + digits + adverbs
    )
    
    return tokenizer

    '''
        Assumming 'dirname/basename' part of file path is unique,
        we use the filename to identify and link video and corresponding transcript file
    '''
def get_sample_ids(file_list):
        
    def path2id(filepath):
        filename = get_filename(filepath)
        directory = path.basename(path.dirname(filepath))
        return path.join(directory, filename)
    return list(map(path2id, file_list))

