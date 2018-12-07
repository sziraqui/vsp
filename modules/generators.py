import os
import h5py
from glob import glob
import numpy as np
from keras.utils import normalize
from .textprocessing import word2ints
from .textprocessing import CODE_BLANK

class GeneratorInterface(object):
    def __init__(self):
        pass
    def next_batch(self, batchSize):
        raise NotImplementedError("Implement in subclass")


class SimpleGenerator(GeneratorInterface):

    def __init__(self, params, seed=-1):
        self.dataList = params['hdf5_data_list']
        self.frameLength = params['frame_length']
        self.frameWidth = params['frame_width']
        self.frameHeigth = params['frame_height']
        self.sampleSize = params['sample_size']
        self.batchSize = params['batch_size']
        self.dataIndex = 0
        self.sampleIndex = 0
        self.seed = seed
        # pre-allocate memory for one batch

    def next_batch(self, batchSize):
        servedSamples = 0
        np.random.shuffle(self.dataList)
        while True:
            X = np.zeros((self.batchSize, self.frameLength, self.frameHeigth, self.frameWidth, 3))
            Y = np.zeros((self.batchSize, self.frameLength, CODE_BLANK+1))
            try:
                with h5py.File(self.dataList[self.dataIndex], 'r') as f:
                    X[:] = f["features"][self.sampleIndex : self.sampleIndex + batchSize]
                    Y[:] = f["labels"][self.sampleIndex : self.sampleIndex + batchSize]
            except IndexError:
                self.dataIndex = (self.dataIndex + 1) % len(self.dataList)
                self.sampleIndex = 0
                continue
            X = normalize(X, axis=1)
            self.shuffle_together(X, Y, self.seed)
            yield X, Y
            del X
            del Y
            self.sampleIndex += batchSize
            servedSamples+=batchSize


    def shuffle_together(self, X, Y, seed=-1):
        if seed < 0:
            seed = np.random.randint(0, 2**(32 - 1) - 1)
        rstate = np.random.RandomState(seed)
        rstate.shuffle(X)
        rstate = np.random.RandomState(seed)
        rstate.shuffle(Y)
