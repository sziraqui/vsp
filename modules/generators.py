import os
import h5py
from glob import glob
import numpy as np
from keras.utils import normalize
from .textprocessing import word2ints
from .textprocessing import CODE_BLANK
from .metrics import CTC_LOSS_STR
from .utils import Log

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
        self.dataIndex = 0
        self.sampleIndex = 0
        self.seed = seed
        # pre-allocate memory for one batch

    def next_batch(self, batchSize):
        endIndex = 0
        startIndex = endIndex
        availableSamples = 0
        np.random.shuffle(self.dataList)
        while True:
            X = np.zeros((batchSize, self.frameLength, self.frameHeigth, self.frameWidth, 3))
            Y = np.zeros((batchSize, self.frameLength, CODE_BLANK+1))
            try:
                with h5py.File(self.dataList[self.dataIndex], 'r') as f:
                    availableSamples = f["labels"][self.sampleIndex : self.sampleIndex + batchSize].shape[0]
                    startIndex = endIndex%batchSize
                    endIndex += availableSamples
                    if  endIndex > batchSize:
                        endIndex = batchSize
                    X[startIndex:endIndex] = f["features"][self.sampleIndex : self.sampleIndex + endIndex - startIndex]
                    Y[startIndex:endIndex] = f["labels"][self.sampleIndex : self.sampleIndex + endIndex - startIndex]
            except IOError as ioe:
                self.dataIndex = (self.dataIndex + 1) % len(self.dataList)
                self.sampleIndex = 0
                endIndex = 0
                Log.info(repr(ioe))
                continue
            if endIndex >= batchSize:
                endIndex = 0
                self.sampleIndex += availableSamples
                X = normalize(X, axis=1)
                self.shuffle_together(X, Y, self.seed)
                yield X, Y
                del X
                del Y
            else:
                self.dataIndex = (self.dataIndex + 1)%len(self.dataList)
                self.sampleIndex = 0


    def shuffle_together(self, X, Y, seed=-1):
        if seed < 0:
            seed = np.random.randint(0, 2**(32 - 1) - 1)
        rstate = np.random.RandomState(seed)
        rstate.shuffle(X)
        rstate = np.random.RandomState(seed)
        rstate.shuffle(Y)


class BatchForCTC(SimpleGenerator):

    def next_batch(self, batchSize):
        endIndex = 0
        startIndex = endIndex
        availableSamples = 0
        np.random.shuffle(self.dataList)
        while True:
            X = np.zeros((batchSize, self.frameLength, self.frameHeigth, self.frameWidth, 3))
            Y = np.zeros((batchSize, 32))
            InpLen = np.zeros(batchSize, dtype='int32')
            LabelLen = np.zeros(batchSize, dtype='int32')
            try:
                with h5py.File(self.dataList[self.dataIndex], 'r') as f:
                    availableSamples = f["labels"][self.sampleIndex : self.sampleIndex + batchSize].shape[0]
                    startIndex = endIndex%batchSize
                    endIndex += availableSamples
                    if  endIndex > batchSize:
                        endIndex = batchSize
                    X[startIndex:endIndex] = f["features"][self.sampleIndex : self.sampleIndex + endIndex - startIndex]
                    Y[startIndex:endIndex] = f["labels"][self.sampleIndex : self.sampleIndex + endIndex - startIndex]
                    InpLen[startIndex:endIndex] = f["input_length"][self.sampleIndex: self.sampleIndex + endIndex - startIndex]
                    LabelLen[startIndex:endIndex] = f["label_length"][self.sampleIndex: self.sampleIndex + endIndex - startIndex]
            except IOError as ioe:
                self.dataIndex = (self.dataIndex + 1) % len(self.dataList)
                self.sampleIndex = 0
                endIndex = 0
                Log.info(repr(ioe))
                continue
            if endIndex >= batchSize:
                endIndex = 0
                self.sampleIndex += availableSamples
                X = normalize(X, axis=1)
                #self.shuffle_together(X, Y, self.seed)
                inputs = {
                    'input': X,
                    'label_input': Y,
                    'input_length': InpLen,
                    'label_length': LabelLen
                }
                outputs = {CTC_LOSS_STR: np.zeros((batchSize, 32))}
                yield inputs, outputs
                del X
                del Y
            else:
                self.dataIndex = (self.dataIndex + 1)%len(self.dataList)
                self.sampleIndex = 0