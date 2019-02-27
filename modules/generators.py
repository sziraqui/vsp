import os
from os import path
import h5py
from glob import glob
import numpy as np
from copy import deepcopy

from keras.utils import normalize
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

from .framestream import VisemeStream, TranscriptFileStream, WordStream
from .textprocessing import word2ints
from .textprocessing import CODE_BLANK
from .metrics import CTC_LOSS_STR
from .utils import Log, shuffle_together, get_sample_ids, load_tokenizer

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
                shuffle_together(X, Y, self.seed)
                yield X, Y
                del X
                del Y
            else:
                self.dataIndex = (self.dataIndex + 1)%len(self.dataList)
                self.sampleIndex = 0


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


'''
    Creates batches from raw videos and transcripts on-the-fly
    VisemeStream is responsible for viseme extraction
'''
class OnlineGridBatch(GeneratorInterface):

    def __init__(self, config, max_label_len=8, transcript_time_factor=1, transcript_ignore_list=['']):
        GeneratorInterface.__init__(self)
        self.config = config
        self.inputShape = (config['frame_length'], config['frame_height'], config['frame_width'], 3)
        self.seqLen = max_label_len
        '''
            videoDir:
            eg. video_list[0]: /home/sziraqui/vsp-dev/datasets/GRID/videos/s1/bbac9n.mpg
            Then videoDir: /home/sziraqui/vsp-dev/datasets/GRID/videos

            Likewise for textDir
            eg. text_list[0]: /home/sziraqui/vsp-dev/datasets/GRID/transcripts/s1/bbac9n.align
            Then textDir: /home/sziraqui/vsp-dev/datasets/GRID/transcripts
        '''
        self.videoDir = path.dirname(path.dirname(config['video_list'][0]))
        self.textDir = path.dirname(path.dirname(config['transcript_list'][0]))

        self.videoExt = path.basename(config['video_list'][0]).split('.')[-1] # eg 'mpg'
        self.textExt = path.basename(config['transcript_list'][0]).split('.')[-1] # eg 'align' or 'txt'

        self.cache = config['cache_dir']

        self.ids = get_sample_ids(config['video_list'])
        np.random.shuffle(self.ids)

        self.sampleIndex = 0 # current sample index in self.ids
        self.tokenizer = load_tokenizer(config['tokenizer'])
        self.vs = VisemeStream(params=config, visualize=True)
        self.ws = WordStream(timeFactor=transcript_time_factor, ignoreList=transcript_ignore_list)
    

    def next_batch(self, batchSize):
        while True:
            X = np.zeros((batchSize,) + (self.inputShape))
            sentences = []
            i = 0 # current batch index
            while len(sentences) < batchSize:
                try:
                    visemes = self.load_from_cache(self.ids[self.sampleIndex + i])
                    if visemes is None:
                        visemes = self.load_from_disk(self.ids[self.sampleIndex + i])
                        self.save_to_cache(self.ids[self.sampleIndex + i], visemes)
                    sentences.append(self.load_transcript(self.ids[self.sampleIndex + i]))
                    
                except Exception as e:
                    Log.error(repr(e) + 'Error processing sample ' + self.ids[self.sampleIndex + i])
                i+=1
            
            sequences = self.tokenizer.texts_to_sequences(sentences)
            Y = pad_sequences(sequences, value=self.tokenizer.word_index['<pad>'], maxlen=self.seqLen, 
                            dtype='uint8', padding='post', truncating='post')
            X /= 255
            if self.sampleIndex >= len(self.ids) - batchSize:
                self.sampleIndex = 0
            else:
                self.sampleIndex += i
            yield X, Y
            del X
            del Y

   
    def load_from_cache(self, id):
        print('cached:', id)
        try:
            with h5py.File(path.join(self.cache, id + '.h5')) as f:
                viseme = f["feature"][:]
                if viseme.shape != self.inputShape:
                    return None
                else:
                    return viseme
        except (OSError, KeyError):
            return None
    

    def load_from_disk(self, id):
        print('video:', id)
        self.vs.set_source(path.join(self.videoDir, id + '.' + self.videoExt))
        visemes = []
        frame = self.vs.next_frame()
        while frame is not None:
            visemes.append(frame)
            frame = self.vs.next_frame()
        return np.array(visemes)

    
    def load_transcript(self, id):
        print('text:', id)
        self.ws.set_source(path.join(self.textDir, id + '.' + self.textExt))
        self.ws.buffer_frames()
        words = self.ws.buffer
        print('words:',words)
        return ['<start>'] + words + ['end']

    
    def save_to_cache(self, id, visemes):
        filename = path.join(self.cache, id + '.h5')
        os.makedirs(path.dirname(filename), exist_ok=True)
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset("features", data=visemes, dtype='uint8', compression="gzip", compression_opts=4)
            return True
        return False