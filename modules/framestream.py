import cv2 as cv
import numpy as np
import glob
import os
from .textprocessing import word2binmat, binmat2word, extract_timestamps_and_word

class StreamInterface:
    def __init__(self):
        self.buffer = []
        self.BUFFER_SIZE = 75
        self.lastIndex = -1
    def next_frame(self):
        raise NotImplementedError("Implement in subclass")
    def buffer_frames(self):
        raise NotImplementedError("Implement in subclass")


"""
    Read input from a folder containing ordered images
    Buffer frames for faster io
"""
class ImageStream(StreamInterface):
    def __init__(self, sourcePath=None):
        StreamInterface.__init__(self)
        self.name = "ImageStream"
        self.sourcePath = sourcePath
        self.fileList = []
        if sourcePath != None:
            self.fileList = glob.glob(os.path.join(self.sourcePath, "*.jpg"))
        

    def next_frame(self):
        if len(self.buffer) < 1:
            self.buffer_frames()
        if len(self.buffer) > 0:
            self.lastIndex-=1
            return self.buffer.pop(0)
        else:
            return None


    def buffer_frames(self):
        for fileName in self.fileList:
            if len(self.buffer) < self.BUFFER_SIZE:
                frame = cv.imread(fileName)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                self.buffer.append(frame)
                self.lastIndex+=1
        

    """
        dim = (height, width)
    """
    def scale_source(self, frame, dim):
        ogDim = frame.shape
        if ogDim[:2] != dim:
            frame = cv.resize(frame, dim)

    
    def __str__(self):
        return self.name + "\nsource = " + str(self.sourcePath) + "\nBuffer size:" + str(len(self.buffer))


"""
    Read input from a video file or camera device
    Buffer frames for faster io
"""
class VideoStream(ImageStream):

    def __init__(self, sourcePath=0):
        ImageStream.__init__(self)
        self.name = "VideoStream"
        self.sourcePath = sourcePath
        self.set_source()
    

    def set_source(self):
        self.stream = cv.VideoCapture(self.sourcePath)


    def buffer_frames(self):
        if not self.stream.isOpened():
            return None
        while self.stream.isOpened() and len(self.buffer) < self.BUFFER_SIZE:
            no_error, frame = self.stream.read()
            if no_error:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                self.buffer.append(frame)
                self.lastIndex+=1
            else:
                break
        self.stream.release()

    
    def __str__(self):
        return self.name + "\nsource = " + str(self.sourcePath) + "\nBuffer size:" + str(len(self.buffer))

"""
    Streams transcript words and converts transcript into binary matrix
    See modules.textprocessing.word2binmat function description for details of binary representation
"""
class TranscriptFileStream(StreamInterface):

    def __init__(self, sourcePath=None, timeFactor=1):
        StreamInterface.__init__(self)
        self.name = "TranscriptFileStream"
        self.transcriptLines = []
        self.timeFactor = timeFactor
        if sourcePath != None:
            self.set_source(sourcePath)
    

    def set_source(self, sourcePath):
        self.sourcePath = sourcePath
        with open(self.sourcePath, 'r') as f:
            for line in f.readlines():
                self.transcriptLines.append(line.strip())
        
    
    def buffer_frames(self):
        self.buffer = np.zeros((self.BUFFER_SIZE, 28))
        for line in self.transcriptLines:
            wordStart, wordEnd, word = extract_timestamps_and_word(line, self.timeFactor) 
            self.buffer[wordStart: wordEnd + 1] = word2binmat(wordStart, wordEnd, word)
        self.lastIndex = self.BUFFER_SIZE - 1
    

    def next_frame(self):
        if len(self.buffer) == 0:
            self.buffer_frames()
        self.lastIndex-=1
        if 0 < self.lastIndex < len(self.buffer):
            return self.buffer[len(self.buffer) - self.lastIndex - 1]
        else:
            None
