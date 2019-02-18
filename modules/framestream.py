from skimage.io import imread
from skimage.color import rgba2rgb, gray2rgb
from skimage.transform import resize
from skvideo.io import FFmpegReader
import numpy as np
import glob
import os
from .textprocessing import extract_timestamps_and_word, wordExpansion, word2ints
from .textprocessing import CHAR_SPACE, CODE_SPACE
from .utils import Log

class StreamInterface:
    def __init__(self, params):
        self.buffer = []
        try:
            self.BUFFER_SIZE = params['frame_length']
        except KeyError:
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
    def __init__(self, sourcePath=None, params={}):
        StreamInterface.__init__(self, params)
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
                frame = imread(fileName, plugin='matplotlib')
                frame = ImageStream.force_to_rgb(frame)
                self.buffer.append(frame)
                self.lastIndex+=1

    '''
        Force any image to be in RGB, even if grayscale or RGBA
    '''
    @staticmethod
    def force_to_rgb(frame):
        # frame is either GRAY, RGB or RGBA
        if ImageStream.isGray(frame):
            return gray2rgb(frame)
    
        if ImageStream.isRGBA(frame):
            return rgba2rgb(frame)
        return frame


    @staticmethod
    def isRGBA(frame):
        return len(frame.shape) == 3 and frame.shape[2] == 4
    

    @staticmethod
    def isGray(frame):
        return len(frame.shape) == 1


    """
        dim = (height, width)
    """
    @staticmethod
    def scale_source(frame, dim):
        frame = resize(frame, dim, order=1, mode='reflect')
        return frame


    def __str__(self):
        return self.name + "\nsource = " + str(self.sourcePath) + "\nBuffer size:" + str(len(self.buffer))


"""
    Read input from a video file or camera device
    Buffer frames for faster io
"""
class VideoStream(ImageStream):

    def __init__(self, sourcePath=None, params={}):
        ImageStream.__init__(self, sourcePath, params)
        self.name = "VideoStream"
        self.sourcePath = sourcePath
        try:
            self.fps = params['fps']
        except KeyError:
            self.fps = 25
        self.set_source()
    

    def set_source(self):
        if self.sourcePath != None and os.path.isfile(self.sourcePath):
            ffmpegInputOpt = {'-r': str(self.fps)}
            ffmpegOutputOpt = {'-r': str(self.fps)}
            self.stream = FFmpegReader(self.sourcePath, inputdict=ffmpegInputOpt, outputdict=ffmpegOutputOpt)


    def buffer_frames(self):
        # frame buffering is handled by ffmpeg
        pass


    def next_frame(self):
        # stream.nextFrame() is a generator
        for frame in self.stream.nextFrame():
            return ImageStream.force_to_rgb(frame)

    
    def __str__(self):
        return self.name + "\nsource = " + str(self.sourcePath)

"""
    Streams transcript words and converts transcript into array of ints
    See modules.textprocessing.wordExpansion function description for details of array representation
"""
class TranscriptFileStream(StreamInterface):
    """
        ignoreList: list of word to be considered as silent portions of video
    """
    def __init__(self, sourcePath=None, timeFactor=1, ignoreList=['']):
        StreamInterface.__init__(self)
        self.name = "TranscriptFileStream"
        self.transcriptLines = []
        self.timeFactor = timeFactor
        self.ignoreList = ignoreList
        if sourcePath != None:
            self.set_source(sourcePath)
    

    def set_source(self, sourcePath):
        self.sourcePath = sourcePath
        with open(self.sourcePath, 'r') as f:
            for line in f.readlines():
                self.transcriptLines.append(line.strip())
        
    
    def buffer_frames(self):
        self.buffer = [CODE_SPACE]*self.BUFFER_SIZE
        for line in self.transcriptLines:
            wordStart, wordEnd, word = extract_timestamps_and_word(line, self.timeFactor)
            if word in self.ignoreList:
                word = CHAR_SPACE
            expandedWord = wordExpansion(wordStart, wordEnd-1, word)
            self.buffer[wordStart: wordEnd] = word2ints(expandedWord)
            #self.buffer[wordEnd-1] = CODE_SPACE # to separate two consecutive words
            self.lastIndex = self.BUFFER_SIZE - 1
    

    def next_frame(self):
        if len(self.buffer) == 0:
            self.buffer_frames()
        self.lastIndex-=1
        if 0 < self.lastIndex < len(self.buffer):
            return self.buffer[len(self.buffer) - self.lastIndex - 1]
        else:
            None
