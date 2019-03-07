from skimage.io import imread
from skimage.color import rgba2rgb, gray2rgb
from skimage.transform import resize
from skvideo.io import FFmpegReader
import numpy as np
import glob
import os
from .textprocessing import extract_timestamps_and_word, wordExpansion, word2ints
from .textprocessing import CHAR_SPACE, CODE_SPACE
from .utils import Log, add_rect, bbox2points, image_resize
from .preprocessing import LipDetectorDlib
import dlib

class StreamInterface:
    def __init__(self):
        self.buffer = []
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
        StreamInterface.__init__(self)
        try:
            self.BUFFER_SIZE = params['frame_length']
        except KeyError:
            self.BUFFER_SIZE = 75
        self.lastIndex = -1
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
        self.stream = None
        try:
            self.fps = params['fps']
        except KeyError:
            self.fps = 25
        self.set_source(sourcePath)
    

    def set_source(self, sourcePath):
        if sourcePath != None and os.path.isfile(sourcePath):
            ffmpegInputOpt = {'-r': str(self.fps)}
            ffmpegOutputOpt = {'-r': str(self.fps)}
            self.stream = FFmpegReader(sourcePath, inputdict=ffmpegInputOpt, outputdict=ffmpegOutputOpt)


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
    def __init__(self, sourcePath=None, buffer_size=75, timeFactor=1, ignoreList=['']):
        StreamInterface.__init__(self)
        self.name = "TranscriptFileStream"
        self.BUFFER_SIZE = buffer_size
        self.transcriptLines = []
        self.timeFactor = timeFactor
        self.ignoreList = ignoreList
        if sourcePath != None:
            self.set_source(sourcePath)
    

    def set_source(self, sourcePath):
        self.sourcePath = sourcePath
        self.transcriptLines = []
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
    

    def next(self):
        if len(self.buffer) == 0:
            self.buffer_frames()
        self.lastIndex-=1
        if 0 < self.lastIndex < len(self.buffer):
            return self.buffer[len(self.buffer) - self.lastIndex - 1]
        else:
            None


"""
    VisemeStream performs viseme extraction using a LipDetector
"""
class VisemeStream(VideoStream):

    def __init__(self, sourcePath=None, params={}, visualize=False):
        VideoStream.__init__(self, sourcePath, params)
        self.name = "VisemeStream"
        self.lipDetector = LipDetectorDlib(params['lip_detector_weights'])
        self.BUFFER_SIZE = params['frame_length']
        # frame refers to expected viseme frame after lip detection
        self.frameHeight = params['frame_height']
        self.frameWidth = params['frame_width']
        self.padVertical, self.padHorizontal = self._cal_padding(params['lip_padding'],  self.frameHeight, self.frameWidth)
        self.aspectRatio = self.frameWidth/self.frameHeight
        self.visualize = visualize
        if self.visualize:
            self.win = dlib.image_window()
        self.lastRect = None # initalized in next_frame()


    def _cal_padding(self, pad_percent, height, width):
        padVert = round(height * pad_percent / 100)
        padHorz = round(width * pad_percent / 100)
        return padVert, padHorz


    def _default_rect(self, parentWidth, parentHeight):
        xcenter = parentWidth/2
        ycenter = parentHeight/2
        x1 = round(xcenter - self.frameWidth/2)
        x3 = round(xcenter + self.frameWidth/2)
        y1 = round(ycenter - self.frameHeight/2)
        y3 = round(ycenter + self.frameHeight/2)
        return (x1,y1,x3,y3)
    

    '''
        Transforms coordinates such that the aspect ratio
        matches a fixed aspect ratio along with padding
    '''
    def _normalize_bounds(self, x1, y1, x3, y3):
        if x1 > x3:
            x1, x3 = x3, x1
        if y1 > y3:
            y1, y3 = y3, y1
        width = (x3 - x1)
        height = (y3 - y1)
        xcenter = x1 + width/2
        ycenter = y1 + height/2
        ratio = width/height
        if ratio > self.aspectRatio:
            # width is more, so the increase height
            height = width/self.aspectRatio
        elif ratio < self.aspectRatio:
            # height is more, so increase width
            width = height * self.aspectRatio
        halfWidth = width/2 + self.padHorizontal
        halfheight = height/2 + self.padVertical
        x1 = round(xcenter - halfWidth)
        x3 = round(xcenter + halfWidth)
        y1 = round(ycenter - halfheight)
        y3 = round(ycenter + halfheight)
        return x1, y1, x3, y3

    
    def extract_viseme(self, frame, include_rect=False):
        rect = None
        if self.lastRect is None:
                self.lastRect = self._default_rect(frame.shape[1], frame.shape[0])
        try:
            bbox = self.lipDetector.get_bbox(frame) # throws IndexError when no faces are detected
            x1, y1, x3, y3 = bbox2points(bbox)
            rect = self._normalize_bounds(x1, y1, x3, y3)
        except:
            # Position of bbox changes only slightly in consecutive frames
            rect = self.lastRect # lets use last bbox in case no face was detected
        
        x1,y1,x3,y3 = rect
        if self.visualize:
            testFrame = add_rect(frame,x1,y1,x3,y3, (0, 255,0))
        
        self.lastRect = rect
        
        if self.visualize:
            testFrame = add_rect(testFrame, x1, y1, x3, y3, (255, 0, 0))
            self.win.set_image(testFrame)
        
        lipImg = frame[y1:y3+1, x1:x3+1,:]
        if include_rect:
            return lipImg, rect
        else:
            return lipImg
    

    def next_frame(self, include_original_frame=False, include_rect=False):
        for frame in self.stream.nextFrame():
            rgbFrame = self.force_to_rgb(frame)
            viseme, rect = None, None
            if include_rect:
                viseme, rect = self.extract_viseme(rgbFrame, include_rect=True)
            else:
                viseme = self.extract_viseme(rgbFrame)
            viseme = image_resize(viseme, self.frameHeight, self.frameWidth)
            if include_original_frame and include_rect:
                return viseme, rgbFrame, rect
            elif include_original_frame:
                return viseme, rgbFrame
            else:
                return viseme
        if include_original_frame and include_rect:
            return None, None, None
        elif include_original_frame:
            return None, None
        else:
            return None
    
    def __str__(self):
        return VideoStream.__str__(self)


class WordStream(TranscriptFileStream):

    def __init__(self, sourcePath=None, timeFactor=1, ignoreList=[]):
        TranscriptFileStream.__init__(self, sourcePath, timeFactor=timeFactor, ignoreList=ignoreList)
        self.name = "WordStream"


    def buffer_frames(self):
        self.buffer = []
        for line in self.transcriptLines:
            wordStart, wordEnd, word = extract_timestamps_and_word(line, self.timeFactor)
            if word not in self.ignoreList:
                self.buffer.append(word)
    

    def next(self):
        if len(self.buffer) <= 0:
            self.buffer_frames()
        return self.buffer.pop(0)
        