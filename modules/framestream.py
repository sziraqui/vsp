import cv2 as cv
import numpy as np
import glob
import os
    
"""
    Read input from a folder containing ordered images
    Buffer frames for faster io
"""
class ImageStream:
    def __init__(self, sourcePath=None):
        self.name = "ImageStream"
        self.sourcePath = sourcePath
        self.buffer = []
        self.BUFFER_SIZE = 750
        self.lastIndex = -1
        self.fileList = []
        if sourcePath != None:
            self.fileList = glob.glob(os.path.join(self.sourcePath, "*.jpg"))
        


    def next_frame(self):
        if len(self.buffer) > 0:
            self.lastIndex-=1
            return self.buffer.pop(0)


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
            self.stream.open()
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
