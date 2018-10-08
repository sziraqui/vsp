#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.framestream import ImageStream
from modules.framestream import VideoStream

def image_stream_test():
    """
    >>> print(ImageStream())
    ImageStream
    source = None
    Buffer size:0
    >>> ims = ImageStream('../tests/frames')
    >>> ims.fileList
    ['../tests/frames/a1.jpg', '../tests/frames/a2.jpg']
    >>> ims.buffer_frames()
    >>> print(ims)
    ImageStream
    source = ../tests/frames
    Buffer size:2
    >>> img = ims.next_frame()
    >>> img.shape
    (189, 267, 3)
    >>> ims.lastIndex
    0
    >>> print(ims)
    ImageStream
    source = ../tests/frames
    Buffer size:1
    """
    pass


def video_stream_test():
    """
    >>> vs = VideoStream('../tests/video/sample.3gp')
    >>> print(vs)
    VideoStream
    source = ../tests/video/sample.3gp
    Buffer size:0
    >>> vs.buffer_frames()
    >>> print(vs)
    VideoStream
    source = ../tests/video/sample.3gp
    Buffer size:44
    >>> img = vs.next_frame()
    >>> img.shape
    (144, 176, 3)
    >>> print(vs)
    VideoStream
    source = ../tests/video/sample.3gp
    Buffer size:43
    """
    pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()