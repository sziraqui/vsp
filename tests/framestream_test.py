#!/usr/bin/env python3

import sys
import os
from os import path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.framestream import VisemeStream
from modules.utils import parse_config, Log

import numpy as np

import unittest

class TestVisemeStream(unittest.TestCase):

    def setUp(self):
        self.config = parse_config(path.join('..', 'config', 'config-example.json'))
        self.videoWithROI = path.join('..', 'tests', 'samples', 'images', 'celeb_face.jpeg')
        self.videoMissedROI = path.join('..', 'tests', 'samples', 'images', 'no_roi.jpg')
        self.visemeStream = VisemeStream(params=self.config)
        self.outShape = (self.config['frame_height'], self.config['frame_width'], 3)
    

    def test_viseme_stream_visible_roi(self):
        self.visemeStream.set_source(self.videoWithROI)
        viseme, ogFrame, rect = self.visemeStream.next_frame(include_original_frame=True, include_rect=True)
        while viseme is not None:
            self.assertEqual(viseme.shape, self.outShape)
            viseme = self.visemeStream.next_frame()

        
if __name__ == '__main__':
    unittest.main()