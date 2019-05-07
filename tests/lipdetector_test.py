#!/usr/bin/env python3

import sys
import os
from os import path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.preprocessing import LipDetectorDlib
from modules.utils import parse_config, bbox2points

from PIL import Image
import numpy as np

import unittest

class TestLipDetector(unittest.TestCase):

    def setUp(self):
        self.config = parse_config(path.join('..', 'config', 'config-example.json'))
        self.imageWithROI = np.array(Image.open(path.join('..', 'tests', 'samples', 'images', 'celeb_face.jpeg')))
        self.imageWithoutROI = np.array(Image.open(path.join('..', 'tests', 'samples', 'images', 'no_roi.jpg')))
        self.detector = LipDetectorDlib(self.config['lip_detector_weights'])
    

    def test_dlib_lipdetector_valid_face(self):
        points = bbox2points(self.detector.get_bbox(self.imageWithROI))
        self.assertEqual(points, (73, 169, 126, 192))


    def test_dlib_lipdetector_invalid_face(self):
        self.assertRaises(IndexError, self.detector.get_bbox, self.imageWithoutROI)

        
if __name__ == '__main__':
    unittest.main()
    