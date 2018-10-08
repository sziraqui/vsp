import sys
import os
import cv2 as cv
import dlib
import numpy as np


class LipDetector:

    def __init__(self, modelName="None"):
        self.name = "LipDetector"
        self.modelName = modelName
        self.model = None
    
    
    def create_dlib_model(self, datFilePath):
        '''
        set self.model to the created dlib model
        '''
        self.model = dlib.shape_predictor(datFilePath)
        self.modelName = "dlib"

    
    def train_dlib(self, facesLandmarksXml, trainParams, output=None):
        '''
            The xml file has annoted landmarks and relative location of images from xml file's root
            Default values are same as those found in dlib samples and they do not seem to work
        '''
        options = dlib.shape_predictor_training_options()
        options.oversampling_amount = trainParams['oversampling'] #300
        options.nu = trainParams['nu'] #0.05
        options.tree_depth = trainParams['treeDepth'] #2
        options.num_threads = trainParams['threads'] # 4 (=cpu_cores)
        options.be_verbose = trainParams['verbose']
        if output == None:
            output = os.path.abspath(os.path.join('..', 'weights', 'dlib_model.dat'))
        dlib.train_shape_predictor(facesLandmarksXml, output, options)
        print("\nTraining accuracy: {}".format(
            dlib.test_shape_predictor(facesLandmarksXml, output)))


    def test_dlib(self, facesLandmarksXml, datFilePath):
        testing_xml_path = test_landmarks
        print("Testing accuracy: {}".format(
            dlib.test_shape_predictor(facesLandmarksXml, datFilePath)))

    
    def create_resnet_model(self):
        '''
        set self.model to the created resnet model
        '''
        raise NotImplemented(self.name + " resnet model is not yet implemented")
    

    def train_resnet(self, trainParams):
        raise NotImplementedError(self.name + " train_resnet is not yet implemented")

   
    def test_resnet(self, testParams):
        raise NotImplementedError(self.name + " test_resnet is not yet implemented")


    def detect_landmarks(self, img):
        '''
        Use dlib to detect 48-67 lip landmarks
        '''
        raise NotImplementedError(self.name + " detect_landmarks is not yet implemented")

    
    def detect_bbox(self, img):
        '''
        Use resnet or dlib to detect face bounding box
        '''
        raise NotImplementedError(self.name + " detect_bbox is not yet implemented")

    
    def __str__(self):
        return self.name + ":\nModel = " + self.modelName

