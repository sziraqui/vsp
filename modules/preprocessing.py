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
        Returns list of tuple of pixel cordinates
        '''
        faceDetector = dlib.get_frontal_face_detector()
        faces = faceDetector(img, 1) # get atleast one face
        lipPoints = []
        landmarks = self.model(img, faces[0]) # we take only one face bounding box
        for i in range(48,68):
            lipPoints.append(landmarks.part(i)) # extract `dlib.point`s 48-68 of jaw region
        
        lipBB = detect_bbox(img, landmarks)
        lipDetection = dlib.full_object_detection(lipBB, lipPoints)
        return lipDetection

    
    def detect_bbox(self, img, landmarks=None):
        '''
            Use dlib to detect lip bounding box
        '''
        if landmarks == None:
            faceDetector = dlib.get_frontal_face_detector()
            faces = faceDetector(img, 1) # get atleast one face
            lipPoints = []
            landmarks = self.model(img, faces[0]) # we take only one face bounding box
        
        lipBB = dlib.rectangle(
            landmarks.part(48).x, # left
            min([landmarks.part(52).y, landmarks.part(48).y, landmarks.part(54).y, landmarks.part(51).y, landmarks.part(52).y]), # top
            landmarks.part(54).x, # right
            max([landmarks.part(57).y, landmarks.part(48).y, landmarks.part(54).y, landmarks.part(56).y, landmarks.part(57).y]) # bottom
        )
        return lipBB

    
    def __str__(self):
        return self.name + ":\nModel = " + self.modelName

