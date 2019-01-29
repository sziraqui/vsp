import sys
import os
import cv2 as cv
import dlib
import numpy as np
from .utils import Log


'''
    Common interface for lip detector models like dlib resnet etc
    All methods must be implemented by a lip detector subclass
'''
class LipDetector(object):
    
    def __init__self(self):
        raise NotImplementedError("Implement in subclass")
    def train(self, trainParams):
        raise NotImplementedError("Implement in subclass")
    def test(self, testParams):
        raise NotImplementedError("Implement in subclass")
    def model_from_file(self, modelFilePath):
        raise NotImplementedError("Implement in subclass")
    def get_bbox(self, img):
        raise NotImplementedError("Implement in subclass")
    def __str__(self):
        raise NotImplementedError("Implement in subclass")


'''
    Lip detector that uses dlib's 68-face-landmarks model
'''
class LipDetectorDlib(LipDetector):

    def __init__(self, weightsFile=None):
        self.modelName = "dlib"
        self.model = None
        if weightsFile != None:
            model_from_file(weightsFile)
        else:
            Log.warning('weightsFile was not provided, you must set model yourself by calling model_from_file("path/to/file")')
        self.faceDetector = dlib.get_frontal_face_detector()
    

    '''
        set self.model to the created dlib model
    '''
    def model_from_file(self, datFilePath):
        if os.path.exists(datFilePath) and os.path.isfile(datFilePath):
            self.model = dlib.shape_predictor(datFilePath)
        else:
            raise IOError("Cannot access valid .dat file")

    
    '''
        The xml file has annoted landmarks and relative location of images from xml file's root
        Default values are same as those found in dlib samples and they do not seem to work
    '''
    def train(self, trainParams, output=None):
        facesLandmarksXml = trainParams['annotationsFile']
        options = dlib.shape_predictor_training_options()
        options.oversampling_amount = trainParams['oversampling'] #300
        options.nu = trainParams['nu'] #0.05
        options.tree_depth = trainParams['treeDepth'] #2
        options.num_threads = trainParams['threads'] # 4 (=cpu_cores)
        options.be_verbose = trainParams['verbose']
        if output == None:
            output = os.path.abspath(os.path.join('..', 'weights', 'dlib_model.dat'))
        dlib.train_shape_predictor(facesLandmarksXml, output, options)
        Log.info("\nTraining accuracy: {}".format(
            dlib.test_shape_predictor(facesLandmarksXml, output)))


    def test(self, testParams):
        testing_xml_path = testParams['annotationsFile']
        datFilePath = testParams['modelFilePath']
        Log.info("Testing accuracy: {}".format(
            dlib.test_shape_predictor(facesLandmarksXml, datFilePath)))


    '''
        arg1: image as cv Mat
        returns: dlib.full_object_detection containing 20 lip landmarks
    '''
    def detect_landmarks(self, img):

        faces = self.faceDetector(img, 1) # get atleast one face
        lipPoints = []
        landmarks = self.model(img, faces[0]) # we take only one face bounding box
        for i in range(48,68):
            lipPoints.append(landmarks.part(i)) # extract `dlib.point`s 48-68 of jaw region
        
        lipBB = get_bbox(img, landmarks)
        lipDetection = dlib.full_object_detection(lipBB, lipPoints)
        return lipDetection

    '''
        arg1: image as cv Mat
        arg2: 68-face-landmarks if available
        returns: dlib.rectangle
    '''
    def get_bbox(self, img, landmarks=None):
        '''
            Use dlib to detect lip bounding box
        '''
        if landmarks == None:
            faces = self.faceDetector(img, 1) # get atleast one face
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
        return "Model = " + self.modelName

