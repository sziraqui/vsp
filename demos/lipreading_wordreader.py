import sys, os
sys.path.insert(0, os.path.abspath('..'))
from modules.lipreading import WordReader
from modules.framestream import VideoStream
from modules.textprocessing import binmat2word
from modules.preprocessing import LipDetectorDlib
from keras.models import load_model
import dlib
from time import time
import numpy as np
import cv2 as cv

trainedModelPath = sys.argv[1]
videoPath = sys.argv[2]

params = {
    'resume':False,
    'frame_length': 15,
    'optimizer': 'adam', 
    'loss_func':'categorical_crossentropy',
    'batch_size': 2,
    'epochs': 3,
    'validation_split': 0.2,
    'model_file': f'../weights/lipnet_15_{time()}.hdf5'
}
wr = WordReader(params)
wr.model = load_model(trainedModelPath)
vs = VideoStream(videoPath)
vs.BUFFER_SIZE = wr.frameLength
lipDetector = LipDetectorDlib()
lipDetector.model_from_file(os.path.join(os.path.abspath('..'), 'weights','shape_predictor_68_face_landmarks.dat'))

inputWin = dlib.image_window()
outputWin = dlib.image_window()

img = vs.next_frame()
features = np.zeros((5,wr.frameLength, wr.frameHeight, wr.frameWidth, 3))
batchNo = 0
frameNo = 1
fullPrediction = ''
while img is not None:
    inputWin.set_image(img)
    outputWin.set_image(img)
    bbox = lipDetector.get_bbox(img)
    outputWin.add_overlay(bbox)
    y1, x1, y2, x2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
    lipImg = img[x1-3:x2+4, y1-3:y2+4,:]
    features[batchNo][frameNo-1] = cv.resize(lipImg, (wr.frameWidth, wr.frameHeight))/255
    if frameNo == wr.frameLength:
        frameNo = 0
        batchNo+=1
    frameNo+=1
    img = vs.next_frame()


labels = wr.model.predict(features)
print(labels)
for label in labels:
    print(binmat2word(label))

