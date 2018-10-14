#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath('..'))
from modules.framestream import VideoStream
from modules.preprocessing import LipDetectorDlib
import cv2 as cv
import dlib

vs = VideoStream()
try:
    vs.sourcePath = sys.argv[1]
    vs.set_source()
except IndexError:
    print('No video source provided')
    sys.exit(1)

lipDetector = LipDetectorDlib()
lipDetector.model_from_file(os.path.join(os.path.abspath('..'), 'weights','shape_predictor_68_face_landmarks.dat'))

FPS = vs.stream.get(cv.CAP_PROP_FPS)
vs.BUFFER_SIZE = FPS * 5
win = dlib.image_window()
img = vs.next_frame()
out = cv.VideoWriter('out_' + os.path.basename(vs.sourcePath), cv.VideoWriter_fourcc('X','V','I','D'), 15, (480,360))
while img is not None:
    win.set_image(img)
    bbox = lipDetector.get_bbox(img)
    win.clear_overlay()
    win.add_overlay(bbox)
    #cv.rectangle(img, (bbox.left(), bbox.top()), (bbox.right(), bbox.bottom()), (255,0,0), 1)
    out.write(img)
    img = vs.next_frame()
out.release()
