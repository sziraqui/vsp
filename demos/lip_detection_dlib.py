#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath('..'))
from modules.framestream import VideoStream
from modules.preprocessing import LipDetectorDlib
from skvideo.io import FFmpegWriter
import dlib
from modules.utils import Log, add_bbox, imshow

vs = VideoStream()
try:
    vs.sourcePath = sys.argv[1]
    vs.set_source()
except IndexError:
    Log.error('No video source provided')
    sys.exit(1)

lipDetector = LipDetectorDlib()
lipDetector.model_from_file(os.path.join(os.path.abspath('..'), 'weights','shape_predictor_68_face_landmarks.dat'))

FPS = 25
vs.BUFFER_SIZE = FPS * 3

img = vs.next_frame()
outfile = os.path.join(os.path.dirname(vs.sourcePath), 'out_' + os.path.basename(vs.sourcePath))
out = FFmpegWriter(outfile)
while img is not None:

    bbox = lipDetector.get_bbox(img)
    img = add_bbox(img, bbox, (0, 255, 0))
    out.writeFrame(img)
    img = vs.next_frame()

out.close()
