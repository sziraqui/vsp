#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath('..'))
from modules.framestream import VisemeStream
from modules.preprocessing import LipDetectorDlib
from skvideo.io import FFmpegWriter
import dlib
from modules.utils import Log, parse_config, imshow

params = parse_config(sys.argv[2])
vs = VisemeStream(sys.argv[1], params, visualize=True)
img = vs.next_frame()
win = dlib.image_window()
outfile = os.path.join(os.path.dirname(vs.sourcePath), 'out_viseme_' + os.path.basename(vs.sourcePath))
out = FFmpegWriter(outfile)
while img is not None:
    win.set_image(img)
    out.writeFrame(img)
    img = vs.next_frame()
out.close()
