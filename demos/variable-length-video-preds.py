#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import dlib
from PIL import Image, ImageDraw, ImageFont
from skvideo.io import FFmpegWriter

from modules.framestream import VisemeStream
from modules.utils import parse_config
from modules.lipreading import SentenceReader
from modules.utils import add_rect, image_resize


def annotate_frame(frame, viseme_list, rect, text, size=(640, 480)):
    color = (0,255,0)
    (x1,y1,x3,y3) = rect
    frame = add_rect(frame, x1,y1,x3,y3, color)
    frame = image_resize(frame, size[1], size[0])
    frame = insert_viseme_window(frame, viseme_list)
    frame = insert_text(frame, text)
    return frame


def insert_text(frame, text):
    h, w = frame.shape[0], frame.shape[1]
    bgLayer = Image.new('RGBA', (w, round(h/10)), color='black')
    txtLayer = Image.new('RGBA', (w, round(h/10)), color=(255,255,255,0))
    font = ImageFont.truetype(font='Lato-Medium.ttf', size=24)
    ImageDraw.Draw(txtLayer, mode='RGBA').multiline_text((8,8), text, font=font, fill='white', align='center')
    out = np.array(Image.alpha_composite(bgLayer, txtLayer).convert('RGB'))
    final = np.concatenate([frame, out], axis=0)
    return final

def insert_viseme_window(frame, viseme_list):
    fw = frame.shape[1]
    visemes = np.concatenate(viseme_list, axis=1)
    vw = visemes.shape[1]
    vh = visemes.shape[0]
    visemes = add_rect(visemes, 0,0, viseme_list[0].shape[1], viseme_list[0].shape[0], (255,255,0))
    ratio = vw/vh
    vw = fw
    vh = int(round(vw/ratio))
    visemes = image_resize(visemes, vh, vw)
    new_frame = np.concatenate([frame, visemes], axis=0)
    return new_frame

config = parse_config(sys.argv[1])
vidPath = sys.argv[2]

sr = SentenceReader(config)
vs = VisemeStream(vidPath, config)
vidOut = FFmpegWriter(os.path.join(os.path.abspath('..'), 'out_viseme_' + os.path.basename(vidPath)))

frameWindow = np.zeros((config['frame_length'], config['frame_height'], config['frame_width'], 3), dtype='uint8')
viseme_list = [frameWindow[i] for i in range(config['frame_length'])]
inFps = config['fps']
outFps = int(sys.argv[3])
frameNo = 0
currViseme, currFrame, rect = vs.next_frame(include_original_frame=True, include_rect=True)
currText = ''
win = dlib.image_window()

while (currViseme is not None):
    annotatedFrame = annotate_frame(currFrame, viseme_list[::-1][:10], rect, currText)
    win.set_image(annotatedFrame)
    if len(viseme_list) >= frameWindow.shape[0]:
        viseme_list.pop(0)
    frameNo+=1
    print(frameNo, end=': ')
    viseme_list.append(currViseme)
    if frameNo%(max(round(inFps/outFps), 1)) == 0:
        frameWindow[:len(viseme_list)] = np.array(viseme_list)
        pred = sr.predict_sentence(frameWindow)
        currText = pred.replace('_', ' ')
        print(pred)
    else:
        print()
    try:
        currViseme, currFrame, rect = vs.next_frame(include_original_frame=True, include_rect=True)
    except IndexError:
        rect = (0,0,0,0)
    vidOut.writeFrame(annotatedFrame)

vidOut.close()
