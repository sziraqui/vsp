#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from modules.framestream import VideoStream, TranscriptFileStream
from modules.preprocessing import LipDetectorDlib
from modules.textprocessing import *
from modules.utils import Log

import dlib
import h5py
from glob import glob
import numpy as np
from tqdm import tqdm

out_img_width = 100
out_img_heigth = 50

dataset_path = sys.argv[1]
sample_start = int(sys.argv[2])
sample_size = int(sys.argv[3])
sample_end = sample_start + sample_size - 1

videos_path = 'videos'
txt_path = 'transcripts'

video_list = sorted(glob(os.path.join(dataset_path, videos_path) + '/*.mpg'))[sample_start: sample_end+1]
video_list.sort()

sample_ids = list(map(lambda x: ''.join(os.path.basename(x).split('.')[:-1]), video_list))

lipDetector = LipDetectorDlib()
lipDetector.model_from_file(os.path.join(os.path.abspath(
        '..'), 'weights', 'shape_predictor_68_face_landmarks.dat'))


X = []
Y = []

for i in tqdm(range(sample_size)):
    transcript = []
    vs = VideoStream()
    ts = TranscriptFileStream(timeFactor=0.001)
    
    try:
        vs.sourcePath = video_list[i]
        vs.set_source()
        ts.set_source(os.path.join(dataset_path, txt_path, sample_ids[i]+'.align'))
    except IndexError:
        Log.error('Index out of range')
        sys.exit(1)
    except IOError:
        Log.error(f'IOError when opening {sample_ids[i]}.align')
    except Exception:
        Log.error(f'Error when processing {sample_ids[i]}')

    for line in ts.transcriptLines:
        wordStart, wordEnd, word = extract_timestamps_and_word(line, ts.timeFactor)
        transcript.append([wordStart, wordEnd, word])

    frame_no = 0

    frames = np.zeros((75, out_img_heigth, out_img_width,3))
    img = vs.next_frame()

    while img is not None:
        bbox = lipDetector.get_bbox(img)
        y1, x1, y2, x2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        lipImg = img[x1-3:x2+4, y1-3:y2+4,:]
        
        lipImg = cv.resize(lipImg, (out_img_width, out_img_heigth))
        #win.set_image(lipImg)
        frames[frame_no] = lipImg
        img = vs.next_frame()
        frame_no+=1
    silentClip = np.concatenate(
        (frames[transcript[0][0]: transcript[0][1]],
        frames[transcript[-1][0]: transcript[-1][1]]))
    
    for w in transcript:
        try:
            length = w[1] - w[0] + 1
            if w[2] in ["sp", "sil"]:
                if length == 15:
                    binmat = word2binmat(w[0], w[1], w[2])
                    X.append(frames[w[0]:w[1]+1])
                    Y.append(binmat)
                else:
                    continue
            elif length <= 15:
                X.append(np.zeros((15, out_img_heigth, out_img_width, 3)))
                Y.append(np.zeros((15,CODE_BLANK+1)))
                s1 = 7 - length//2
                s2 = 7 + int(length/2 + 0.5)
                #Log.debug(length,s1,s2)
                if s1>0:
                    X[-1][:s1,:,:] = silentClip[:s1]
                    Y[-1][:s1,:] = word2binmat(0, s1-1, CHAR_SPACE)
                X[-1][s1:s2,:,:] = frames[s1:s2]
                Y[-1][s1:s2,:] = word2binmat(w[0], w[1], w[2])
                if s2<15:
                    X[-1][s2:,:,:] = silentClip[:15-s2]
                    Y[-1][s2:,:] = word2binmat(s2, 14, CHAR_SPACE)
        except AssertionError:
            continue

X = np.array(X, dtype=np.uint8)
outFile = f"../datasets/grid_words15_{sample_start}-{sample_end}.hdf5"
with h5py.File(outFile, "w") as f:
    Log.debug(f"Saving to {outFile}...")
    f.create_dataset("features", data=X, dtype='uint8', compression="gzip", compression_opts=4)
    f.create_dataset("labels", data=Y, dtype='uint8', compression='gzip', compression_opts=4)
Log.debug("Done")
