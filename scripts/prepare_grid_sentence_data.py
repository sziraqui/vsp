#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from modules.framestream import VideoStream, TranscriptFileStream
from modules.preprocessing import LipDetectorDlib
from modules.textprocessing import *

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
#print(sample_ids)

lipDetector = LipDetectorDlib()
lipDetector.model_from_file(os.path.join(os.path.abspath(
        '..'), 'weights', 'shape_predictor_68_face_landmarks.dat'))

chdict = {ch:i for i,ch in enumerate('abcdefghijklmnopqrstuvwxyz ')}

X = []
Y = []
LabelLength = []
InputLength = []

for i in tqdm(range(sample_size)):
    transcript = []
    vs = VideoStream()
    ts = TranscriptFileStream(timeFactor=0.001)
    #print(f"Processing file: {sample_ids[i]}")
    try:
        vs.sourcePath = video_list[i]
        vs.set_source()
        ts.set_source(os.path.join(dataset_path, txt_path, sample_ids[i]+'.align'))
    except IndexError:
        print('Index out of range')
        sys.exit(1)
    except IOError:
        print(f'IOError when opening {sample_ids[i]}.align')
    except Exception:
        print(f'Error when processing {sample_ids[i]}')


    for line in ts.transcriptLines:
        wordStart, wordEnd, word = extract_timestamps_and_word(line, ts.timeFactor)
        if word not in ['sp', 'sil']:
            transcript.append(word)
    sentence = list(map(lambda ch: chdict[ch] , ' '.join(transcript)))
    labels = np.concatenate([np.array(sentence), np.zeros(32 - len(sentence)) + CODE_BLANK])

    #print(labels)

    frame_no = 0

    frames = np.zeros((75, out_img_heigth, out_img_width,3))
    img = vs.next_frame()

    #win = dlib.image_window()
    while img is not None:
        bbox = lipDetector.get_bbox(img)
        y1, x1, y2, x2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        lipImg = img[x1-3:x2+4, y1-3:y2+4,:]
        
        lipImg = vs.scale_source(lipImg, (out_img_width, out_img_heigth))
        #win.set_image(lipImg)
        frames[frame_no] = lipImg
        img = vs.next_frame()
        frame_no+=1
    X.append(frames)
    InputLength.append(75)
    Y.append(labels)
    LabelLength.append(len(sentence))

X = np.array(X)
X = np.round(X * 255).astype(numpy.uint8)

outFile = f"../datasets/grid_sentences_ctc_{sample_start}-{sample_end}.hdf5"
with h5py.File(outFile, "w") as f:
    print(f"Saving to {outFile}...")
    f.create_dataset("features", data=X, dtype='uint8', compression="gzip", compression_opts=4)
    f.create_dataset("labels", data=Y, dtype='uint8', compression='gzip', compression_opts=4)
    f.create_dataset("input_length", data=InputLength, dtype='uint8', compression='gzip', compression_opts=4)
    f.create_dataset("label_length", data=LabelLength, dtype='uint8', compression='gzip', compression_opts=4)
print("Done")
