import os
import h5py
from glob import glob
import numpy as np
from cv2 import resize as resizeImage
from .framestream import ImageStream, VideoStream, TranscriptFileStream
from .preprocessing import LipDetector
from .textprocessing import word2binmat, extract_timestamps_and_word


class BatchGenerator(object):
    def __init__(self):
        raise NotImplementedError("Implement in subclass")
    def generate(self, batchSize):
        raise NotImplementedError("Implement in subclass")
    
class WordGenerator(BatchGenerator);

    def __init__(self, lipDetector, params):
        self.featuresPath = params['videos_path']
        self.labelsPath = params['transcript_path']
        self.featureFilesPattern = ['video_file_pattern']
        self.labelFilesPattern = ['transcript_file_pattern']
        self.timeFactor = ['transcript_time_factor']
        self.frameLength = ['frame_length']
        self.frameWidth = ['frame_width']
        self.frameHeigth = ['frame_height']
        self.cachePath = ['cache_path']
        assert isinstance(lipDetector, LipDetector)
        self.lipDetector = lipDetector

        self.mediaList = glob(os.path.join(featuresPath, featureFilesPattern))
        self.textExtension = glob(os.path.join(labelsPath, labelFilesPattern))[0].split('.')[-1]
        self.sampleIDs = list(map(lambda x: ''.join(os.path.basename(x).split('.')[:-1]), mediaList))
        self.sampleIndex = 0
    

    def generate(self, batchSize):

        while True:    
            X = np.zeros((batchSize, self.frameHeigth, self.frameWidth, 3))
            Y = np.zeros((batchSize, 28))
            i = 0
            while j < batchSize:

                cachedFile = os.path.join(self.cachePath, self.sampleIDs[j] + '.hdf5')
                try:
                    with h5py.open(cachedFile, 'r') as f:
                        X[j] = f["features"][:]
                        Y[j] = f["labels"][:]
                except:
                    vidFile = self.mediaList[j]
                    textFile = os.path.join(self.labelsPath, self.sampleIDs[j] + self.textExtension)
                    transcript = []
                    vs = VideoStream()
                    ts = TranscriptFileStream(timeFactor=self.timeFactor)
                    try:
                        vs.sourcePath = vidFile
                        vs.set_source()
                        ts.set_source(textFile)
                    except IndexError:
                        print('Sample index out of range')
                        continue
                    except IOError:
                        print(f'IOError when opening {self.sampleIDs[j]}')
                        continue
                    except Exception:
                        print(f'Error when processing {self.sampleIDs[j]}')
                        continue

                    for line in ts.transcriptLines:
                        wordStart, wordEnd, word = extract_timestamps_and_word(line, ts.timeFactor)
                        transcript.append([wordStart, wordEnd, word])

                    frame_no = 0

                    frames = np.zeros((75, self.frameHeigth, self.frameWidth, 3))
                    img = vs.next_frame()

                    while img is not None:
                        bbox = lipDetector.get_bbox(img)
                        y1, x1, y2, x2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
                        lipImg = img[x1-3:x2+4, y1-3:y2+4,:]

                        lipImg = resizeImage(lipImg, (self.frameWidth, self.frameHeigth))
                        frames[frame_no] = lipImg
                        img = vs.next_frame()
                        frame_no+=1
                        silentClip = np.concatenate(
                            (frames[transcript[0][0]: transcript[0][1]],
                            frames[transcript[-1][0]: transcript[-1][1]]))

                    # Pair word with frames all of length self.frameLength
                    for w in transcript:
                        length = w[1] - w[0] + 1
                        if w[2] in ["sp", "sil"]:
                            if length == self.frameLength:
                                binmat = word2binmat(w[0], w[1])
                                X[j] = frames[w[0]:w[1]+1]
                                Y[j] = binmat
                            else:
                                continue
                        elif length <= self.frameLength:
                            X[j] = np.zeros((self.frameLength, self.frameHeigth, self.frameWidth, 3))
                            Y[j] = np.zeros((self.frameLength, 28))
                            s1 = 7 - length//2
                            s2 = 7 + int(length/2 + 0.5)
                            if s1>0:
                                X[-1][:s1,:,:] = silentClip[:s1]
                                Y[-1][:s1,:] = word2binmat(0, s1-1)
                            X[-1][s1:s2,:,:] = frames[s1:s2]
                            Y[-1][s1:s2,:] = word2binmat(w[0], w[1], w[2])
                            if s2<self.frameLength:
                                X[-1][s2:,:,:] = silentClip[:self.frameLength-s2]
                                Y[-1][s2:,:] = word2binmat(s2, 14)
                with h5py.File(f"../datasets/grid_word_{sample_start}-{sample_end}_100x50.hdf5", "w") as f:
                    f.create_dataset("features", data=X, dtype='i1', compression="gzip", compression_opts=4)
                    f.create_dataset("labels", data=Y, dtype='i1', compression='gzip', compression_opts=4)
