import os, sys
from flask import Flask, redirect, url_for, request ,render_template, flash, json
from werkzeug import secure_filename
sys.path.insert(0, os.path.abspath('../../'))
from modules.framestream import VideoStream, TranscriptFileStream, VisemeStream
from modules.preprocessing import LipDetectorDlib
from modules.lipreading import SentenceReader
from modules.utils import *

import dlib
from modules.textprocessing import *
import h5py
from glob import glob
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from time import time
from modules.utils import parse_config
import pygame.camera
import pygame.image
import pygame.surfarray
from pygame.locals import *
from skvideo.io import FFmpegWriter

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

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
    font = ImageFont.truetype(font='ubuntu/UbuntuMono-R.ttf', size=24)
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


params = parse_config('../../config/config-example.json')
sr = SentenceReader(params)
lipDetector = LipDetectorDlib(params['lip_detector_weights'])

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DATA_DIR ='static'

app.config['SECRET_KEY'] = '123456789qwertyq'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def ok():
    return render_template("VSP.html", data = None)
    

@app.route('/', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        video= FFmpegWriter("uploads/output.mp4")
        pygame.init()
        pygame.camera.init()
        screen = pygame.display.set_mode((640,480))
        cam = pygame.camera.Camera("/dev/video0",(640,480))
        cam.start()
        running = True
        while running:
            image= cam.get_image()
            pilimagestring = pygame.image.tostring(image,"RGBA",False)
            pilimage = Image.frombytes("RGBA",(640,480),pilimagestring)
            enhancer = ImageEnhance.Brightness(pilimage)
            enhanced_im = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Contrast(enhanced_im)
            enhanced_im = enhancer.enhance(1.5)
            value = np.array(enhanced_im)
            video.writeFrame(value)
            
            screen.blit(image,(0,0))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    cam.stop()
        video.close()
        flash('password recorded!','success')     
        return render_template('VSP.html',data=None)   

    """
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(url_for('ok'))
        else:      
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            flash('File uploaded successfully!','success')
            return render_template('VSP.html', data=None)
"""

@app.route('/process')
def process():
    filelist = glob('uploads/*')
    if len(filelist) == 0:
        return render_template("VSP.html", data=None)
    path = filelist[0]
    X = visualize_predictions(path)
    #X = extract_visemes(path)
    if "soon" in X:
        return render_template("Successful.html")
    else:
        flash('User Not Authenticated','error')
    os.remove(path)
    #output = predict(X[0])
    return render_template("VSP.html", data = X)


def visualize_predictions(vidPath):
    vs = VisemeStream(vidPath, params)
    frameWindow = np.zeros((params['frame_length'], params['frame_height'], params['frame_width'], 3), dtype='uint8')
    viseme_list = [frameWindow[i] for i in range(params['frame_length'])]
    inFps = params['fps']
    outFps = 5
    frameNo = 0
    currViseme, currFrame, rect = vs.next_frame(include_original_frame=True, include_rect=True)
    currText = ''
    win = dlib.image_window()
    pred = ''
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
        currViseme, currFrame, rect = vs.next_frame(include_original_frame=True, include_rect=True)
    return pred

def extract_visemes(path):
    t= time()
    out_img_heigth, out_img_width = 50, 100
    vs = VideoStream(path)

    frame_no = 0
    frames = np.zeros((75, out_img_heigth, out_img_width,3))
    FPS = 25
    vs.BUFFER_SIZE = FPS * 3
    win = dlib.image_window()
    img = vs.next_frame()
    while img is not None:
        win.set_image(img)
        bbox = lipDetector.get_bbox(img)
        
        y1, x1, y2, x2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        lipImg = img[x1-3:x2+4, y1-3:y2+4,:]
        lipImg = resize(lipImg, (out_img_heigth, out_img_width), order=1, mode='reflect')        
        #lipImg = VideoStream.scale_source(lipImg, (out_img_heigth, out_img_width))
        
        frames[frame_no] = lipImg
        img = vs.next_frame()
        frame_no+=1
        if frame_no == vs.BUFFER_SIZE:
            break
   
        img = vs.next_frame()
    
    
    t = time() - t
    print(t)    
    output = sr.predict_sentence(frames)
    t =time()
    t = time() - t
    print(t)

    return output
    
def init():
    sr = SentenceReader(params)
    return 


@app.route('/output')
def display():
	SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
	json_url = os.path.join(SITE_ROOT, 'static', 'example_1.json')
	data = json.load(open(json_url))
	
	return render_template("Output.html", data = data)

if __name__ == '__main__':
   app.run(debug = True)
