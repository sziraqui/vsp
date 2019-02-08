import os, sys
from flask import Flask, redirect, url_for, request ,render_template, flash, json
from werkzeug import secure_filename

sys.path.insert(0, os.path.abspath('../../'))
from modules.framestream import VideoStream, TranscriptFileStream
from modules.preprocessing import LipDetectorDlib
import cv2 as cv
import dlib
from modules.textprocessing import *
import h5py
from glob import glob
import numpy as np
from tqdm import tqdm


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DATA_DIR ='static'

app.config['SECRET_KEY'] = '123456789qwertyq'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def ok():
   return render_template("VSP.html", data = null)

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      if f.filename == '':
            flash('No selected file')
            return redirect(url_for('ok'))
      else:      
      		filename = secure_filename(f.filename)
      		f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
      		flash('File uploaded successfully!','success')
      		return redirect(url_for('ok'))



   			




def process():
	filelist = glob('uploads/*')
    path = filelist[0]
    X = lipdetector(path)
	output = predict(X)
	output_path='/static/output.txt'
	file = open(output_path,'w')
	file.write(output)
	file.close()
	return render_template("VSP.html",data = output)




def lipdetector(path):
        vs = VideoStream()
    try:
        vs.sourcePath = path
        vs.set_source()
    except IndexError:
        print('No video source provided')
        sys.exit(1)

    lipDetector = LipDetectorDlib()
    lipDetector.model_from_file(os.path.join(os.path.abspath('..'), 'weights','shape_predictor_68_face_landmarks.dat'))
    frame_no = 0
    X = []

    frames = np.zeros((75, out_img_height, out_img_width,3))
    FPS = vs.stream.get(cv.CAP_PROP_FPS)
    vs.BUFFER_SIZE = FPS * 5
    win = dlib.image_window()
    img = vs.next_frame()
    out = cv.VideoWriter('out_' + os.path.basename(vs.sourcePath), cv.VideoWriter_fourcc('X','V','I','D'), 15, (480,360))
    while img is not None:
        win.set_image(img)
        bbox = lipDetector.get_bbox(img)
        
        y1, x1, y2, x2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        lipImg = img[x1-3:x2+4, y1-3:y2+4,:]
        
        lipImg = cv.resize(lipImg, (out_img_width, out_img_height))
        #win.set_image(lipImg)
        frames[frame_no] = lipImg
        img = vs.next_frame()
        frame_no+=1
   
        img = vs.next_frame()
     X.append(frames)
     return X   
    
def predict(frames):
    params = {
    "resume": True,
    "initial_epoch": 0,
    "frame_length": 75,
    "frame_width": 100,
    "frame_height": 50,
    "hdf5_data_list": glob("../datasets/*sentence*.hdf5"),
    "generator_queue_size": 1, 
    "loss_func": {'ctc_loss': lambda ytrue,ypred: ypred},
    "sample_size": 128,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 1e-03,
    "learning_beta1": 0.9,
    "learning_beta2": 0.999,
    "learning_decay": 1e-08,
    "validation_split": 0.2,
    "log_dir": "../logs"
    }    
    params["model_file"] = f"../weights/lipnet_ctc_s{params['sample_size']}_b{params['batch_size']}_e{params['epochs']}_{time()}.hdf5"
    params['sample_size'] = 64
    params['batch_size'] = 32
    params['epochs'] = 1
    params['model_file_checkpoint'] = "../weights/lipnet_ctc_s128_b32_e300_18-12-2018-19-02-24.hdf5"
    sr = SentenceReader(params)
    generator = BatchForCTC(params)
    X,Y, il,ll = tuple([None]*4)
    samples = 32
    for i,o in generator.next_batch(1):
        X,Y,il,ll = i['input'], i['label_input'], i['input_length'], i['label_length']
        break
    print(X.shape)
    print(Y.shape)
    output = sr.predict_sentence(frames)
    return output

@app.route('/output')
def display():
	SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
	json_url = os.path.join(SITE_ROOT, 'static', 'example_1.json')
	data = json.load(open(json_url))
	
	return render_template("Output.html", data = data)

if __name__ == '__main__':
   app.run(debug = True)