import os
from flask import Flask, redirect, url_for, request ,render_template, flash, json
from werkzeug import secure_filename
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DATA_DIR ='static'

app.config['SECRET_KEY'] = '123456789qwertyq'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def ok():
   return render_template("VSP.html")

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
      return  redirect(url_for('ok'))

@app.route('/output')
def display():
	SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
	json_url = os.path.join(SITE_ROOT, 'static', 'example_1.json')
	data = json.load(open(json_url))
	
	return render_template("Output.html", data = data)

if __name__ == '__main__':
   app.run(debug = True)