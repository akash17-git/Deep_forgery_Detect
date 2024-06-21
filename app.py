from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tiff'}

model = load_model('tamper_detection_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_ela_image(image_path, quality):
    original_image = Image.open(image_path)
    original_image.save('temp.jpg', 'JPEG', quality=quality)
    temporary_image = Image.open('temp.jpg')
    ela_image = ImageChops.difference(original_image, temporary_image)
    ela_image = ela_image.convert('L')
    ela_image = ImageEnhance.Brightness(ela_image).enhance(30)
    return ela_image

def prepare_image(image_path):
    ela_image = convert_to_ela_image(image_path, 90).resize((128, 128))
    ela_array = np.array(ela_image).flatten() / 255.0
    ela_array = ela_array.reshape(-1, 128, 128, 1)
    return ela_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            ela_image = prepare_image(filepath)
            prediction = model.predict(ela_image)
            result = np.argmax(prediction, axis=1)[0]
            verdict = "Tampered" if result == 1 else "Not Tampered"
            
            return render_template('result.html', verdict=verdict, filename=filename)
    
    return render_template('upload.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
