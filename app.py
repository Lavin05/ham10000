from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from skimage.io import imread
from skimage import transform
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import sys
import os
import glob
import re
import numpy as np

app = Flask(__name__)

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


# Load the trained model
model = load_model(r'C:\Users\jprup\Downloads\New folder\New folder\resnet_model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

SKIN_CLASSES = {
  'akiec': 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  'bcc': 'Basal Cell Carcinoma',
  'bkl': 'Benign Keratosis',
  'df': 'Dermatofibroma',
  'mel': 'Melanoma',
  'nv': 'Melanocytic Nevi',
  'vasc': 'Vascular skin lesion'
}

def predict_pose(image_path):
    img = imread(image_path)
    img = transform.resize(img, (170, 170, 3))
    img = np.expand_dims(img, axis=0) 

    predictions = model.predict(img)

    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    
    predicted_class_description = SKIN_CLASSES[predicted_class]

    return predicted_class_description
    
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join( basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        pose = predict_pose(file_path)
    
        return pose
    return None

if __name__ == '__main__':
    app.run(debug=True)
