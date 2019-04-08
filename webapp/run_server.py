import os
import sys
import json
import inspect
import random
import string
import numpy as np
from glob import glob
from PIL import Image
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask import send_from_directory, render_template
from werkzeug.utils import secure_filename

from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential
from keras.applications import xception, resnet50

import tensorflow as tf

# import parent dir
CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, PARENT_DIR)

from facenet.align import detect_face

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


g_backbone = None
g_classifier = None
g_dog_names = []
g_facenet = []
g_dog_detector = None
g_session = None


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_filename(filename):
    random_name = ''.join(random.choices(string.ascii_letters, k=6))
    filename_list = filename.split('.')
    filename_list[0] += random_name
    filename = '.'.join(filename_list)
    return filename


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'uploaded_img' not in request.files:
            flash('No file part')
            return redirect(request.url)
        user_img = request.files['uploaded_img']
        if user_img.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if user_img and allowed_file(user_img.filename):
            filename = secure_filename(user_img.filename)
            filename = unique_filename(filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            user_img.save(save_path)
    return jsonify({"url": url_for("get_uploaded_image", filename=filename), "filename": filename})


@app.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_image(filename):
    # check the original jpeg exist or not
    real_file_name = os.path.join(os.path.abspath('./' + app.config['UPLOAD_FOLDER']), filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(real_file_name))


@app.route('/run_inference/', methods=['GET'])
def run_inference():
    """Run inference

    Params: GET
        img_file: the uploaded image file name, e.g. human_4.jpg
    """
    params = request.args.to_dict()
    img_file = params['filename']
    img_file = os.path.join(os.path.abspath('./' + app.config['UPLOAD_FOLDER']), img_file)

    message = "oops, looks like can't find either human or dog :("
    pred_breed = predict_breed(img_file)
    if detect_dog(img_file):
        message = f"Detected dog, predict breed: <br/>{pred_breed}"
        if detect_human_face(img_file):
            message += "<br/>Detected human as well!"
    elif detect_human_face(img_file):
        message = f"Detected human, looks like: <br/>{pred_breed}"

    return jsonify({"message": message})


def predict_breed(image_file):
    # get features from backbone
    global g_backbone
    global g_classifier
    img = preprocessing(image_file)
    img_features = g_backbone.predict(xception.preprocess_input(img))
    # obtain predicted vector
    pred = g_classifier.predict(img_features)
    # return dog breed that is predicted by the model
    return g_dog_names[np.argmax(pred)]


def preprocessing(image_file):
    # loads RGB image as PIL.Image.Image type
    img = Image.open(image_file).convert('RGB').resize((224, 224))
    img = np.asarray(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(img, axis=0)


def detect_human_face(img_file):
    global g_facenet
    minsize = 20  # minimum size of face
    threshold = [0.85, 0.85, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    img = np.asarray(Image.open(img_file).convert('RGB'))
    p, r, o = g_facenet
    bounding_boxes, _ = detect_face.detect_face(img, minsize, p, r, o, threshold, factor)

    return len(bounding_boxes) > 0


def detect_dog(img_file):
    global g_dog_detector
    img = resnet50.preprocess_input(preprocessing(img_file))
    prediction = np.argmax(g_dog_detector.predict(img))
    return ((prediction <= 268) & (prediction >= 151))


def setup_app(app):
    """Initialize DNN objects
    """
    global g_backbone
    global g_classifier
    global g_dog_names
    global g_facenet
    global g_dog_detector
    global g_session
    # get dog names
    with open('../data/class_names.json', 'r') as fp:
        g_dog_names = json.load(fp)['dog_names']
    print(f"Loaded dog names: {len(g_dog_names)}")
    # initialize facenet
    g_session = tf.Session()
    p, r, o = detect_face.create_mtcnn(g_session, '../facenet/align/')
    g_facenet = [p, r, o]
    print("Loaded FACENET.")
    # intialize dog detector
    g_dog_detector = resnet50.ResNet50(weights='imagenet')
    g_dog_detector._make_predict_function()
    print("Loaded dog detector")
    # load DNN
    g_backbone = xception.Xception(weights='imagenet', include_top=False)
    g_backbone._make_predict_function()
    g_classifier = Sequential()
    g_classifier.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    g_classifier.add(Dense(133, activation='softmax'))
    g_classifier.load_weights('../saved_models/weights.best.Xception.hdf5')
    g_classifier._make_predict_function()
    print("Load dog breed classifier")


if __name__ == "__main__":
    setup_app(app)
    app.run(host='0.0.0.0', port=3001, debug=False)
