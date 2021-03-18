import base64
import os
import numpy as np
import pandas as pd
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def get_model():
    global model 
    model = load_model('best_model.h5')

def preprocess_image(image, size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(size)
    image = img_to_array(image)
    #image = image - [116.23950232394932, 116.23950232394932, 116.23950232394932]
    image = np.expand_dims(image, axis=0)
    image = keras.applications.vgg16.preprocess_input(image)

    return image

get_model()


#@app.route('/batch')
def batch_prediction():
    names = pd.read_csv('../Test.csv')
    names = names.sort_values(by='ID')

    tests = names['ID'].values

    means = []

    for test in tests:
        image = Image.open('../test/test/'+test+'.png')
        processed_image = preprocess_image(image, size=(224,224))
        pred = model.predict(processed_image)
        means.append(pred[0][0])

    names['LABEL'] = np.array(means).astype('float64')
    names = names.drop('filename', axis=1)

    names.to_csv('submission2.csv', index=False, sep=',')
    batch_prediction()

    #return render_template('predict.html')




@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == "POST":
        img = request.files['xray']
        img.save(img.filename)
        image = Image.open(img.filename)
        #print(np.asarray(image).mean(axis=(0,1)))
        processed_image = preprocess_image(image, size=(224,224))
        #print(processed_image.mean(axis=(0,1), dtype='float64'))
        pred = model.predict(processed_image)
        context = float("{:.3f}".format(pred[0][0]*100))
        if os.path.exists(img.filename):
            os.remove(img.filename)

        return render_template('predict.html', text=context)
    else:
        return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True)

