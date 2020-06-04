from flask import Flask, render_template , request
import numpy as np
import keras.models
import re
import sys
import os
from scipy.misc import imread,imresize
sys.path.append(os.path.abspath('./model'))
from load import *

app = Flask(__name__)

global model , graph
model , graph = init()

import base64
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png','wb') as output:
         output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('output.png' , mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))
    x = x.reshape(1,28,28,1)
   
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.argmax(out, axis=1)
        return str(response)







if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 5000, debug=True)
