import os
import numpy as np 
import glob 
from PIL import Image, ImageFilter 
from flask import Flask, render_template, request
from neuralnetwork import NeuralNet
# from neuralnetwork import NeuralNet

import cv2
from scipy import ndimage
import math
import numpy as np
import math
import _pickle as pickle
from scipy import io as spio


# %tensorflow_version 1.x
import tensorflow as tf

# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
from scipy import io as spio
import time 



UPLOAD_FOLDER = '/static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)

def getInputs(arr):
    # function returns list of input values from image value list
    return 0.01 + (np.asfarray(arr[1:]) * float(99/100) * float(1/255))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def imageProcessing(filepath):
    """
    Function for pre-processing images
    """
    imagedata = []
    for img in glob.glob(filepath):
        edit_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        edit_image = cv2.resize(255-edit_image, (28, 28))

        cv2.imwrite(img, edit_image)
        (thresh, edit_image) = cv2.threshold(edit_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        while np.sum(edit_image[0]) == 0:
            edit_image = edit_image[1:]

        while np.sum(edit_image[:,0]) == 0:
            edit_image = np.delete(edit_image,0,1)

        while np.sum(edit_image[-1]) == 0:
            edit_image = edit_image[:-1]

        while np.sum(edit_image[:,-1]) == 0:
            edit_image = np.delete(edit_image,-1,1)

        rows,cols = edit_image.shape
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            edit_image = cv2.resize(edit_image, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            edit_image = cv2.resize(edit_image, (cols, rows))

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        edit_image = np.lib.pad(edit_image,(rowsPadding,colsPadding),'constant')
        shiftx,shifty = getBestShift(edit_image)
        shifted = shift(edit_image,shiftx,shifty)
        edit_image = shifted
        edit_image = edit_image.flatten()
        imagedata.append(edit_image)
    return imagedata

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')
        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))
        
            fn = "./static/uploads/"+str(file.filename)
            uploaded_image = imageProcessing(fn)
            inputs = uploaded_image[0]
            # outputs = nn.query(inputs)

            outputs = nn.query(inputs, nn)
            label1 = np.argmax(outputs)
            # lbl1 = np.argmax(outputs)


            # balanced_map = dict()
            # for i in range(47):
            #     if i < 10:
            #         balanced_map[i] = i+48
            #     elif (i>=10) and (i < 36):
            #         balanced_map[i] = i+55
            #     elif (i>= 38) and (i < 43):
            #         balanced_map[i] = i+62
            #     else:
            #         if i == 36:
            #             balanced_map[i] = 97
            #         if i == 37:
            #             balanced_map[i] = 98
            #         if i == 43:
            #             balanced_map[i] = 110
            #         if i == 44:
            #             balanced_map[i] = 113
            #         if i == 45:
            #             balanced_map[i] = 114
            #         if i == 46:
            #             balanced_map[i] = 116


            # print(balanced_map)
            # label_ascii = balanced_map[lbl1]
            # label1 = chr(label_ascii)

            # alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
            # balanced_map = dict()
            # for i in range(47):
            #     if i < 10:
            #         balanced_map[i] = str(i)
            #     else:
            #         balanced_map[i] = alph[i-28]
            
            # label1 = balanced_map[lbl1]

            # print("outputs", outputs)

            # # print("predict: ", lbl1,",", label_ascii, ",", label1 )
            # print("predict: ", lbl1, ",", label1 )

            
            
            
            return render_template('upload.html',
                                   msg='Successfully processed',
                                   extracted_text=label1, 
                                   img_src=(UPLOAD_FOLDER + file.filename))
    elif request.method == 'GET':
        return render_template('upload.html')

if __name__ == '__main__':
    input_nodes, hidden_nodes, output_nodes, layer_num, learning_rate = 784, 200, 10, 1, 0.1
    # nn = NeuralNet(input_nodes, hidden_nodes, output_nodes, layer_num, learning_rate)
    nn = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate, 'Sigmoid')
    nn.load()
    # nn.load("11")
    app.run(debug=True)
    