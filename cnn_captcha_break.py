import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import string
import matplotlib.image as mpimg
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from PIL import Image


#define hyperparameters
characters = string.digits
width, height, n_len, n_class = 120, 37, 4, len(characters)

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def cnn_model(n_len=4, width=120, height=37):
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
        for j in range(n_cnn):
            x = Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(2)(x)

    x = Flatten()(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
    model = Model(inputs=input_tensor, outputs=x)
    model.load_weights('cnn_best.h5')
    return model

def recognize():

    #define model and load weights
    model = cnn_model()
    #find images
    img_path = os.getcwd() + "\\test"
    img_list = []
    for file_name in os.listdir(img_path):
        if file_name[-3:] in ['jpg', 'png']:
            img_list.append(file_name)
    #recognize images
    for img_name in img_list:
        img_file = os.path.join(img_path, img_name)
        if img_name[-3:] in ['jpg']:
            img = Image.open(img_file)
            img = img.resize((width, height))
            X = np.array(img) / 255.0

        elif img_name[-3:] in ['png']:
            img = mpimg.imread(img_file)
            X = np.array(img)

        X = X[np.newaxis,:]      
        y_pred = model.predict(X)
        print(img_name,decode(y_pred))

recognize()