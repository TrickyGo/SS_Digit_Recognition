# -*- coding: UTF-8 -*-
"""
构建flask接口服务
接收 files={'image_file': ('captcha.jpg', BytesIO(bytes), 'application')} 参数识别验证码
"""
import json
from io import BytesIO
import os
import time
from flask import Flask, request, jsonify, Response
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as k

# Flask对象
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
graph = tf.get_default_graph()

# 默认使用CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with open(basedir + "\\server_config.json", "r") as f:
    sample_conf = json.load(f)
# 配置参数
image_height = sample_conf["image_height"]
image_width = sample_conf["image_width"]
max_captcha = sample_conf["max_captcha"]
api_image_dir = sample_conf["api_image_dir"]
model_save_dir = sample_conf["model_save_dir"]
image_suffix = sample_conf["image_suffix"]  # 文件后缀
use_labels_json_file = sample_conf['use_labels_json_file']

if use_labels_json_file:
    with open("tools/labels.json", "r") as f:
        char_set = f.read().strip()
else:
    char_set = sample_conf["char_set"]

def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/b', methods=['POST'])
def up_image():
    if request.method == 'POST' and request.files.get('image_file'):
        timec = str(time.time()).replace(".", "")
        file = request.files.get('image_file')
        img = file.read()
        img = BytesIO(img)
        img = Image.open(img, mode="r")
        # username = request.form.get("name")
        print("接收图片尺寸: {}".format(img.size))
        s = time.time()
        value = rec_image(img, model)
        e = time.time()
        print("识别结果: {}".format(value))
        # # 保存图片
        # print("保存图片： {}{}_{}.{}".format(api_image_dir, value, timec, image_suffix))
        # file_name = "{}_{}.{}".format(value, timec, image_suffix)
        # file_path = os.path.join(api_image_dir + file_name)
        # img.save(file_path)
        result = {
            'time': timec,   # 时间戳
            'value': value,  # 预测的结果
            'speed_time(ms)': int((e - s) * 1000)  # 识别耗费的时间
        }
        img.close()
        return jsonify(result)
    else:
        content = json.dumps({"error_code": "1001"})
        resp = response_headers(content)
        return resp

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([char_set[x] for x in y])

def cnn_model(n_len=4, width=120, height=37):
    input_tensor = k.layers.Input((height, width, 1))
    x = input_tensor
    for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
        for j in range(n_cnn):
            x = k.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = k.layers.BatchNormalization()(x)
            x = k.layers.Activation('relu')(x)
        x = k.layers.MaxPooling2D(2)(x)

    x = k.layers.Flatten()(x)
    x = [k.layers.Dense(len(char_set), activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
    model = k.models.Model(inputs=input_tensor, outputs=x)
    model.load_weights('cnn_best.h5')
    return model

def rec_image(img, model):
    img = img.convert('L')
    img = img.resize((image_width, image_height))
    X = np.array(img).reshape(image_height,image_width,1) / 255.0
    X = X[np.newaxis,:]      
    global graph
    sess = tf.Session(graph=graph)
    with graph.as_default():
        with sess.as_default() as sess:
            sess.run(tf.global_variables_initializer())
            model.load_weights('cnn_best.h5')
            y_pred = model.predict(X)
    return decode(y_pred)

if __name__ == '__main__':
    #define model and load weights
    model = cnn_model()

    app.run(
        host='0.0.0.0',
        port=6000,
        debug=True
    )
