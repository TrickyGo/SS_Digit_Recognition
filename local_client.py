#!/usr/bin/python
# -*- coding: UTF-8 -*-
import datetime
import requests
from io import BytesIO
import time
import json
import os


def recognize_captcha(test_path, save_path, image_suffix):
    image_file_name = 'captcha.{}'.format(image_suffix)

    with open(test_path, "rb") as f:
        content = f.read()

    # 识别
    s = time.time()
    url = "http://127.0.0.1:6000/b"
    files = {'image_file': (image_file_name, BytesIO(content), 'application')}
    r = requests.post(url=url, files=files)
    e = time.time()

    # 识别结果
    print("接口响应: {}".format(r.text))
    predict_text = json.loads(r.text)["value"]
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("【{}】 耗时：{}ms 预测结果：{}".format(now_time, int((e-s)*1000), predict_text))

    # 保存文件
    img_name = "{}_{}.{}".format(predict_text, str(time.time()).replace(".", ""), image_suffix)
    path = os.path.join(save_path, img_name)
    with open(path, "wb") as f:
        f.write(content)
    print("============== end ==============")

basedir = os.getcwd()
with open(basedir + "\\server_config.json", "r") as f:
    sample_conf = json.load(f)

img_list = os.listdir(basedir+"\\test")
print(img_list)

test_path = basedir + "\\test\\x0.jpg"  # 测试识别的图片路径
image_suffix = sample_conf["image_suffix"]  # 文件后缀
recognize_captcha(test_path, save_path, image_suffix)