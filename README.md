# SS_Digit_Recognition

## 部署部分（8/26更新）
  1. flask_server.py（需要读取cnn_best.h5） ：在服务器上使用python3解释器运行，运行后将使用6000端口接收request
  2. local_client.py : 本地读取验证码图片，然后调用服务器接口
  3. server_config.json 配置文件，放在同一目录下即可
  
## 实验部分：
### 文件说明：
  1.cnn_captcha_break.0.5.py 执行后：首先定义CNN模型，然后读取同目录下的cnn_best.h5参数，然后将同目录下的test文件夹所有图片读入并print识别结果。图片可为任意大小的.jpg或.png（可以截个屏放在test里）, 为了提高检查率，请尽量使图片接近37*120（验证码原图大小）。
  
  2.cnn_best.h5 是 <1.cnn_captcha_break.0.5.py> 所需要读取的网络参数（我已经训练的差不多的，其实并未充分训练，可以继续）。
  
  
  3.captcha_break_test_0.4.ipynb 是模型的训练过程。会读取train目录下的灰度图像进行训练，然后生成<2.cnn_best.h5>。
  
  4.requirements.txt 是执行 <1.cnn_captcha_break.0.5.py>的所需libs

### 使用方法:
  cnn_capcha_break.py ：将网络参数文件<cnn_best.h5>和<cnn_capcha_break.py>放于同一文件夹下，所需识别的图片放入（test文件夹下），然后使用python解释器执行它（需提前安装好需要导入的包），观察打印结果。
  
  captcha_break_test_0.5.ipynb 使用方法：jupyter notebook
