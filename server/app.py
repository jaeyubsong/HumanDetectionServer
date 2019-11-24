import sys
from mmdet.apis import init_detector, inference_detector, show_result
from flask import Flask, request, jsonify
import numpy as np
import cv2

config_file = '../configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
checkpoint_file = '../checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
app = Flask(__name__)

@app.route('/')
def index():
  return 'Flask working'


@app.route('/checkObject', methods=['POST'])
def checkObject():
  string_img = request.data
  nparr = np.fromstring(string_img, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  result = inference_detector(model, img)
  end_result=show_result(img, result, model.CLASSES, out_file='result.jpg')
  myResponse = {'result': False}
  print(end_result)
  for elem in end_result:
    if elem['label'] == 'person':
      myResponse = {'result': True}
      break
  response = jsonify(myResponse)
  return response


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)
