import torch
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from model.RetinaNet import RetinaNet
from test.test_retinanet import test_retinanet
from utils.data_utils import IEC_dataset_preprocessing, normalize

import time

DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)

CORS(app, support_credentials=True)

@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/UploadFile', methods=['POST'])
def upload_file():
    f = request.files.get('raw')
    f.seek(0x200) # 512
    values = np.frombuffer(f.read(), dtype=np.int16)
    signals = np.array([values[0::2], values[1::2]])[0].reshape((1,-1,1))[:, :, :4992].astype(float)
    signals = IEC_dataset_preprocessing(signals, smooth=False, dns=True).cuda()
    signals = normalize(signals)

    net = RetinaNet(3).cuda()
    net.load_state_dict(torch.load("weights/retinanet_best_IEC.pkl"))
    _, _, pred_signals = test_retinanet(net, signals, 4992, visual=False)

    return jsonify({'raw': signals.squeeze().tolist(), "label": pred_signals.tolist()})

def run():
    app.run(host='0.0.0.0')