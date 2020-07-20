import torch
import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from model.RetinaNet import RetinaNet
from evaluation.test_retinanet import test_retinanet
from utils.data_utils import IEC_dataset_preprocessing, normalize

import time

DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)

CORS(app, support_credentials=True)

def get_CSE_ekg(f):
    f.seek(0x200) # 512
    values = np.frombuffer(f.read(), dtype=np.int16)
    signals = np.array([values[0::2], values[1::2]])
    return signals

def get_big_exam_ekg(f):
    """
    Args:
        f: (File Object)
    Return:
        data: (Array) with sized [#channels, signal length],
        (sampling rate)
    """
    f.seek(0xE8)
    data_length = int.from_bytes(f.read(2), byteorder='little', signed=False)

    f.seek(0xE0)
    number_channels_ekg = int.from_bytes(f.read(2), byteorder='little', signed=False)

    f.seek(0xE4)
    number_channels_hs = int.from_bytes(f.read(2), byteorder='little', signed=False) # heart sound
    number_channels = number_channels_ekg + number_channels_hs

    data = [ list() for _ in range(number_channels) ]

    # data start
    f.seek(0x4B8)
    for index_cycle in range(data_length):
        raw = f.read(2 * number_channels)
        if len(raw) < 2 * number_channels:
            break
        for index_channel in range(number_channels):
            data[index_channel].append(int.from_bytes(
            raw[index_channel*2: (index_channel+1)*2],
            byteorder='little', signed=True))

    data = np.array(data)
    return data, [1000.]*number_channels # sampling rates

@app.route('/PQRSTSegmentation', methods=['POST'])
def testing_using_retinanet():
    """
    Args:
        signal: (tensor) with sized [1, 1, data_length]
    """
    signal = request.json["raw"]
    signal = np.asarray(signal).astype(float)
    signal = signal.reshape((signal.shape[0], signal.shape[1], 1))
    signal = IEC_dataset_preprocessing(signal, smooth=False, dns=False).cuda()
    signal = normalize(signal)
    net = RetinaNet(3).cuda()
    #net.load_state_dict(torch.load("weights/retinanet_best_IEC.pkl"))
    net.load_state_dict(torch.load("weights/retinanet_best_pass_all(CAL)_22.pkl"))
    final_preds = []
    for i in range(signal.size(0) // 128 + 1):
        _, _, pred_signals = test_retinanet(net, signal[i*128:(i+1)*128, :, :], 4992, visual=False)
        final_preds.append(pred_signals)
    final_preds = torch.cat(final_preds, dim=0)

    return jsonify({'raw': signal.tolist(), "label": final_preds.tolist()})

@app.route('/UploadFile', methods=['POST'])
def read_input_file():
    f = request.files.get('raw')
    try:
        # file type bin
        signals, sampling_rate = get_big_exam_ekg(f)
        signal = signals[1][::2].reshape(1, -1)
    except:
        # file type raw
        signals = get_CSE_ekg(f)
        signal = signals[0].reshape((1, -1))

    payload = {"raw": signal.tolist()}
    
    res = requests.post('http://gpu4.mip.nctu.me:8888/PQRSTSegmentation', json=payload)

    return res.json()

def run():
    app.run(host='0.0.0.0', port=8888)