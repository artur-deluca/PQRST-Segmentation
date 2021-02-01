import torch
import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import zipfile
import io

from model.RetinaNet import RetinaNet
from evaluation.test_retinanet import test_retinanet
from utils.data_utils import IEC_dataset_preprocessing, normalize

import time

DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)

CORS(app, support_credentials=True)

retinanet_model_path = None

def outlier_removal(signal, outlier_value=-32768):
    '''Remove the suddent outliers (which have the values of -32768) from the signal inplace.
    args:
        signal: 2d-array of shape (?, signal_length)

    output:
        clean signal
    '''
    for i in range(signal.shape[0]):
        to_be_removed, x = (signal[i] == outlier_value), lambda z: z.nonzero()[0]
        signal[i, to_be_removed]= np.interp(x(to_be_removed), x(~to_be_removed), signal[i, ~to_be_removed])
    
    return signal

def get_CSE_ekg(f):
    """
    get CSE data from CSE raw file.

    Args:
        f: (File Object)
    Return:
        signals: (Array) with sized[2, signal_length]
    """
    f.seek(0x200) # 512
    values = np.frombuffer(f.read(), dtype=np.int16)
    signals = np.array([values[0::2], values[1::2]])
    return signals

def get_big_exam_ekg(f):
    """
    get big exam data from big exam raw file.

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

@app.route('/submit_PQRST', methods=['POST'])
def testing_using_retinanet():
    """
    test RetinaNet and return the segmentation result.

    Args:
        signal: (tensor) with sized [batch_size, data_length]
    Returns:
        (json): result prediction with preprocessed input and final predict segmentation.
    """
    global retinanet_model_path
    signal = request.json['ECG']
    #denoise = request.json["denoise"]
    denoise = False
    
    signal = np.asarray(signal).astype(float)
    signal = signal.reshape((signal.shape[0], signal.shape[1], 1))
    target_length = (signal.shape[1] // 64) * 64
    signal = IEC_dataset_preprocessing(signal, smooth=False, dns=denoise, target_length=target_length).cuda()
    signal = normalize(signal)
    net = RetinaNet(3).cuda()
    #net.load_state_dict(torch.load("weights/retinanet_best_IEC.pkl"))
    net.load_state_dict(torch.load(retinanet_model_path))
    batch_size = 128
    final_preds = []
    for i in range(signal.size(0) // batch_size + 1):
        _, _, pred_signals = test_retinanet(net, signal[i*batch_size:(i+1)*batch_size, :, :], target_length, visual=False)
        final_preds.append(pred_signals)
    final_preds = torch.cat(final_preds, dim=0)
    
    return jsonify({'raw': signal.tolist(), "label": final_preds.tolist()})

#@app.route('/UploadFile', methods=['POST'])
def read_input_file():
    """
    API enter point. Check the file type and processed the signal.
    Returns:
        (json) same as function "testing_using_retinanet"
    """
    f = request.files.get('raw')

    labeling = request.form.get('labeling')

    if not labeling:
        try:
            # file type bin
            signals, sampling_rate = get_big_exam_ekg(f)
            signal = signals[1][::2].reshape(1, -1)
        except:
            # file type raw
            signals = get_CSE_ekg(f)
            signal = signals[0].reshape((1, -1))

        payload = {"raw": signal.tolist()}

        res = requests.post('http://gpu4.miplab.org:8899/PQRSTSegmentation', json=payload)

        return res.json()
    
    else:
        try:
            signals, sampling_rate = get_big_exam_ekg(f)
            signal = signals[:, ::2]
        except:
            pass
        
        payload = jsonify({"raw": signal.tolist()})

        return payload

def read_snp_file():
    """
    API enter point. read snp file and return EKG and PCG signal.
    """
    f = request.files.get('raw')

@app.route('/UploadFile', methods=['POST'])
def read_snp(main_sampling_rate=500, channel_sampling_rate=None):
    channel_sampling_rate = [500, 500] if channel_sampling_rate is None else channel_sampling_rate
    
    raw_file = request.files['snp'].read()

    outputs = list()

    with io.BytesIO(raw_file) as f:
        # reading header
        f.read(0x24) # padding
        number_channels = int.from_bytes(f.read(0x1), byteorder='little')
        print(number_channels)

        # calculate reading order
        data_cycle = int(main_sampling_rate // channel_sampling_rate[-1])
        index_order = list()
        number_value_per_cycle = [0] * number_channels
        index_value_per_cycle = [list() for _ in range(number_channels)]
        for cycle in range(data_cycle):
            for index_channel in range(number_channels):
                if cycle % (main_sampling_rate // channel_sampling_rate[index_channel]) == 0:
                    index_order.append(index_channel)

        for index_channel, index_value in zip(index_order, range(len(index_order))):
            number_value_per_cycle[index_channel] += 1
            index_value_per_cycle[index_channel].append(index_value)

        # calculate number of cycle
        f.seek(0, 2) # to the end of file
        file_size = f.tell()
        number_cycles = (file_size - 512) // 2 // len(index_order)
        total_time_in_sec = number_cycles * index_order.count(0) / channel_sampling_rate[0]

        # reading raw file
        f.seek(0x200) # 512
        values = np.frombuffer(f.read(0x2 * number_cycles * len(index_order)), dtype=np.int16)
        channel_signals = [ np.ndarray([number_cycles * number_value_per_cycle[i]]) for i in range(number_channels) ]
        for index_channel in range(number_channels):
            for index_value in range(number_value_per_cycle[index_channel]):
                channel_signals[index_channel][index_value::number_value_per_cycle[index_channel]] = values[index_value_per_cycle[index_channel][index_value]::len(index_order)]
                
        # convert to numpy array
        channel_signals = np.array(channel_signals)
        
        outputs.append(channel_signals)
    #outputs = np.array(outputs)
    outputs = outlier_removal(outputs[0])

    payload = jsonify({'ECG': list(outputs[0]), 'PCG': list(outputs[1])})
    return payload

def run(model_path):
    global retinanet_model_path
    retinanet_model_path = model_path
    app.run(host='0.0.0.0', port=8899)
