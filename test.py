import numpy as np
import torch
import viz
import matplotlib.pyplot as plt
import pickle
import wandb
wandb.init(project="PQRST-segmentation")

from model import UNet
from data_generator import dataset_preprocessing
from audicor_reader.reader import read_IEC

def test(net, x, ground_truth=None):
    net.eval()
    # input size should be (1, 1, 500 * seconds)
    output = net(x)
    # output size should be (1, 4, 500 * seconds)
    if ground_truth is not None:
        plot = viz.predict_plotter(x[0][0], output[0], ground_truth[0])
    else:
        plot = viz.predict_plotter(x[0][0], output[0])

    return plot

if __name__ == "__main__":
    ekg_filename = '/home/Wr1t3R/PQRST/unet/data/IEC/IEC_from_audicor/CSE001.raw'
    ekg_sig = read_IEC(ekg_filename)
    ekg_sig = np.reshape(ekg_sig[0], (1,len(ekg_sig[0]), 1))
    ekg_sig = dataset_preprocessing(ekg_sig, False)
    print(ekg_sig.shape)
    ekg_sig = ekg_sig.to('cuda')
    net = UNet(in_ch=1, out_ch=4)
    net.to('cuda')
    net.load_state_dict(torch.load("net_params.pkl"))
    plot = test(net, ekg_sig)
    wandb.log({'predict': plot})
