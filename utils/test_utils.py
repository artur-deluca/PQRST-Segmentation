import torch
import numpy as np
from utils.data_utils import normalize, IEC_dataset_preprocessing

def load_IEC(denoise=True, pre=False):
    """
    Arg:
        pre: (bool) load from saved preprocessed data or not
    """
    # (num of ekg signal, length, 1)
    if pre:
        ekg_sig = torch.load("./data/IEC_preprocessed_data.pt").to('cuda')
    else:
        ekg_sig = []
        for i in range(1, 126):
            ekg_filename = '/home/Wr1t3R/PQRST/unet/data/IEC/IEC_from_audicor/CSE'+ str(i).rjust(3, '0') + '.raw'
            try:
                sig = read_IEC(ekg_filename)
                sig = np.reshape(sig[0], (len(sig[0]), 1))
                ekg_sig.append(sig.astype(float))
            except IOError:
                print("file {} does not exist".format("CSE"+str(i).rjust(3, '0')))
        ekg_sig = dataset_preprocessing(ekg_sig, smooth=False, dns=denoise)
        ekg_sig = ekg_sig.to('cuda')
        ekg_sig = normalize(ekg_sig, instance=True)
        torch.save(ekg_sig, "./data/IEC_preprocessed_data.pt")

    return ekg_sig