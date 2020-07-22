import torch
from scripts.test import test, compare_to_standard
from model.RetinaNet import RetinaNet

if __name__ == "__main__":
    net = RetinaNet(3).cuda()
    net.load_state_dict(torch.load("./weights/retinanet_best_pass_all(CAL)_22.pkl"))

    intervals, qrs_intervals = test(net, "./data/IEC/IEC_from_audicor/CAL20002_1.raw")
    
    standard_intervals = [{"p_duration": {'mean': 76}, "pq_interval": {'mean': 128}, "qrs_duration": {'mean': 100}, "qt_interval": {'mean': 328}}]
    standard_qrs_intervals = [{'q_duration': 0, 'r_duration': 50, 's_duration': 50}]
    print(compare_to_standard(intervals, qrs_intervals, standard_intervals, standard_qrs_intervals))