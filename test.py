import argparse
import torch
import evaluation.test_retinanet as test_retinanet
import evaluation.test_unet as test_unet
from model.RetinaNet import RetinaNet
from model.UNet import UNet
import wandb

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

configs = {
    "test_denoise": True,
}

if __name__ == "__main__":
    wandb.init(project="PQRST-segmentation")
    wandb.config.setdefaults(configs)
    num_classes = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="test with specific model(default: retinanet)", type=str, choices=["retinanet", "unet"])
    parser.add_argument("-v", "--visual", help="test with specific model and visualize the result.", action="store_true")
    parser.add_argument("-p", "--path", help="test with specific model weight file from path", type=str, default=config["RetinaNet"]["weight_load_path"])
    args = parser.parse_args()

    if args.model == "retinanet":
        net = RetinaNet(num_classes).cuda()
        net.load_state_dict(torch.load(args.path))
        result = test_retinanet.test_retinanet_using_IEC(net, args.visual)
        result2 = test_retinanet.test_retinanet_using_ANE_CAL(net, args.visual)
    elif args.model == "unet":
        net = UNet(1, 4).cuda()
        net.load_state_dict(torch.load(args.path))
        result = test_unet.test_unet_using_IEC(net)
    else:
        # default using retinanet to test
        net = RetinaNet(num_classes).cuda()
        net.load_state_dict(torch.load(args.path))
        #result = test_retinanet.test_retinanet_using_IEC(net, args.visual)
        #result2 = test_retinanet.test_retinanet_using_ANE_CAL(net, args.visual)
        test_retinanet.test_retinanet_by_qrs(net)