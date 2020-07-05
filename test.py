import argparse
import torch
import test.test_retinanet as test_retinanet
import test.test_unet as test_unet
from model.RetinaNet import RetinaNet
from model.UNet import UNet
import wandb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = {
    "test_denoise": True,
}

if __name__ == "__main__":
    wandb.init(project="PQRST-segmentation")
    wandb.config.setdefaults(config)
    num_classes = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="test with specific model(default: retinanet)", type=str, choices=["retinanet", "unet"])
    parser.add_argument("-v", "--visual", help="test with specific model and visualize the result.", action="store_true")
    parser.add_argument("-p", "--path", help="test with specific model weight file from path", type=str, default="weights/retinanet_pass_all_IEC_1.pkl")
    args = parser.parse_args()

    if args.model == "retinanet":
        net = RetinaNet(num_classes).cuda()
        net.load_state_dict(torch.load(args.path))
        #result = test_retinanet.test_retinanet_using_IEC(net, args.visual)
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
        result2 = test_retinanet.test_retinanet_using_ANE_CAL(net, args.visual)