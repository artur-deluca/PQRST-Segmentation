import argparse
import torch
import test.test_retinanet as test_retinanet
import test.test_unet as test_unet
from model.RetinaNet import RetinaNet
from model.UNet import UNet
import wandb

if __name__ == "__main__":
    wandb.init(project="PQRST-segmentation")
    num_classes = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="test with specific model(default: retinanet)", type=str, choices=["retinanet", "unet"])
    args = parser.parse_args()

    if args.model == "retinanet":
        net = RetinaNet(num_classes).cuda()
        net.load_state_dict(torch.load("weights/retinanet_best.pkl"))
        result = test_retinanet.test_retinanet_using_IEC(net)
    elif args.model == "unet":
        net = UNet(1, 4).cuda()
        net.load_state_dict(torch.load("weights/unet_best.pkl"))
        result = test_unet.test_unet_using_IEC(net)
    else:
        net = RetinaNet(num_classes).cuda()
        net.load_state_dict(torch.load("weights/retinanet_best.pkl"))
        result = test_retinanet.test_retinanet_using_IEC(net)
