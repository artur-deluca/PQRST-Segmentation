import argparse

import train.train_retinanet as train_retinanet
import train.train_unet as train_unet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="train with specific model(default: retinanet)", type=str, choices=["retinanet", "unet"])
    args = parser.parse_args()

    if args.model == "retinanet":
        train_retinanet.train()
    elif args.model == "unet":
        train_unet.train()
    else:
        # default using retinanet to train
        train_retinanet.train() 