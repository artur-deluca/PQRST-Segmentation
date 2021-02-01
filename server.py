import argparse
from web import api

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='API server for PQRST segmentation.')
    parser.add_argument('--model_path', '-p', type=str, nargs='?', help='The path of Retinanet models.')
    args = parser.parse_args()
    api.run(args.model_path)
    #"weights/retinanet_best_IEC.pkl"