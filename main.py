import argparse
from data_loader.data_loaders import *
from train_test import *
from parse_config import ConfigParser
import pickle


def main(config):
    # data = load_data(config)
    # with open("/data/RAGNRec_data/data/data2.pkl", 'wb') as f:
    #     pickle.dump(data, f)

    with open("/data/RAGNRec_data/data/data.pkl", 'rb') as f:
        data = pickle.load(f)

    train_test(data, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnchorKG')

    parser.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(parser)
    main(config)