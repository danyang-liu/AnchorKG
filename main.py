import argparse
from data_loader.data_loaders import *
from train_test import *
from utils.parse_config import ConfigParser


def main(config):
    data = load_data(config)
    train(data, config)
    test(data, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnchorKG')

    parser.add_argument('-c', '--config', default="./config/anchorkg_config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('--use_nni', action='store_true', help='use nni to tune hyperparameters')

    config = ConfigParser.from_args(parser)
    seed_everything(config['seed'])
    main(config)