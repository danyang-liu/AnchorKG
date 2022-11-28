import sys
import os
sys.path.append('')
import argparse
from utils.util import process_mind_data, get_mind_data_set, download_deeprec_resources, process_KPRN_data, seed_everything
from data_loader.data_loaders import process_data_and_cache
from utils.parse_config import ConfigParser


parser = argparse.ArgumentParser(description='data processing')
parser.add_argument('-c', '--config', default="./config/data_config.json", type=str, help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--use_nni', action='store_true', help='use nni to tune hyperparameters')

config = ConfigParser.from_args(parser)
seed_everything(config['seed'])

# Options: demo, small, large
# The download speed here is a bit slow, you can go to the corresponding url to download the zip and put it in the corresponding directory
MIND_type = config['MIND_type']
data_path = config['datapath']

train_news_file = data_path + config['train_news']#"train_news": "/mind/train/news.tsv",
valid_news_file = data_path + config['valid_news']
knowledge_graph_file = data_path + config['kg_file']

mind_url, mind_train_dataset, mind_dev_dataset, _ = get_mind_data_set(MIND_type)

kg_url = "https://kredkg.blob.core.windows.net/wikidatakg/"

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, train_news_file.rsplit("/", 1)[0], mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, valid_news_file.rsplit("/", 1)[0], mind_dev_dataset)

if not os.path.exists(knowledge_graph_file):
    download_deeprec_resources(kg_url, os.path.join(data_path, 'kg'), "kg.zip")


process_mind_data(config)

process_data_and_cache(config)

process_KPRN_data(config)