import argparse
from data_loader.data_loaders import *
from model.AnchorKG import AnchorKG
from model.Recommender import Recommender
from model.Reasoner import Reasoner
from trainer.trainer import Trainer
from utils.parse_config import ConfigParser


def main(config):
    seed_everything(config['seed'])
    device, _ = prepare_device(config['n_gpu'])
    data = load_data(config)
    _, _, _, _, _, doc_feature_embedding, entity_adj, relation_adj, entity_id_dict, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, hit_dict = data
    
    model_anchor = AnchorKG(config, doc_entity_dict, entity_doc_dict, doc_feature_embedding, entity_adj, relation_adj, hit_dict, entity_id_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, device)
    model_recommender = Recommender(config, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, device)
    model_reasoner = Reasoner(config, doc_feature_embedding, entity_embedding, relation_embedding, device)

    trainer = Trainer(config, model_anchor, model_recommender, model_reasoner, device, data)
    trainer.logger.info("config {}".format(config.config))

    if config['use_nni'] and config['warm_up']:#can not tune warm_up and main code together
        trainer.warm_up()
    elif config['use_nni'] and not config['warm_up']:
        trainer.train()
    else:
        trainer.train()
        trainer.test()


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
    main(config)