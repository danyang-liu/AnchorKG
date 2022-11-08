import torch
from utils.parse_config import ConfigParser
import argparse
from utils.util import *
import json
import random
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KPRN')

    parser.add_argument('-c', '--config', default="./config/data_config.json", type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--use_nni', action='store_true', help='use nni to tune hyperparameters')

    config = ConfigParser.from_args(parser)
    entity_id_dict = np.load(config['datapath']+"/cache/entity_id_dict.npy", allow_pickle=True).item()
    Train_data = build_train(config)
    doc_entity_dict, entity_doc_dict = load_news_entity(config, entity_id_dict)
    hit_dict = build_hit_dict(config)
    entity_adj = torch.load(config['datapath']+"/cache/entity_adj.pt")
    relation_adj = torch.load(config['datapath']+"/cache/relation_adj.pt")

    news_set=set()
    for item1, item2, label in zip(Train_data['item1'], Train_data['item2'], Train_data['label']):
        news_set.add(item1)

    def find_path(data, news, entities, relations, pre_ents, pre_edges):
        for ent, rel in zip(entities, relations):
            if ent==0:
                break
            other_news = set(entity_doc_dict[ent]) - set([news]) if ent in entity_doc_dict else set()
            for other in other_news:
                if (news, other) in data:
                    data[(news, other)]["paths"].append(pre_ents+[ent])
                    data[(news, other)]["edges"].append(pre_edges+[rel])
                elif other in hit_dict[news]:
                    data[(news, other)] = { "label": 1, "item1": news, "item2": other, "paths": [pre_ents+[ent]], "edges": [pre_edges+[rel]]}
                else:
                    data[(news, other)] = { "label": 0, "item1": news, "item2": other, "paths": [pre_ents+[ent]], "edges": [pre_edges+[rel]]}
    
    count_1=0
    count_0=0
    with open(config['datapath']+config['KPRN_train_file'], "w") as f_train:
        with open(config['datapath']+config['KPRN_val_file'], "w") as f_dev:
            for item1 in tqdm(news_set, total=len(news_set)):
                data = {}
                news_entity = doc_entity_dict[item1].tolist()
                find_path(data, item1, news_entity, [0]*20, [], [])
                for ent in news_entity:
                    hop1_entity = entity_adj[ent].tolist()
                    hop1_relation = relation_adj[ent].tolist()
                    find_path(data, item1, hop1_entity, hop1_relation, [ent], [0])
                    for ent_hop1, rel_hop1 in zip(hop1_entity, hop1_relation):
                        hop2_entity = entity_adj[ent_hop1].tolist()
                        hop2_relation = relation_adj[ent_hop1].tolist()
                        find_path(data, item1, hop2_entity, hop2_relation, [ent, ent_hop1], [0, rel_hop1])

                for item in data:
                    if data[item]["label"]!=1 and random.random()>=0.001:#过滤负例
                        continue
                    if len(data[item]['paths']) > 20:#每个(news1, news2)间最多取20条路径
                        indices = random.sample(range(len(data[item]['paths'])), 20)
                        data[item]['paths'] = [data[item]['paths'][idx] for idx in indices]
                        data[item]['edges'] = [data[item]['edges'][idx] for idx in indices]
                    line = json.dumps(data[item])+"\n"
                    if data[item]["label"]==1:
                        count_1+=1
                    else:
                        count_0+=1
                    if random.random()<0.2:
                        f_dev.write(line)
                    else:
                        f_train.write(line)
    print(count_1, count_0)

#15619 29992338
#15629 30123