import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def build_train(config):
    print('constructing train ...')
    train_data = {}
    item1 = []
    item2 = []
    label = []
    fp_train = open(config['data']['train_file'], 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        item1.append(linesplit[1])
        item2.append(linesplit[2])
        label.append(float(linesplit[0]))

    train_data['item1'] = item1
    train_data['item2'] = item2
    train_data['label'] = label
    return train_data

def build_val(config):
    print('constructing val ...')
    test_data = {}
    item1 = []
    item2 = []
    label = []
    fp_train = open(config['data']['val_file'], 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        item1.append(linesplit[1])
        item2.append(linesplit[2])
        label.append(float(linesplit[0]))
    test_data['item1'] = item1
    test_data['item2'] = item2
    test_data['label'] = label
    return test_data

def build_test(config):
    print('constructing test ...')
    test_data = {}
    fp_train = open(config['data']['test_file'], 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        if linesplit[0] == "1":
            if linesplit[1] not in test_data:
                test_data[linesplit[1]] = []
            if linesplit[2] not in test_data:
                test_data[linesplit[2]] = []
            test_data[linesplit[1]].append(linesplit[2])
            test_data[linesplit[2]].append(linesplit[1])
    return test_data

def build_doc_feature_embedding(config):
    print('constructing doc feature embedding ...')
    fp_doc_feature = open(config['data']['doc_feature_embedding_file'], 'r', encoding='utf-8')
    doc_embedding_feature_dict = {}
    for line in fp_doc_feature:
        linesplit = line.split('\n')[0].split('\t')
        doc_embedding_feature_dict[linesplit[0]] = np.array(linesplit[1].split(' ')).astype(np.float)
    return doc_embedding_feature_dict

def build_network(config):
    print('constructing adjacency matrix ...')
    entity_id_dict = {}
    fp_entity2id = open(config['data']['entity2id_file'], 'r', encoding='utf-8')
    for line in fp_entity2id:
        linesplit = line.split('\n')[0].split('\t')
        entity_id_dict[linesplit[0]] = int(linesplit[1])+1# 0 for padding

    relation_id_dict = {}
    fp_relation2id = open(config['data']['relation2id_file'], 'r', encoding='utf-8')
    for line in fp_relation2id:
        linesplit = line.split('\n')[0].split('\t')
        relation_id_dict[linesplit[0]] = int(linesplit[1])+1# 0 for padding

    network = nx.MultiGraph()
    print('constructing kg env ...')

    # add news entity to kg
    fp_news_entities = open(config['data']['doc_feature_entity_file'], 'r', encoding='utf-8')
    for line in fp_news_entities:
        linesplit = line.strip().split('\t')
        newsid = linesplit[0]
        news_entities = linesplit[1:]
        for entity in news_entities:
            if entity in entity_id_dict:
                network.add_edge(newsid, entity_id_dict[entity], label="innews")
                network.add_edge(entity_id_dict[entity], newsid, label="innews")

    adj_file_fp = open(config['data']['kg_file'], 'r', encoding='utf-8')
    adj = {}
    for line in adj_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        if entity_id_dict[linesplit[0]] not in adj:
            adj[entity_id_dict[linesplit[0]]] = []

        if len(linesplit) <=20:
            for i in range(1, len(linesplit)):
                entity, relation = linesplit[i].split('#')
                adj[entity_id_dict[linesplit[0]]].append(
                        (int(entity_id_dict[entity]), int(relation_id_dict[relation])))
                network.add_edge(entity_id_dict[linesplit[0]], int(entity_id_dict[entity]),
                                 label=str(relation_id_dict[relation]))
                network.add_edge(int(entity_id_dict[entity]), entity_id_dict[linesplit[0]],
                                 label=str(relation_id_dict[relation]))
            for i in range(len(linesplit),21):
                adj[entity_id_dict[linesplit[0]]].append(
                    (0, 0))
        else:
            for i in range(1, 21):
                entity, relation = linesplit[i].split('#')
                adj[entity_id_dict[linesplit[0]]].append(
                        (int(entity_id_dict[entity]), int(relation_id_dict[relation])))
                network.add_edge(entity_id_dict[linesplit[0]], int(entity_id_dict[entity]),
                                 label=str(relation_id_dict[relation]))
                network.add_edge(int(entity_id_dict[entity]), entity_id_dict[linesplit[0]],
                                 label=str(relation_id_dict[relation]))
    adj_entity = {}
    adj_entity[0] = list(map(lambda x:int(x), np.zeros(20)))
    for item in adj:
        adj_entity[item] = list(map(lambda x:x[0], adj[item]))
    adj_relation = {}
    adj_relation[0] = list(map(lambda x:int(x), np.zeros(20)))
    for item in adj:
        adj_relation[item] = list(map(lambda x: x[1], adj[item]))
    return adj_entity, adj_relation, entity_id_dict, network

def build_entity_relation_embedding(config):
    print('constructing embedding ...')
    entity_embedding = []
    relation_embedding = []
    fp_entity_embedding = open(config['data']['entity_embedding_file'], 'r', encoding='utf-8')
    for line in fp_entity_embedding:
        entity_embedding.append(np.array(line.strip().split('\t')).astype(np.float))
    fp_relation_embedding = open(config['data']['relation_embedding_file'], 'r', encoding='utf-8')
    for line in fp_relation_embedding:
        relation_embedding.append(np.array(line.strip().split('\t')).astype(np.float))
    return torch.FloatTensor(entity_embedding), torch.FloatTensor(relation_embedding)

def load_news_entity(config, news_entities_path):
    entityid2index = {}
    fp_entity2id = open(config['data']['entity2id_file'], 'r', encoding='utf-8')
    for line in fp_entity2id:
        entityid, entityindex = line.strip().split('\t')
        entityid2index[entityid] = int(entityindex)+1 #0 for padding

    doc_entities = {}
    entity_doc = {}
    fp_news_entities = open(news_entities_path, 'r', encoding='utf-8')
    for line in fp_news_entities:
        linesplit = line.strip().split('\t')
        newsid = linesplit[0]
        news_entities = linesplit[1:]
        doc_entities[newsid] = []
        for entity in news_entities:
            if entity in entityid2index:
                doc_entities[newsid].append(entityid2index[entity])
                if entityid2index[entity] not in entity_doc:
                    entity_doc[entityid2index[entity]] = []
                entity_doc[entityid2index[entity]].append(newsid)
        if len(doc_entities[newsid]) > config['model']['news_entity_num']:
            doc_entities[newsid] = doc_entities[newsid][:config['model']['news_entity_num']]
        for i in range(config['model']['news_entity_num']-len(doc_entities[newsid])):#todo
            doc_entities[newsid].append(0)
    for item in entity_doc:
        if len(entity_doc[item])>config['model']['news_entity_num']:
            entity_doc[item] = entity_doc[item][:config['model']['news_entity_num']] #todo load entity in titles

    return doc_entities, entity_doc

def load_doc_feature(config):
    doc_embeddings = {}
    fp_news_feature = open(config['data']['doc_feature_entity_file'], 'r', encoding='utf-8')
    for line in fp_news_feature:
        newsid, news_embedding = line.strip().split('\t')
        news_embedding = news_embedding.split(',')
        doc_embeddings[newsid] = news_embedding
    return doc_embeddings

def get_anchor_graph_data(config):
    print('constructing anchor doc ...')
    test_data = {}
    item1 = []
    item2 = []
    label = []
    fp_news_entities = open(config['data']['doc_feature_entity_file'], 'r', encoding='utf-8')
    for line in fp_news_entities:
        linesplit = line.strip().split('\t')
        newsid = linesplit[0]
        item1.append(newsid)
        item2.append(newsid)
        label.append(0.0)

    test_data['item1'] = item1
    test_data['item2'] = item2
    test_data['label'] = label
    return test_data

def build_hit_dict(config):
    print('constructing hit dict ...')
    hit_dict = {}
    fp_train = open(config['data']['train_file'], 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        if linesplit[0] == '1':
            if linesplit[1] not in hit_dict:
                hit_dict[linesplit[1]] = set()
            if linesplit[2] not in hit_dict:
                hit_dict[linesplit[2]] = set()
            hit_dict[linesplit[1]].add(linesplit[2])
            hit_dict[linesplit[2]].add(linesplit[1])
    return hit_dict

def load_warm_up(config):
    print('build warm up data ...')
    warm_up_data = {}
    fp_warmup = open(config['data']['warm_up_train_file'], 'r', encoding='utf-8')
    item1 = []
    item2 = []
    label = []
    for line in fp_warmup:
        linesplit = line.strip().split('\t')
        newsid = linesplit[0]
        entityid = linesplit[1]
        item1.append(newsid)
        item2.append(entityid)
        label.append(float(linesplit[2]))
    warm_up_data['item1'] = item1
    warm_up_data['item2'] = item2
    warm_up_data['label'] = label
    return warm_up_data

def build_neibor_embedding(config, entity_doc_dict, doc_feature_embedding):
    print('build neiborhood embedding ...')
    entity_id_dict = {}
    fp_entity2id = open(config['data']['entity2id_file'], 'r', encoding='utf-8')
    for line in fp_entity2id:
        linesplit = line.split('\n')[0].split('\t')
        entity_id_dict[linesplit[0]] = int(linesplit[1]) + 1  # int
    entity_num = len(entity_id_dict)
    entity_neibor_embedding_list = []
    entity_neibor_num_list = []
    for i in range(entity_num+1):
        entity_neibor_embedding_list.append(np.zeros(768))
        entity_neibor_num_list.append(1)
    for entity in entity_doc_dict:
        entity_news_embedding_list = []
        for news in entity_doc_dict[entity]:
            entity_news_embedding_list.append(doc_feature_embedding[news])
        entity_neibor_embedding_list[entity] = np.sum(entity_news_embedding_list, axis=0)
        if len(entity_doc_dict[entity])>=2:
            entity_neibor_num_list[entity] = len(entity_doc_dict[entity])-1
    return torch.tensor(entity_neibor_embedding_list).cuda(), torch.tensor(entity_neibor_num_list).cuda()
