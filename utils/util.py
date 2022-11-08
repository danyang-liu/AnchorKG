import json
import torch
import numpy as np
from pathlib import Path
import networkx as nx
from itertools import repeat
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import requests
import math
import random
import zipfile
import os
from tqdm import tqdm

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

def seed_everything(seed=2022):
    """Set seed for reproducibility.

    Args:
        seed (int): Seed number.
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

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

def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath

def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)

def get_mind_data_set(type):
    """ Get MIND dataset address

    Args:
        type (str): type of mind dataset, must be in ['large', 'small', 'demo']

    Returns:
        list: data url and train valid dataset name
    """
    assert type in ["large", "small", "demo"]

    if type == "large":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDlarge_train.zip",
            "MINDlarge_dev.zip",
            "MINDlarge_utils.zip",
        )

    elif type == "small":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDsmall_train.zip",
            "MINDsmall_dev.zip",
            "MINDsma_utils.zip",
        )

    elif type == "demo":
        return (
            "https://recodatasets.blob.core.windows.net/newsrec/",
            "MINDdemo_train.zip",
            "MINDdemo_dev.zip",
            "MINDdemo_utils.zip",
        )

def download_deeprec_resources(azure_container_url, data_path, remote_resource_name):
    """Download resources.

    Args:
        azure_container_url (str): URL of Azure container.
        data_path (str): Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))

def build_train(config):
    print('constructing train ...')
    train_data = {}
    item1 = []
    item2 = []
    label = []
    fp_train = open(config['datapath']+config['train_file'], 'r', encoding='utf-8')
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
    fp_train = open(config['datapath']+config['val_file'], 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        item1.append(linesplit[1])
        item2.append(linesplit[2])
        label.append(float(linesplit[0]))
    test_data['item1'] = item1
    test_data['item2'] = item2
    test_data['label'] = label
    return test_data

def build_test(config):#test集里为每个item创建其正例list
    print('constructing test ...')
    test_data = {}
    fp_train = open(config['datapath']+config['test_file'], 'r', encoding='utf-8')
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
    fp_doc_feature = open(config['datapath']+config['doc_feature_embedding_file'], 'r', encoding='utf-8')
    doc_embedding_feature_dict = {}
    for line in fp_doc_feature:
        linesplit = line.split('\n')[0].split('\t')
        doc_embedding_feature_dict[linesplit[0]] = torch.tensor(np.array(linesplit[1].split(' ')).astype(np.float), dtype=torch.float32)
    return doc_embedding_feature_dict

def build_network(config):
    print('constructing adjacency matrix ...')
    entity_id_dict = {}
    fp_entity2id = open(config['datapath']+config['entity2id_file'], 'r', encoding='utf-8')
    _ = fp_entity2id.readline()
    for line in fp_entity2id:
        linesplit = line.split('\n')[0].split('\t')
        entity_id_dict[linesplit[0]] = int(linesplit[1])+1# 0 for padding

    relation_id_dict = {}
    fp_relation2id = open(config['datapath']+config['relation2id_file'], 'r', encoding='utf-8')
    _ = fp_relation2id.readline()
    for line in fp_relation2id:
        linesplit = line.split('\n')[0].split('\t')
        relation_id_dict[linesplit[0]] = int(linesplit[1])+1# 0 for padding

    network = nx.MultiGraph()
    print('constructing kg env ...')

    # add news entity to kg
    fp_news_entities = open(config['datapath']+config['doc_feature_entity_file'], 'r', encoding='utf-8')
    for line in fp_news_entities:
        try:
            linesplit = line.strip().split('\t')
            newsid = linesplit[0]
            news_entities = linesplit[1].split(" ")
            for entity in news_entities:
                if entity in entity_id_dict:
                    network.add_edge(newsid, entity_id_dict[entity], label="innews")
                    network.add_edge(entity_id_dict[entity], newsid, label="innews")
        except:
            pass

    adj_file_fp = open(config['datapath']+config['kg_file'], 'r', encoding='utf-8')
    adj = {}
    for line in adj_file_fp:
        head, relation, tail = line.split('\n')[0].split('\t')#(head, relation, tail)
        if entity_id_dict[head] not in adj:
            adj[entity_id_dict[head]] = []
        elif len(adj[entity_id_dict[head]])>=20:
            continue

        adj[entity_id_dict[head]].append((entity_id_dict[tail], relation_id_dict[relation]))
        network.add_edge(entity_id_dict[head], entity_id_dict[tail], label=str(relation_id_dict[relation]))
        network.add_edge(entity_id_dict[tail], entity_id_dict[head], label=str(relation_id_dict[relation]))#为何强行加成双向

    for item in entity_id_dict.values():
        if item not in adj:
            adj[item] = [(0, 0)]*20
        else:
            for _ in range(len(adj[item]), 20):
                adj[item].append((0, 0))

    adj_entity = {}
    adj_entity[0] = torch.zeros(20, dtype=torch.long)
    for item in adj:
        adj_entity[item] = torch.tensor(list(map(lambda x:x[0], adj[item])), dtype=torch.long)
    adj_relation = {}
    adj_relation[0] = torch.zeros(20, dtype=torch.long)
    for item in adj:
        adj_relation[item] = torch.tensor(list(map(lambda x: x[1], adj[item])), dtype=torch.long)
    return adj_entity, adj_relation, entity_id_dict, relation_id_dict, network

def build_entity_relation_embedding(config, entity_num, relation_num):
    print('constructing embedding ...')
    entity_embedding = np.zeros((entity_num+1, 100))
    relation_embedding = np.zeros((relation_num+1, 100))
    fp_entity_embedding = open(config['datapath']+config['entity_embedding_file'], 'r', encoding='utf-8')
    for i, line in enumerate(fp_entity_embedding):
        entity_embedding[i+1] = np.array(line.strip().split('\t')).astype(np.float)
    fp_relation_embedding = open(config['datapath']+config['relation_embedding_file'], 'r', encoding='utf-8')
    for i, line in enumerate(fp_relation_embedding):
        relation_embedding[i+1] = np.array(line.strip().split('\t')).astype(np.float)
    return torch.FloatTensor(entity_embedding), torch.FloatTensor(relation_embedding)

def load_news_entity(config, entity_id_dict):
    doc2entities = {}
    entity2doc = {}
    fp_news_entities = open(config['datapath']+config['doc_feature_entity_file'], 'r', encoding='utf-8')
    for line in fp_news_entities:
        linesplit = line.strip().split('\t')
        newsid = linesplit[0]
        if len(linesplit)>1:
            news_entities = linesplit[1].split(" ")
        else:
            news_entities=[]
        doc2entities[newsid] = []
        for entity in news_entities:
            if entity in entity_id_dict:
                doc2entities[newsid].append(entity_id_dict[entity])
                if entity_id_dict[entity] not in entity2doc:
                    entity2doc[entity_id_dict[entity]] = []
                entity2doc[entity_id_dict[entity]].append(newsid)
        if len(doc2entities[newsid]) > config['news_entity_num']:
            doc2entities[newsid] = doc2entities[newsid][:config['news_entity_num']]
        else:
            for i in range(config['news_entity_num']-len(doc2entities[newsid])):#todo
                doc2entities[newsid].append(0)
    for item in doc2entities:
        doc2entities[item] = torch.tensor(doc2entities[item], dtype=torch.long)
    # for item in entity2doc:#对doc数目也要限制？
    #     if len(entity2doc[item])>config['news_entity_num']:
    #         entity2doc[item] = entity2doc[item][:config['news_entity_num']] #todo load entity in titles

    return doc2entities, entity2doc

def load_doc_feature(config):
    doc_embeddings = {}
    fp_news_feature = open(config['datapath']+config['doc_feature_entity_file'], 'r', encoding='utf-8')
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
    fp_news_entities = open(config['datapath']+config['doc_feature_entity_file'], 'r', encoding='utf-8')
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
    fp_train = open(config['datapath']+config['train_file'], 'r', encoding='utf-8')
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

def build_neibor_embedding(config, entity_doc_dict, doc_feature_embedding, entity_id_dict):#返回的是每个entity连接的doc的embedding的和和doc数目-1， 用于计算coherence reward
    print('build neiborhood embedding ...')
    entity_num = len(entity_id_dict)
    #每个entity（包括0）的邻居embedding及邻居个数
    entity_neibor_embedding_list = torch.zeros([entity_num+1, 768],dtype=torch.float32)
    entity_neibor_num_list = torch.ones(entity_num+1, dtype=torch.long)
    for entity in entity_doc_dict:
        entity_news_embedding_list = torch.zeros([len(entity_doc_dict[entity]), 768], dtype=torch.float32)
        for i, news in enumerate(entity_doc_dict[entity]):
            entity_news_embedding_list[i] = doc_feature_embedding[news]
        entity_neibor_embedding_list[entity] = torch.sum(entity_news_embedding_list, dim=0)
        if len(entity_doc_dict[entity])>=2:
            entity_neibor_num_list[entity] = len(entity_doc_dict[entity])-1#为何要-1,排除掉forward里的news本身
    return entity_neibor_embedding_list, entity_neibor_num_list#todo torch.tensor(entity_neibor_embedding_list).cuda(), torch.tensor(entity_neibor_num_list).cuda()

def build_item2item_dataset(config):
    print("constructing item2item dataset ...")
    fp_train = open(config['datapath']+config['train_behavior'], 'r', encoding='utf-8')
    user_history_dict = {}#每个用户历史点击的新闻id，包括history和behavior
    news_click_dict = {}#每个新闻被所有用户点击的总次数
    doc_doc_dict = {}#两个文档对的共同点击用户数,(doc1,doc2)和(doc2,doc1)是一样的
    all_news_set = set()#behavior里所有新闻id
    for line in fp_train:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        if userid not in user_history_dict:
            user_history_dict[userid] = set()
        for news in behavior:
            newsid, news_label = news.split('-')
            if newsid=='':
                continue
            all_news_set.add(newsid)
            if news_label == "1":
                user_history_dict[userid].add(newsid)
                if newsid not in news_click_dict:
                    news_click_dict[newsid] = 1
                else:
                    news_click_dict[newsid] = news_click_dict[newsid] + 1
        news = history.split(' ')#history可能缺失
        for newsid in news:
            if newsid=='':
                continue
            user_history_dict[userid].add(newsid)
            if newsid not in news_click_dict:
                news_click_dict[newsid] = 1
            else:
                news_click_dict[newsid] = news_click_dict[newsid] + 1
    for user in user_history_dict:
        list_user_his = list(user_history_dict[user])
        for i in range(len(list_user_his) - 1):
            for j in range(i + 1, len(list_user_his)):
                doc1 = list_user_his[i]
                doc2 = list_user_his[j]
                if doc1 != doc2:
                    if (doc1, doc2) not in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = 1
                    elif (doc1, doc2) in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = doc_doc_dict[(doc1, doc2)] + 1
                    elif (doc2, doc1) in doc_doc_dict and (doc1, doc2) not in doc_doc_dict:
                        doc_doc_dict[(doc2, doc1)] = doc_doc_dict[(doc2, doc1)] + 1
    weight_doc_doc_dict = {}#保留为postive instance的条件
    for item in doc_doc_dict:
        if item[0] in news_click_dict and item[1] in news_click_dict:
            weight_doc_doc_dict[item] = doc_doc_dict[item] / math.sqrt(
                news_click_dict[item[0]] * news_click_dict[item[1]])

    THRED_CLICK_TIME = 10
    freq_news_set = set()
    for news in news_click_dict:
        if news_click_dict[news] > THRED_CLICK_TIME:
            freq_news_set.add(news)
    news_pair_thred_w_dict = {}  # {(new1, news2): click_weight}
    for item in weight_doc_doc_dict:
        if item[0] in freq_news_set and item[1] in freq_news_set:
            news_pair_thred_w_dict[item] = weight_doc_doc_dict[item]

    news_positive_pairs = []
    for item in news_pair_thred_w_dict:
        if news_pair_thred_w_dict[item] > 0.05:
            news_positive_pairs.append(item)

    fp_train_data = open(config['datapath'] + config['train_file'], 'w', encoding='utf-8')
    fp_valid_data = open(config['datapath'] + config['val_file'], 'w', encoding='utf-8')
    fp_test_data = open(config['datapath'] + config['test_file'], 'w', encoding='utf-8')
    for item in news_positive_pairs:#这里有个问题，即负例可能是正例
        random_num = random.random()
        if random_num < 0.8:
            fp_train_data.write("1" + '\t' + item[0] + '\t' + item[1] + '\n')
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                fp_train_data.write("0" + '\t' + item[0] + '\t' + negative + '\n')
        elif random_num < 0.9:
            fp_valid_data.write("1" + '\t' + item[0] + '\t' + item[1] + '\n')
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                fp_valid_data.write("0" + '\t' + item[0] + '\t' + negative + '\n')
        else:
            fp_test_data.write("1" + '\t' + item[0] + '\t' + item[1] + '\n')
            negative_list = random.sample(list(freq_news_set), 99)
            for negative in negative_list:
                fp_test_data.write("0" + '\t' + item[0] + '\t' + negative + '\n')
    fp_train_data.close()
    fp_valid_data.close()
    fp_test_data.close()


def build_doc_feature(config):#包含每个news的doc embedding，以及其包含的entity id
    print("constructing news features ... ")

    news_features = {}

    news_feature_dict = {}
    fp_train_news = open(config['datapath'] + config['train_news'], 'r', encoding='utf-8')
    for line in fp_train_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        news_feature_dict[newsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)
    # entityid, entity_freq, entity_position, entity_type
    fp_dev_news = open(config['datapath'] + config['valid_news'], 'r', encoding='utf-8')
    for line in fp_dev_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        news_feature_dict[newsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)

    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    for news in news_feature_dict:
        sentence_embedding = model.encode(news_feature_dict[news][0])
        title_entity_json = json.loads(news_feature_dict[news][1])
        abstract_entity_json = json.loads(news_feature_dict[news][2])
        news_entity_feature = set()
        for item in title_entity_json:
            news_entity_feature.add(item['WikidataId'])
        for item in abstract_entity_json:
            news_entity_feature.add(item['WikidataId'])
        news_features[news] = (sentence_embedding, list(news_entity_feature))

    fp_doc_feature_entity = open(config['datapath'] + config['doc_feature_entity_file'], 'w', encoding='utf-8')
    fp_doc_feature_embedding = open(config['datapath'] + config['doc_feature_embedding_file'], 'w', encoding='utf-8')
    for news in news_features:
        fp_doc_feature_entity.write(news+'\t')
        fp_doc_feature_entity.write(' '.join(news_features[news][1])+'\n')
        fp_doc_feature_embedding.write(news + '\t')
        fp_doc_feature_embedding.write(' '.join(list(map(lambda x:str(x),news_features[news][0]))) + '\n')

    fp_doc_feature_entity.close()
    fp_doc_feature_embedding.close()

def process_mind_data(config):
    build_item2item_dataset(config)
    build_doc_feature(config)