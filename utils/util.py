import json
import torch
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import requests
import math
import random
import zipfile
import os
from tqdm import tqdm
import copy

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
            "MINDsmall_utils.zip",
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

def build_item2item_dataset(config):
    print("constructing item2item dataset ...")
    fp_train = open(config['datapath']+config['train_behavior'], 'r', encoding='utf-8')
    fp_valid = open(config['datapath']+config['valid_behavior'], 'r', encoding='utf-8')
    lines = fp_train.readlines()
    lines.extend(fp_valid.readlines())
    fp_train.close()
    fp_valid.close()
    user_history_dict = {}#history clicked news id for each user, including history and behavior
    news_click_dict = {}#The total number of clicks each news was clicked by all users
    doc_doc_dict = {}#The number of co-clicks for a pair of documents, (doc1,doc2)和(doc2,doc1) are the same
    for line in lines:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        if userid not in user_history_dict:
            user_history_dict[userid] = set()
        for news in behavior:
            newsid, news_label = news.split('-')
            if newsid=='':
                continue
            if news_label == "1":
                user_history_dict[userid].add(newsid)
                if newsid not in news_click_dict:
                    news_click_dict[newsid] = 1
                else:
                    news_click_dict[newsid] = news_click_dict[newsid] + 1
        news = history.split(' ')#history may be empty
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
    weight_doc_doc_dict = {}#postive instance principle
    for item in doc_doc_dict:
        if item[0] in news_click_dict and item[1] in news_click_dict:
            weight_doc_doc_dict[item] = doc_doc_dict[item] / math.sqrt(news_click_dict[item[0]] * news_click_dict[item[1]])

    THRED_CLICK_TIME = 10
    freq_news_set = set()
    for news in news_click_dict:
        if news_click_dict[news] > THRED_CLICK_TIME:
            freq_news_set.add(news)
    news_positive_pairs = []#[(new1, news2), ...]
    for item in weight_doc_doc_dict:
        if item[0] in freq_news_set and item[1] in freq_news_set and weight_doc_doc_dict[item]>0.05:
            news_positive_pairs.append(item)

    os.makedirs(config['datapath'] + config['pos_train_file'].rsplit("/", 1)[0], exist_ok=True)
    fp_train_data = open(config['datapath'] + config['pos_train_file'], 'w', encoding='utf-8')
    fp_valid_data = open(config['datapath'] + config['pos_val_file'], 'w', encoding='utf-8')
    fp_test_data = open(config['datapath'] + config['pos_test_file'], 'w', encoding='utf-8')
    for item in news_positive_pairs:#把负例采样放到后面，进行Knowledge-aware的负例采样
        random_num = random.random()
        if random_num < 0.8:
            fp_train_data.write("1" + '\t' + item[0] + '\t' + item[1] + '\n')
        elif random_num < 0.9:
            fp_valid_data.write("1" + '\t' + item[0] + '\t' + item[1] + '\n')
        else:
            fp_test_data.write("1" + '\t' + item[0] + '\t' + item[1] + '\n')
    fp_train_data.close()
    fp_valid_data.close()
    fp_test_data.close()
    fp_all_news = open(config['datapath'] + config['all_news_file'], 'w', encoding='utf-8')
    for news in freq_news_set:
        fp_all_news.write(news + '\n')
    fp_all_news.close()

def build_doc_feature(config):#doc embedding for each news, and entity ids for each news
    print("constructing news features ... ")

    news_features = {}

    news_feature_dict = {}
    fp_train_news = open(config['datapath'] + config['train_news'], 'r', encoding='utf-8')
    for line in fp_train_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        news_feature_dict[newsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)
    # entityid, entity_freq, entity_position, entity_type
    fp_val_news = open(config['datapath'] + config['valid_news'], 'r', encoding='utf-8')
    for line in fp_val_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        news_feature_dict[newsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)

    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    for news in tqdm(news_feature_dict, total=len(news_feature_dict)):
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

def process_KPRN_data(config):#todo : remove the ring paths
    Train_data = build_train(config)
    entity_adj = torch.load(config['cache_path']+"/entity_adj.pt")
    relation_adj = torch.load(config['cache_path']+"/relation_adj.pt")
    doc_entity_dict = np.load(config['cache_path']+"/doc_entity_dict.npy", allow_pickle=True).item()
    entity_doc_dict = np.load(config['cache_path']+"/entity_doc_dict.npy", allow_pickle=True).item()
    hit_dict = np.load(config['cache_path']+"/hit_dict.npy", allow_pickle=True).item()
    
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
    os.makedirs(config['datapath'] + config['KPRN_train_file'].rsplit("/", 1)[0], exist_ok=True)
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
                    if data[item]["label"]!=1 and random.random()>=0.001:#filter negative samples
                        continue
                    if len(data[item]['paths']) > 20:#(news1, news2) pair can not have more than 20 paths
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
    print(count_1, count_0)#21013 39647

def build_ent_rel_2id(config):
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
    
    return entity_id_dict, relation_id_dict#note that the id starts from 1

def build_adj_matrix(config, entity_id_dict, relation_id_dict):
    print('constructing adjacency matrix ...')

    adj_file_fp = open(config['datapath']+config['kg_file'], 'r', encoding='utf-8')
    adj = {}
    for line in adj_file_fp:
        head, relation, tail = line.split('\n')[0].split('\t')#(head, relation, tail)
        if entity_id_dict[head] not in adj:
            adj[entity_id_dict[head]] = [(entity_id_dict[tail],relation_id_dict[relation])]
        else:
            adj[entity_id_dict[head]].append((entity_id_dict[tail],relation_id_dict[relation]))

    entity_adj = torch.zeros([len(entity_id_dict)+1, config['news_entity_num']], dtype=torch.long)
    relation_adj = torch.zeros([len(entity_id_dict)+1, config['news_entity_num']], dtype=torch.long)
    for key in adj:
        tail_rel = adj[key] + [(0,0)]*(config['news_entity_num']-len(adj[key])) if len(adj[key])<config['news_entity_num'] else random.sample(adj[key], config['news_entity_num'])
        entities = [x[0] for x in tail_rel]
        relations = [x[1] for x in tail_rel]
        entity_adj[key] = torch.tensor(entities, dtype=torch.long)
        relation_adj[key] = torch.tensor(relations, dtype=torch.long)

    return entity_adj, relation_adj

def build_entity_relation_embedding(config, entity_num, relation_num):
    print('constructing embedding ...')
    entity_embedding = np.zeros((entity_num+1, config['entity_embedding_size']))
    relation_embedding = np.zeros((relation_num+1, config['entity_embedding_size']))
    fp_entity_embedding = open(config['datapath']+config['entity_embedding_file'], 'r', encoding='utf-8')
    for i, line in enumerate(fp_entity_embedding):
        entity_embedding[i+1] = np.array(line.strip().split('\t')).astype(np.float)
    fp_relation_embedding = open(config['datapath']+config['relation_embedding_file'], 'r', encoding='utf-8')
    for i, line in enumerate(fp_relation_embedding):
        relation_embedding[i+1] = np.array(line.strip().split('\t')).astype(np.float)
    return torch.tensor(entity_embedding, dtype=torch.float32), torch.tensor(relation_embedding, dtype=torch.float32)

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
                    entity2doc[entity_id_dict[entity]] = set()
                entity2doc[entity_id_dict[entity]].add(newsid)
        if len(doc2entities[newsid]) > config['news_entity_num']:
            doc2entities[newsid] = doc2entities[newsid][:config['news_entity_num']]
        else:
            for i in range(config['news_entity_num']-len(doc2entities[newsid])):#todo
                doc2entities[newsid].append(0)
    for item in doc2entities:
        doc2entities[item] = torch.tensor(doc2entities[item], dtype=torch.long)
    # for item in entity2doc:
    #     if len(entity2doc[item])>config['news_entity_num']:
    #         entity2doc[item] = entity2doc[item][:config['news_entity_num']] #todo load entity in titles

    return doc2entities, entity2doc

def build_doc_feature_embedding(config):
    print('constructing doc feature embedding ...')
    fp_doc_feature = open(config['datapath']+config['doc_feature_embedding_file'], 'r', encoding='utf-8')
    doc_embedding_feature_dict = {}
    for line in fp_doc_feature:
        linesplit = line.split('\n')[0].split('\t')
        doc_embedding_feature_dict[linesplit[0]] = torch.tensor(np.array(linesplit[1].split(' ')).astype(np.float), dtype=torch.float32)
    return doc_embedding_feature_dict

def build_neibor_embedding(config, entity_doc_dict, doc_feature_embedding, entity_id_dict):#return doc embedding sum and num-1 for each entity, for calculating coherence reward
    print('build neiborhood embedding ...')
    entity_num = len(entity_id_dict)
    #neibor_embedding and neibor_num for each entity(inlcuding id=0)
    entity_neibor_embedding_list = torch.zeros([entity_num+1, config['doc_embedding_size']],dtype=torch.float32)
    entity_neibor_num_list = torch.ones(entity_num+1, dtype=torch.long)
    for entity in entity_doc_dict:
        entity_news_embedding_list = torch.zeros([len(entity_doc_dict[entity]), config['doc_embedding_size']], dtype=torch.float32)
        for i, news in enumerate(entity_doc_dict[entity]):
            entity_news_embedding_list[i] = doc_feature_embedding[news]
        entity_neibor_embedding_list[entity] = torch.sum(entity_news_embedding_list, dim=0)
        if len(entity_doc_dict[entity])>=2:
            entity_neibor_num_list[entity] = len(entity_doc_dict[entity])-1#-1 for news ifself in forward function
    return entity_neibor_embedding_list, entity_neibor_num_list#todo torch.tensor(entity_neibor_embedding_list).cuda(), torch.tensor(entity_neibor_num_list).cuda()

def build_hit_dict(config):
    print('constructing hit dict ...')
    hit_dict = {}
    fp_train = open(config['datapath']+config['pos_train_file'], 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        if linesplit[1] not in hit_dict:
            hit_dict[linesplit[1]] = set()
        if linesplit[2] not in hit_dict:
            hit_dict[linesplit[2]] = set()
        hit_dict[linesplit[1]].add(linesplit[2])
        hit_dict[linesplit[2]].add(linesplit[1])
    fp_train.close()

    train_val_hit_dict = copy.deepcopy(hit_dict)
    fp_val = open(config['datapath']+config['pos_val_file'], 'r', encoding='utf-8')
    for line in fp_val:
        linesplit = line.split('\n')[0].split('\t')
        if linesplit[1] not in train_val_hit_dict:
            train_val_hit_dict[linesplit[1]] = set()
        if linesplit[2] not in train_val_hit_dict:
            train_val_hit_dict[linesplit[2]] = set()
        train_val_hit_dict[linesplit[1]].add(linesplit[2])
        train_val_hit_dict[linesplit[2]].add(linesplit[1])
    fp_val.close()
    return hit_dict, train_val_hit_dict

def knowledge_neg_sampling():
    pass
    '''
    with open(config['datapath']+config['knowledge_neg_sample_train_file'], 'w', encoding='utf-8') as kg_fp_train:
        for item1, item2 in positive_pairs:
            rs_fp_train.write('1\t'+item1+'\t'+item2+'\n')
            k = random.sample([1,2,3], 1)[0]
            if k==1:
                one_hop_entity = set(doc_entity_dict[item1].numpy().tolist())
                one_hop_entity.remove(0)
                if len(one_hop_entity)==0:
                    continue
                neg_news_list=[]
                for ent in one_hop_entity:
                    try:
                        neg_news_list.extend(entity_doc_dict[ent])
                    except:
                        pass
                neg_news_list = list(set(neg_news_list)-set([item1, item2]))
                if len(neg_news_list)==0:
                    continue
                neg_news = random.sample(neg_news_list, 1)[0]
                if (item1, neg_news) not in positive_pairs and (neg_news, item1) not in positive_pairs:
                    kg_fp_train.write('0\t'+item1+'\t'+neg_news+'\n')
    '''         

def random_neg_sampling(config, all_news, positive_pairs, data_set, fp):
    for item1, item2 in data_set:
        fp.write('1\t'+item1+'\t'+item2+'\n')
        while True:
            flag=0
            negative_items = random.sample(all_news, config['train_neg_num'])
            for item in negative_items:
                if (item1, item) in positive_pairs or (item, item1) in positive_pairs:
                    flag=1
                    break
            if flag==0:
                break
        for item in negative_items:
            fp.write('0\t'+item1+'\t'+item+'\n')

def negative_sampling(config):
    all_news = []
    with open(config['datapath']+config['all_news_file'], 'r', encoding='utf-8') as fp_all_news:
        for line in fp_all_news:
            news = line.strip()
            all_news.append(news)

    positive_pairs=set()
    with open(config['datapath']+config['pos_train_file'], 'r', encoding='utf-8') as fp_train:
        for line in fp_train:
            linesplit = line.split('\n')[0].split('\t')
            positive_pairs.add((linesplit[1], linesplit[2]))
    
    rs_fp_train = open(config['datapath']+config['train_file'], 'w', encoding='utf-8')
    random_neg_sampling(config, all_news, positive_pairs, positive_pairs, rs_fp_train)
    rs_fp_train.close()
    
    val_positive_pairs=set()
    with open(config['datapath']+config['pos_val_file'], 'r', encoding='utf-8') as fp_val:
        for line in fp_val:
            linesplit = line.split('\n')[0].split('\t')
            positive_pairs.add((linesplit[1], linesplit[2]))
            val_positive_pairs.add((linesplit[1], linesplit[2]))

    rs_fp_val = open(config['datapath']+config['val_file'], 'w', encoding='utf-8')
    random_neg_sampling(config, all_news, positive_pairs, val_positive_pairs, rs_fp_val)
    rs_fp_val.close()              

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
    valid_data = {}
    item1 = []
    item2 = []
    label = []
    fp_train = open(config['datapath']+config['val_file'], 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        item1.append(linesplit[1])
        item2.append(linesplit[2])
        label.append(float(linesplit[0]))
    valid_data['item1'] = item1
    valid_data['item2'] = item2
    valid_data['label'] = label
    return valid_data

def build_test(config):#create all positive item list for every item
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


# def cache_data(config):
#     os.makedirs(config['datapath']+"/cache", exist_ok=True)
#     if os.path.exists(config['datapath']+"/cache/entity_adj.pt"):
#         entity_adj = torch.load(config['datapath']+"/cache/entity_adj.pt")
#         relation_adj = torch.load(config['datapath']+"/cache/relation_adj.pt")
#         entity_id_dict = np.load(config['datapath']+"/cache/entity_id_dict.npy", allow_pickle=True).item()
#         relation_id_dict = np.load(config['datapath']+"/cache/relation_id_dict.npy", allow_pickle=True).item()
#         kg_env = None #nx.read_gpickle(config['datapath']+"/cache/kg_env.gpickle")
#         entity_embedding = torch.load(config['datapath']+"/cache/entity_embedding.pt")
#         relation_embedding = torch.load(config['datapath']+"/cache/relation_embedding.pt")
#     else:
#         entity_adj, relation_adj, entity_id_dict, relation_id_dict, kg_env = build_network(config)
#         entity_embedding, relation_embedding = build_entity_relation_embedding(config, len(entity_id_dict), len(relation_id_dict))
#         os.makedirs(config['datapath']+"/cache", exist_ok=True)
#         torch.save(entity_adj, config['datapath']+"/cache/entity_adj.pt")
#         torch.save(relation_adj, config['datapath']+"/cache/relation_adj.pt")
#         np.save(config['datapath']+"/cache/entity_id_dict.npy", entity_id_dict)
#         np.save(config['datapath']+"/cache/relation_id_dict.npy", relation_id_dict)
#         nx.write_gpickle(kg_env, config['datapath']+"/cache/kg_env.gpickle")
#         torch.save(entity_embedding, config['datapath']+"/cache/entity_embedding.pt")
#         torch.save(relation_embedding, config['datapath']+"/cache/relation_embedding.pt")

# def load_doc_feature(config):
#     doc_embeddings = {}
#     fp_news_feature = open(config['datapath']+config['doc_feature_entity_file'], 'r', encoding='utf-8')
#     for line in fp_news_feature:
#         newsid, news_embedding = line.strip().split('\t')
#         news_embedding = news_embedding.split(',')
#         doc_embeddings[newsid] = news_embedding
#     return doc_embeddings

# def get_anchor_graph_data(config):
#     print('constructing anchor doc ...')
#     test_data = {}
#     item1 = []
#     item2 = []
#     label = []
#     fp_news_entities = open(config['datapath']+config['doc_feature_entity_file'], 'r', encoding='utf-8')
#     for line in fp_news_entities:
#         linesplit = line.strip().split('\t')
#         newsid = linesplit[0]
#         item1.append(newsid)
#         item2.append(newsid)
#         label.append(0.0)

#     test_data['item1'] = item1
#     test_data['item2'] = item2
#     test_data['label'] = label
#     return test_data