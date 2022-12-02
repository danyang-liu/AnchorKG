from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils.util import *
from KPRN_train import KPRN_Dataset

class NewsDataset(Dataset):
    def __init__(self, dic_data, transform=None):
        self.dic_data = dic_data
        self.transform = transform
    def __len__(self):
        return len(self.dic_data['label'])
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'item1': self.dic_data['item1'][idx], 'item2': self.dic_data['item2'][idx], 'label': self.dic_data['label'][idx]}
        return sample

def process_data_and_cache(config):
    entity_id_dict, relation_id_dict = build_ent_rel_2id(config)
    entity_adj, relation_adj = build_adj_matrix(config, entity_id_dict, relation_id_dict)
    entity_embedding, relation_embedding = build_entity_relation_embedding(config, len(entity_id_dict), len(relation_id_dict))
    doc_entity_dict, entity_doc_dict = load_news_entity(config, entity_id_dict)
    doc_feature_embedding = build_doc_feature_embedding(config)
    neibor_embedding, neibor_num = build_neibor_embedding(config, entity_doc_dict, doc_feature_embedding, entity_id_dict)
    hit_dict, train_val_hit_dict = build_hit_dict(config)
    negative_sampling(config)

    os.makedirs(config['cache_path'], exist_ok=True)
    np.save(config['cache_path']+"/entity_id_dict.npy", entity_id_dict)
    np.save(config['cache_path']+"/relation_id_dict.npy", relation_id_dict)
    torch.save(entity_adj, config['cache_path']+"/entity_adj.pt")
    torch.save(relation_adj, config['cache_path']+"/relation_adj.pt")
    torch.save(entity_embedding, config['cache_path']+"/entity_embedding.pt")
    torch.save(relation_embedding, config['cache_path']+"/relation_embedding.pt")
    np.save(config['cache_path']+"/doc_entity_dict.npy", doc_entity_dict)
    np.save(config['cache_path']+"/entity_doc_dict.npy", entity_doc_dict)
    np.save(config['cache_path']+"/doc_feature_embedding.npy", doc_feature_embedding)
    torch.save(neibor_embedding, config['cache_path']+"/neibor_embedding.pt")
    torch.save(neibor_num, config['cache_path']+"/neibor_num.pt")
    np.save(config['cache_path']+"/hit_dict.npy", hit_dict)
    np.save(config['cache_path']+"/train_val_hit_dict.npy", train_val_hit_dict)
    return entity_id_dict, relation_id_dict, entity_adj, relation_adj, entity_embedding, relation_embedding, doc_entity_dict, entity_doc_dict, doc_feature_embedding, neibor_embedding, neibor_num, hit_dict, train_val_hit_dict

def load_data(config):

    if os.path.exists(config['cache_path']):
        print("Loading data from cache...")
        entity_id_dict = np.load(config['cache_path']+"/entity_id_dict.npy", allow_pickle=True).item()
        #relation_id_dict = np.load(config['cache_path']+"/relation_id_dict.npy", allow_pickle=True).item()
        entity_adj = torch.load(config['cache_path']+"/entity_adj.pt")
        relation_adj = torch.load(config['cache_path']+"/relation_adj.pt")
        entity_embedding = torch.load(config['cache_path']+"/entity_embedding.pt")
        relation_embedding = torch.load(config['cache_path']+"/relation_embedding.pt")
        doc_entity_dict = np.load(config['cache_path']+"/doc_entity_dict.npy", allow_pickle=True).item()
        entity_doc_dict = np.load(config['cache_path']+"/entity_doc_dict.npy", allow_pickle=True).item()
        doc_feature_embedding = np.load(config['cache_path']+"/doc_feature_embedding.npy", allow_pickle=True).item()
        neibor_embedding = torch.load(config['cache_path']+"/neibor_embedding.pt")
        neibor_num = torch.load(config['cache_path']+"/neibor_num.pt")
        hit_dict = np.load(config['cache_path']+"/hit_dict.npy", allow_pickle=True).item()
        train_val_hit_dict = np.load(config['cache_path']+"/train_val_hit_dict.npy", allow_pickle=True).item()
    else:
        entity_id_dict, relation_id_dict, entity_adj, relation_adj, entity_embedding, relation_embedding, doc_entity_dict, entity_doc_dict, doc_feature_embedding, neibor_embedding, neibor_num, hit_dict, train_val_hit_dict = process_data_and_cache(config)

    Train_data = build_train(config)
    Val_data = build_val(config)
    Test_data = build_test(config)
    train_dataset = NewsDataset(Train_data)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['batch_size'],
        sampler=RandomSampler(train_dataset),
        num_workers=config['num_workers'],
    )

    if config['warm_up']:
        with open(config['datapath']+config['warm_up_train_file'], "r") as f:
            warmup_train_data = [json.loads(line) for line in f.readlines()]
        with open(config['datapath']+config['warm_up_dev_file'], "r") as f:
            warmup_dev_data = [json.loads(line) for line in f.readlines()]
        
        warmup_train_dataset = KPRN_Dataset(warmup_train_data)
        warmup_dev_dataset = KPRN_Dataset(warmup_dev_data)

        def collate_fn(data):
            batch = {'label':[], 'item1':[], 'item2':[], 'paths':[], 'edges':[]}
            for item in data:
                path_len = len(item['paths'])
                batch['label'].append([item['label']]*path_len + [-1]*(3-path_len))
                batch['item1'].append(item['item1'])
                batch['item2'].append(item['item2'])
                batch['paths'].append(item['paths']+[0]*(3-path_len))
                batch['edges'].append(item['edges']+[0]*(3-path_len))

            batch['label'] = torch.tensor(batch['label'], dtype=torch.float)
            batch['item1'] = [item['item1'] for item in data]
            batch['item2'] = [item['item2'] for item in data]
            batch['paths'] = torch.tensor(batch['paths'], dtype=torch.long)
            batch['edges'] = torch.tensor(batch['edges'], dtype=torch.long)
            return batch

        warmup_train_dataloader = DataLoader(
            dataset=warmup_train_dataset, 
            batch_size=config['warmup_batch_size'],
            sampler=RandomSampler(warmup_train_dataset),
            num_workers=config['warmup_num_workers'],
            collate_fn=collate_fn
        )
        warmup_dev_dataloader = DataLoader(
            dataset=warmup_dev_dataset,
            batch_size=config['warmup_batch_size'],
            sampler=SequentialSampler(warmup_dev_dataset),
            num_workers=config['warmup_num_workers'],
            collate_fn=collate_fn
        )
    else:
        warmup_train_dataloader = None
        warmup_dev_dataloader = None

    print("fininsh loading data!")

    return warmup_train_dataloader, warmup_dev_dataloader, train_dataloader, Val_data, Test_data, doc_feature_embedding, entity_adj, relation_adj, entity_id_dict, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, hit_dict, train_val_hit_dict

