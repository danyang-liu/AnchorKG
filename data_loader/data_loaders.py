from base.base_data_loader import BaseDataLoader
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.util import *

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

class NewsDataLoader(BaseDataLoader):
    """
        News data loading using BaseDataLoader
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        super().__init__(self.dataset, batch_size, shuffle, num_workers)



def load_data(config):

    Train_data = build_train(config)
    Val_data = build_val(config)
    Test_data = build_test(config)
    doc_feature_embedding = build_doc_feature_embedding(config)
    entity_adj, relation_adj, entity_id_dict, kg_env = build_network(config)
    doc_entity_dict, entity_doc_dict = load_news_entity(config)
    neibor_embedding, neibor_num = build_neibor_embedding(config, entity_doc_dict, doc_feature_embedding)
    entity_embedding, relation_embedding = build_entity_relation_embedding(config)
    hit_dict = build_hit_dict(config)
    train_data = NewsDataset(Train_data)
    train_dataloader = DataLoader(train_data, batch_size=config['data_loader']['batch_size'])

    if config['trainer']['warm_up']:
        warm_up_train_data = load_warm_up(config)
        warm_up_test_data = load_warm_up(config)
        warmup_train_data = NewsDataset(warm_up_train_data)
        warmup_train_dataloader = DataLoader(warmup_train_data, batch_size=config['data_loader']['batch_size'])
    else:
        warmup_train_dataloader = None
        warm_up_test_data = None

    print("fininsh loading data!")

    return warmup_train_dataloader, warm_up_test_data, train_dataloader, Val_data, Test_data, doc_feature_embedding, entity_adj, relation_adj, entity_id_dict, kg_env, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, hit_dict

