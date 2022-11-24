import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils.parse_config import ConfigParser
import argparse
from utils.util import *
from utils.metrics import *
import json
import random
import time
from utils.pytorchtools import *

class KPRN(nn.Module):
    def __init__(self, config, doc_feature_embedding, entity_embedding, relation_embedding, device=torch.device('cpu')):
        super(KPRN, self).__init__()
        self.device = device
        self.config = config
        self.gamma = config["gamma"]
        self.doc_feature_embedding = doc_feature_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)

        self.news_compress = nn.Sequential(
                                nn.Linear(self.config['doc_embedding_size'], self.config['embedding_size']),
                                nn.ELU(),
                                nn.Linear(self.config['embedding_size'], self.config['embedding_size']),
                                nn.Tanh()
                            ).to(device)
        self.entity_compress =  nn.Sequential(
                                    nn.Linear(self.config['entity_embedding_size'], self.config['embedding_size']),
                                    nn.Tanh(),
                                ).to(device)
        self.relation_compress = nn.Sequential(
                                    nn.Linear(self.config['entity_embedding_size'], self.config['embedding_size']),
                                    nn.Tanh(),
                                ).to(device)
        
        self.lstm = nn.LSTM(2*self.config['embedding_size'], self.config['embedding_size'], batch_first=True).to(device)
        self.mlp = nn.Sequential(
                        nn.Linear(self.config['embedding_size'], self.config['embedding_size']),
                        nn.ReLU(),
                        nn.Linear(self.config['embedding_size'], 1)
                    ).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        batch_predict = []
        batch_path_scores = []
        for item1, item2, paths, edges in zip(data['item1'], data['item2'], data['paths'], data['edges']):
            news1 = self.news_compress(self.doc_feature_embedding[item1].to(self.device)).unsqueeze(dim=0)
            news2 = self.news_compress(self.doc_feature_embedding[item2].to(self.device)).unsqueeze(dim=0)
            path_scores=[]
            for path, edge in zip(paths, edges):
                path_node_embeddings = self.entity_compress(self.entity_embedding(torch.tensor(path)).to(self.device))#(path_len-2, embedding_size)
                path_edge_embeddings = self.relation_compress(self.relation_embedding(torch.tensor(edge+[0,0])).to(self.device))#(path_len, embedding_size)
                path_node_embeddings = torch.cat((news1, path_node_embeddings, news2), dim=0) #(path_len, embedding_size)
                path_node_embeddings = torch.unsqueeze(path_node_embeddings, 0)#(1, path_len, embedding_size)
                path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 0)#(1, path_len, embedding_size)
                output, _ = self.lstm(torch.cat((path_node_embeddings, path_edge_embeddings), dim=2))#(1, path_len, embedding_size)
                path_score = self.mlp(torch.squeeze(output)[-1])#[1]
                path_scores.append(path_score)
            path_scores = torch.stack(path_scores, dim=0) #(path_num, 1)
            batch_path_scores.append(path_scores)
            predict = self.sigmoid(torch.logsumexp(path_scores /self.gamma, dim=0))#[1]#path_scores.shape[0]
            batch_predict.append(predict)
        
        predicts = torch.cat(batch_predict, dim=0) #[batch_size]
        loss_fn = nn.BCELoss()
        loss = loss_fn(predicts, data['label'].to(self.device))
        return loss, predicts, batch_path_scores

class KPRN_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return sample


class KPRN_Trainer():
    def __init__(self, config, model, train_dataloader, dev_dataloader, test_dataloader, device) -> None:
        super().__init__()  
        self.config=config
        self.logger = config.get_logger('trainer', config['verbosity'])
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.epochs = config['epochs']
        self.num_train_steps = int(len(train_dataloader) * self.epochs)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        
        #self.scheduler = ExponentialLR(self.optimizer, gamma=0.98)
        self.early_stopping = EarlyStopping(patience=config['early_stop'], greater=True)
        self.ckpt_dir = config.save_dir
    
    def train(self):
        result = self._valid_epoch(-1)
        for epoch in range(1, self.epochs+1):
            self.logger.info("Epoch {}/{}".format(epoch, self.epochs))
            self.logger.info("Training")
            self._train_epoch(epoch)

            self.logger.info("Validation...")
            result = self._valid_epoch(epoch)
            auc_score = result['auc_score']
            if self.config['use_nni']:
                nni.report_intermediate_result({'default': auc_score})

            self.early_stopping(auc_score)
            if self.early_stopping.early_stop:
                self.logger.info("Early stop at epoch {}, best auc score: {:.5f}".format(epoch, self.early_stopping.best_score))
                break
            elif self.early_stopping.counter == 0:
                self._save_checkpoint(epoch)
        if self.config['use_nni']:
            nni.report_final_result({"default":self.early_stopping.best_score})
                    
            
    def _train_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()
        t1 = time.time()
        for batch in self.train_dataloader:
            loss, _, _ = self.model(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()
            epoch_loss += loss.item()
        t2 = time.time()
        self.logger.info("epoch {}, train Loss {:.5f}, time {:.5f}".format(epoch, epoch_loss/len(self.train_dataloader), t2-t1))

    def _valid_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_predicts = []
            dev_labels = []
            for batch in self.dev_dataloader:
                loss, predicts, _ = self.model(batch)
                dev_loss += loss.item()
                dev_predicts.extend(predicts.tolist())
                dev_labels.extend(batch['label'])
            dev_predicts = np.array(dev_predicts)
            dev_labels = np.array(dev_labels)
            auc_score = cal_auc(dev_labels, dev_predicts)
            self.logger.info("epoch: {}, dev_loss: {:.5f}, dev_auc: {:.5f}".format(epoch, dev_loss/len(dev_dataloader), auc_score))
        return {'auc_score': auc_score}


    def predict(self):
        ckpt_path = os.path.join(self.ckpt_dir, 'KPRN_model.ckpt')
        #ckpt_path = "./out/saved/models/KPRN/fy6ad2rp/NhjQB-1109_211741/KPRN_model.ckpt"
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()
        with torch.no_grad():
            output_path = []
            for batch in self.test_dataloader:
                _, _, batch_path_scores = self.model(batch)#(batch_size, num_paths, 1)
                for i in range(len(batch['item1'])):
                    label = batch['label'][i].item()
                    item1 = batch['item1'][i]
                    item2 = batch['item2'][i]
                    paths = batch['paths'][i]
                    edges = batch['edges'][i]
                    try:
                        indices = batch_path_scores[i].reshape(-1).topk(2).indices.tolist()
                    except:
                        indices = [0]
                    for index in indices:
                        path = paths[index]
                        edge = edges[index]
                        output_path.append({'label': label, 'item1': item1, 'item2': item2, 'paths': path, 'edges': edge})
        with open(self.config['datapath']+self.config['KPRN_predict_train_file'], "w") as f1:
            with open(self.config['datapath']+self.config['KPRN_predict_dev_file'], "w") as f2:
                for _ in output_path:
                    if random.random() < 0.8:
                        f1.write(json.dumps(_)+"\n")
                    else:
                        f2.write(json.dumps(_)+"\n")
                                
    
    def _save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, 'KPRN_model.ckpt')
        torch.save(self.model.state_dict(), ckpt_path)
        self.logger.info("Saving model checkpoint to {}".format(ckpt_path))

def collate_fn(data):
    batch = {}
    batch['label'] = torch.tensor([item['label'] for item in data], dtype=torch.float)
    batch['item1'] = [item['item1'] for item in data]
    batch['item2'] = [item['item2'] for item in data]
    batch['paths'] = [item['paths'] for item in data]
    batch['edges'] = [item['edges'] for item in data]
    return batch

def create_dataloaders(config):
    with open(config['datapath']+config['KPRN_train_file'], "r") as f:
        train_data = [json.loads(line) for line in f.readlines()]
    with open(config['datapath']+config['KPRN_val_file'], "r") as f:
        dev_data = [json.loads(line) for line in f.readlines()]

    train_dataset = KPRN_Dataset(train_data)
    dev_dataset = KPRN_Dataset(dev_data)
    test_dataset = KPRN_Dataset(train_data + dev_data)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        sampler=RandomSampler(train_dataset),
        num_workers=0,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=config['batch_size'],
        sampler=SequentialSampler(dev_dataset),
        num_workers=0,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        sampler=SequentialSampler(test_dataset),
        num_workers=0,
        collate_fn=collate_fn
    )
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KPRN')

    parser.add_argument('-c', '--config', default="./config/KPRN_config.json", type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--use_nni', action='store_true', help='use nni to tune hyperparameters')

    config = ConfigParser.from_args(parser)
    if config['use_nni']:
        import nni
    seed_everything(config['seed'])
    device, deviceids = prepare_device(config['n_gpu'])

    #dataset
    train_dataloader, dev_dataloader, test_dataloader = create_dataloaders(config)

    #model
    if os.path.exists(config['datapath']+"/cache/entity_embedding.pt"):
        entity_embedding = torch.load(config['datapath']+"/cache/entity_embedding.pt")
        relation_embedding = torch.load(config['datapath']+"/cache/relation_embedding.pt")
    else:
        entity_adj, relation_adj, entity_id_dict, relation_id_dict, kg_env = build_network(config)
        entity_embedding, relation_embedding = build_entity_relation_embedding(config, len(entity_id_dict), len(relation_id_dict))
    doc_feature_embedding = build_doc_feature_embedding(config)

    model = KPRN(config, doc_feature_embedding, entity_embedding, relation_embedding, device=device)

    trainer = KPRN_Trainer(config, model, train_dataloader, dev_dataloader, test_dataloader, device=device)

    trainer.logger.info("config {}".format(config.config))
    trainer.logger.info("save dir {}".format(config.save_dir))
    trainer.logger.info("log dir {}".format(config.log_dir))
    trainer.logger.info("model {}".format(model))

    trainer.train()
    trainer.predict()

