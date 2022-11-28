import time
from numpy import inf
import torch
from utils.metrics import *
from utils.pytorchtools import *
from utils.logger import *
from model.AnchorKG import *
from tqdm import tqdm
import nni
import hnswlib
import json

class Trainer():
    """
    Trainer class
    """
    def __init__(self, config, model_anchor, model_recommender, model_reasoner, device, data):
        super().__init__()

        self.config = config
        self.logger = config.get_logger('trainer', config['verbosity'])

        self.model_anchor = model_anchor
        self.model_recommender = model_recommender
        self.model_reasoner = model_reasoner

        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer_anchor = torch.optim.Adam(self.model_anchor.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.optimizer_recommender = torch.optim.Adam(self.model_recommender.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.optimizer_reasoner = torch.optim.Adam(self.model_reasoner.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        self.epochs = config['epochs']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.device = device

        self.warmup_train_dataloader = data[0]
        self.warmup_dev_dataloader = data[1]
        self.train_dataloader = data[2]
        self.val_data = data[3]
        self.test_data = data[4]

    def actor_critic_loss(self, rewards_steps, act_probs_steps, state_values_steps, embedding_loss, reasoning_loss, actor_loss_list, critic_loss_list):
        #rewards_steps:[(batch, 5), (batch, 15), (batch, 30);  
        # act_probs_steps:[(batch, 5), (batch, 15), (batch, 30)];    
        # state_values_steps:[(batch, 1), (batch,5,1), (batch,15,1)]
        num_steps = len(rewards_steps)
        for i in range(num_steps):
            shape0 = state_values_steps[i].shape[0]
            shape1 = state_values_steps[i].shape[1]
            shape2 = self.config['topk'][i]
            baseline = state_values_steps[i] if i>=1 else state_values_steps[i][:,:,None]
            if i==num_steps-1:
                terminal_reward = self.config['alpha1'] * (1 - embedding_loss.detach()) + (1-self.config['alpha1']) * (1 - reasoning_loss.detach())
                td_error = self.config['alpha2'] * rewards_steps[i] + ((1-self.config['alpha2']) * terminal_reward)[:,None] - baseline.expand(shape0, shape1, shape2).reshape(shape0, shape1*shape2)
            else:
                td_error = self.config['alpha2']*rewards_steps[i] + (self.config['gamma']*state_values_steps[i+1].squeeze(-1)).detach() - baseline.expand(shape0, shape1, shape2).reshape(shape0, shape1*shape2)
            actor_loss = - torch.log(act_probs_steps[i]) * td_error.detach()
            critic_loss = td_error.pow(2)
            actor_loss_list.append(actor_loss.mean())
            critic_loss_list.append(critic_loss.mean())

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_anchor.train()
        self.model_recommender.train()
        self.model_reasoner.train()
        anchor_all_loss = 0
        embedding_all_loss = 0
        reasoning_all_loss = 0
        time_anchor = 0
        time_recommender = 0
        time_reasoner = 0
        time_optimize = 0
        for _, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            t1=time.time()
            act_probs_steps1, state_values_steps1, rewards_steps1, anchor_graph1, anchor_relation1 = self.model_anchor(batch['item1'])
            act_probs_steps2, state_values_steps2, rewards_steps2, anchor_graph2, anchor_relation2 = self.model_anchor(batch['item2'])
            t2=time.time()
            embedding_predict = self.model_recommender(batch['item1'], batch['item2'], anchor_graph1, anchor_graph2)[0]#similarities between item1 and item2，normalize to [0,1]
            t3=time.time()
            reasoning_predict = self.model_reasoner(batch['item1'], batch['item2'], anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2)[0]
            t4=time.time()

            embedding_loss = self.criterion(embedding_predict, batch['label'].to(self.device).float())
            reasoning_loss = self.criterion(reasoning_predict, batch['label'].to(self.device).float())

            embedding_loss_mean = torch.mean(embedding_loss)
            reasoning_loss_mean = torch.mean(reasoning_loss)

            embedding_all_loss = embedding_all_loss + embedding_loss_mean.item()
            reasoning_all_loss = reasoning_all_loss + reasoning_loss_mean.item()

            self.optimizer_recommender.zero_grad()
            embedding_loss_mean.backward(retain_graph=True)
            self.optimizer_recommender.step()

            self.optimizer_reasoner.zero_grad()
            reasoning_loss_mean.backward(retain_graph=True)
            self.optimizer_reasoner.step()

            actor_loss_list = []
            critic_loss_list = []
            self.actor_critic_loss(rewards_steps1, act_probs_steps1, state_values_steps1, embedding_loss, reasoning_loss, actor_loss_list, critic_loss_list)
            self.actor_critic_loss(rewards_steps2, act_probs_steps2, state_values_steps2, embedding_loss, reasoning_loss, actor_loss_list, critic_loss_list)

            actor_losses = torch.stack(actor_loss_list).sum()
            critic_losses = torch.stack(critic_loss_list).sum()
            anchor_loss = actor_losses + critic_losses

            self.optimizer_anchor.zero_grad()
            anchor_loss.backward()
            self.optimizer_anchor.step()
            anchor_all_loss = anchor_all_loss + anchor_loss.item()

            #torch.cuda.empty_cache()
            t5=time.time()
            time_anchor += t2-t1
            time_recommender += t3-t2
            time_reasoner += t4-t3
            time_optimize += t5-t4
            # print(embedding_loss_mean.item())
            # print(reasoning_loss_mean.item())
            # print(actor_losses.item())
            # print(critic_losses.item())
        
        self.logger.info("time_anchor: {:.5f}, time_recommender: {:.5f}, time_reasoner: {:.5f}, time_optimize: {:.5f}".format(time_anchor, time_recommender, time_reasoner, time_optimize))
        self.logger.info("anchor all loss :{:.5f}".format(anchor_all_loss/len(self.train_dataloader)))
        self.logger.info("embedding all loss :{:.5f}".format(embedding_all_loss/len(self.train_dataloader)))
        self.logger.info("reasoning all loss :{:.5f}".format(reasoning_all_loss/len(self.train_dataloader)))

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_anchor.eval()
        self.model_recommender.eval()
        self.model_reasoner.eval()

        # get all news embeddings
        y_pred = []
        start_list = list(range(0, len(self.val_data['label']), self.config['batch_size']))
        for start in tqdm(start_list, total=len(start_list)):
            end = start + self.config['batch_size']
            _, _, _, anchor_graph1, anchor_relation1 = self.model_anchor(self.val_data['item1'][start:end])
            _, _, _, anchor_graph2, anchor_relation2 = self.model_anchor(self.val_data['item2'][start:end])
            embedding_predict = self.model_recommender(self.val_data['item1'][start:end], self.val_data['item2'][start:end], anchor_graph1, anchor_graph2)[0]
            reasoning_predict = self.model_reasoner(self.val_data['item1'][start:end], self.val_data['item2'][start:end], anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2)[0]
            predict = self.config['alpha1'] * embedding_predict + (1-self.config['alpha1']) * reasoning_predict
            y_pred.extend(predict.cpu().data.numpy())

        truth = self.val_data['label']
        auc_score = cal_auc(truth, y_pred)
        self.logger.info("epoch: {}, auc: {:.5f}".format(epoch, auc_score))
        return auc_score

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if save_best:
            filename_anchor = str(self.checkpoint_dir / 'model_anchor.ckpt')
            filename_recommender = str(self.checkpoint_dir / 'model_recommender.ckpt')
            filename_reasoner = str(self.checkpoint_dir / 'model_reasoner.ckpt')
        else:
            filename_anchor = str(self.checkpoint_dir / f'model_anchor-{epoch}.ckpt')
            filename_recommender = str(self.checkpoint_dir / f'model_recommender-{epoch}.ckpt')
            filename_reasoner = str(self.checkpoint_dir / f'model_reasoner-{epoch}.ckpt')

        state_anchor = self.model_anchor.state_dict()
        state_recommender = self.model_recommender.state_dict()
        state_reasoner = self.model_reasoner.state_dict()

        torch.save(state_anchor, filename_anchor)
        self.logger.info("Saving checkpoint: {} ...".format(filename_anchor))

        torch.save(state_recommender, filename_recommender)
        self.logger.info("Saving checkpoint: {} ...".format(filename_recommender))

        torch.save(state_reasoner, filename_reasoner)
        self.logger.info("Saving checkpoint: {} ...".format(filename_reasoner))

    def train(self):
        """
            Full training logic
        """
        if self.config['warm_up_ckpt']:
            self.logger.info("warm up ckpt loading")
            self.model_anchor.load_state_dict(torch.load(self.config['warm_up_ckpt']))
        # warm up training stage
        elif self.config['warm_up']:
            self.warm_up()
            self.model_anchor.load_state_dict(torch.load(str(self.checkpoint_dir / 'warmup_model.ckpt')))

        self.logger.info("anchor graph training")
        valid_scores = []
        early_stopping = EarlyStopping(patience=self.config['early_stop'], greater=True)
        self.logger.info("Validation before training")
        valid_score = self._valid_epoch(0)
        valid_scores.append(valid_score)
        for epoch in range(self.start_epoch, self.epochs+1):
            self.logger.info("Epoch {}/{}".format(epoch, self.epochs))
            self.logger.info("Training")
            self._train_epoch(epoch)

            self.logger.info("Validation...")
            valid_score = self._valid_epoch(epoch)
            valid_scores.append(valid_score)
            if self.config['use_nni']:
                nni.report_intermediate_result({'default': valid_score})

            self._save_checkpoint(epoch)
            early_stopping(valid_score)
            if early_stopping.early_stop:
                self.logger.info("Early stop at epoch {}, best auc score: {:.5f}".format(epoch, early_stopping.best_score))
                break
            elif early_stopping.counter==0:
                self._save_checkpoint(epoch, save_best=True)
                
        if self.config['use_nni']:
            nni.report_final_result({"default": early_stopping.best_score})

    def warm_up(self):
        self.logger.info("warm up training")
        warmup_model = self.model_anchor
        optimizer_warmup = torch.optim.Adam(warmup_model.parameters(), lr=self.config['warmup_lr'], weight_decay=self.config['warmup_weight_decay'])
    
        early_stopping = EarlyStopping(patience=self.config['warmup_early_stop'], greater=True)
        for epoch in range(self.config['warmup_epochs']):
            warmup_model.train()
            warmup_all_loss = 0
            for _, batch in enumerate(self.warmup_train_dataloader):
                warmup_loss, _, _ = warmup_model.warm_train(batch)
                optimizer_warmup.zero_grad()
                warmup_loss.backward()
                optimizer_warmup.step()
                warmup_all_loss += warmup_loss.item()
            self.logger.info("epoch {}, warmup train Loss: {:.5f}".format(epoch, warmup_all_loss/len(self.warmup_train_dataloader)))

            warmup_model.eval()
            with torch.no_grad():
                dev_loss = 0
                y_preds = []
                y_labels = []
                for _, batch in enumerate(self.warmup_dev_dataloader):
                    warmup_loss, y_pred, y_label = warmup_model.warm_train(batch)
                    dev_loss += warmup_loss.item()
                    y_preds.extend(y_pred.tolist())
                    y_labels.extend(y_label.tolist())
                auc_score = cal_auc(y_labels, y_preds)
            self.logger.info("epoch: {}, warmup dev_loss: {:.5f}, warmup dev_auc: {:.5f}".format(epoch, dev_loss/len(self.warmup_dev_dataloader), auc_score))
            if self.config['use_nni'] and epoch % 5 == 0:
                nni.report_intermediate_result({'default': auc_score})
                
            early_stopping(auc_score)
            if early_stopping.early_stop:
                self.logger.info("warmup Early stop at epoch {}, best warmup auc score: {:.5f}".format(epoch, early_stopping.best_score))
                break
            elif early_stopping.counter==0:
                torch.save(warmup_model.state_dict(), str(self.checkpoint_dir / 'warmup_model.ckpt'))
                self.logger.info("Saving model checkpoint to {}".format(str(self.checkpoint_dir / 'warmup_model.ckpt')))
            
        if self.config['use_nni']:
            nni.report_final_result({"default": early_stopping.best_score})

    def test(self):
        self.logger.info('testing')
        #load model
        self.model_anchor.load_state_dict(torch.load(str(self.checkpoint_dir / 'model_anchor.ckpt')))
        self.model_recommender.load_state_dict(torch.load(str(self.checkpoint_dir / 'model_recommender.ckpt')))
        self.model_reasoner.load_state_dict(torch.load(str(self.checkpoint_dir / 'model_reasoner.ckpt')))
    
        #test
        self.model_anchor.eval()
        self.model_recommender.eval()
        self.model_reasoner.eval()

        # get all news embeddings
        doc_list = list(self.test_data.keys())
        self.logger.info('len(doc_list) : {}'.format(len(doc_list)))
        start_list = list(range(0, len(doc_list), self.config['batch_size']))
        doc_embedding = []
        doc_embedding_dict = {}
        for start in start_list:
            end = start + self.config['batch_size']
            _, _, _, anchor_graph1, _ = self.model_anchor(doc_list[start:end])
            doc_embedding.extend(self.model_recommender(doc_list[start:end], doc_list[start:end], anchor_graph1, anchor_graph1)[1].cpu().data.numpy())
            
        # knn search topk
        ann = hnswlib.Index(space='cosine', dim=128)
        ann.init_index(max_elements=len(doc_list), ef_construction=200, M=16)
        for i in range(len(doc_list)):
            doc_embedding_dict[doc_list[i]] = doc_embedding[i]
            ann.add_items(doc_embedding[i], i)
        ann.set_ef(100)
        predict_dict = {}
        for doc in self.test_data:
            doc_embedding = doc_embedding_dict[doc]
            labels, distances = ann.knn_query(doc_embedding, k=10)
            predict_dict[doc] = list(map(lambda x: doc_list[x], labels[0]))
        # compute metric
        avg_precision, avg_recall, avg_ndcg, avg_hit, invalid_users = evaluate(predict_dict, self.test_data)
        self.logger.info('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))
        
    def predict_anchor_graph(self):
        self.logger.info('predicting anchor graph')
        #load model
        self.model_anchor.load_state_dict(torch.load(str(self.checkpoint_dir / 'model_anchor.ckpt')))
        self.model_recommender.load_state_dict(torch.load(str(self.checkpoint_dir / 'model_recommender.ckpt')))
        self.model_reasoner.load_state_dict(torch.load(str(self.checkpoint_dir / 'model_reasoner.ckpt')))
    
        #test
        self.model_anchor.eval()
        self.model_recommender.eval()
        self.model_reasoner.eval()

        results=[]
        doc_list = list(self.test_data.keys())
        start_list = list(range(0, len(doc_list), self.config['batch_size']))
        for start in start_list:
            end = start + self.config['batch_size']
            _, _, _, anchor_nodes, anchor_relations = self.model_anchor(doc_list[start:end])
            _, nodes_list = self.model_reasoner.get_anchor_graph_list(anchor_nodes, anchor_nodes[0].shape[0])
            _, relations_list = self.model_reasoner.get_anchor_graph_list(anchor_relations, anchor_nodes[0].shape[0])
            for i in range(len(nodes_list)):
                results.append({'news_id':doc_list[start+i], 'nodes':nodes_list[i], 'relations':relations_list[i]})

        with open('anchor_graph.json', 'w') as f:
            for item in results:
                line = json.dumps(item)+"\n"
                f.write(line)


