import numpy as np
from numpy import inf
import torch
from torch import nn, optim
from utils.metrics import *
from utils.pytorchtools import *
from base.base_trainer import BaseTrainer
from logger.logger import *
from model.AnchorKG import *

def get_anchor_graph_data(doc_feature_file):
    print('constructing anchor doc ...')
    test_data = {}
    item1 = []
    item2 = []
    label = []
    fp_news_entities = open(doc_feature_file, 'r', encoding='utf-8')
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

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, model_anchor, model_recommender, model_reasoner, criterion, optimizer_anchor, optimizer_recommender, optimizer_reasoner, device,
                 data):
        super().__init__()

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model_anchor = model_anchor
        self.model_recommender = model_recommender
        self.model_reasoner = model_reasoner
        self.criterion = criterion
        self.optimizer_anchor = optimizer_anchor
        self.optimizer_recommender = optimizer_recommender
        self.optimizer_reasoner = optimizer_reasoner

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.device = device

        self.warmup_train_dataloader = data[0]
        self.warmup_test_data = data[1]
        self.train_dataloader = data[2]
        self.val_data = data[3]


        self.entity_id_dict = data[4]
        self.doc_feature_embedding = data[5]
        self.entity_embedding = data[6]


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
        actor_all_loss = 0
        critic_all_loss = 0
        embedding_all_loss = 0
        reasoning_all_loss = 0
        for step, batch in enumerate(self.train_dataloader):
            if step % 1000 == 0:
                print(step)
            act_probs_steps1, q_values_steps1, act_probs_steps2, q_values_steps2, step_rewards1, step_rewards2, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2 = self.model_anchor(
                    batch['item1'], batch['item2'])
            embedding_predict = \
            self.model_recommender(batch['item1'], batch['item2'], anchor_graph1, anchor_graph2)[0]#item1和item2的相似度衡量值，归一到[0,1]
            reasoning_predict = self.model_reasoner(batch['item1'], batch['item2'], anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2)[0]

            embedding_loss = self.criterion(embedding_predict, batch['label'].to(self.device).float())
            reasoning_loss = self.criterion(reasoning_predict, batch['label'].to(self.device).float())
            reasoning_loss = reasoning_loss.requires_grad_()#为何默认没有梯度

            embedding_loss_mean = torch.mean(embedding_loss)
            reasoning_loss_mean = torch.mean(reasoning_loss)

            embedding_all_loss = embedding_all_loss + embedding_loss_mean.data
            reasoning_all_loss = reasoning_all_loss + reasoning_loss_mean.data


            self.optimizer_recommender.zero_grad()
            embedding_loss_mean.backward(retain_graph=True)
            self.optimizer_recommender.step()

            self.optimizer_reasoner.zero_grad()
            reasoning_loss_mean.backward(retain_graph=True)
            self.optimizer_reasoner.step()

            batch_rewards1 = step_rewards1
            num_steps1 = len(batch_rewards1)
            if len(self.config['model']['topk']) == 3:
                batch_rewards1[1] = torch.reshape(batch_rewards1[1],
                                                      (batch_rewards1[1].shape[0], self.config['model']['topk'][0],
                                                       self.config['model']['topk'][1]))
                batch_rewards1[2] = torch.reshape(batch_rewards1[2], (
                        batch_rewards1[2].shape[0], self.config['model']['topk'][0], self.config['model']['topk'][1],
                        self.config['model']['topk'][2]))
            else:
                print("error, layer num not match")
            for i in range(1, num_steps1):#本轮生成的图算是一次采样，基于此次采样求总的intermediate reward
                batch_rewards1[num_steps1 - i - 1] = batch_rewards1[num_steps1 - i - 1] + self.config['model'][
                                                                                                  'gamma'] * torch.mean(
                    batch_rewards1[num_steps1 - i], dim=-1)
            if len(self.config['model']['topk']) == 3:
                batch_rewards1[1] = torch.reshape(batch_rewards1[1],
                                                      (batch_rewards1[1].shape[0],
                                                       self.config['model']['topk'][0] * self.config['model']['topk'][
                                                           1]))
                batch_rewards1[2] = torch.reshape(batch_rewards1[2], (
                        batch_rewards1[2].shape[0], self.config['model']['topk'][0] * self.config['model']['topk'][1] *
                        self.config['model']['topk'][2]))
            else:
                print("error, layer num not match")

            batch_rewards2 = step_rewards2
            num_steps2 = len(batch_rewards2)
            if len(self.config['model']['topk']) == 3:
                batch_rewards2[1] = torch.reshape(batch_rewards2[1],
                                                      (batch_rewards2[1].shape[0], self.config['model']['topk'][0],
                                                       self.config['model']['topk'][1]))
                batch_rewards2[2] = torch.reshape(batch_rewards2[2], (
                        batch_rewards2[2].shape[0], self.config['model']['topk'][0], self.config['model']['topk'][1],
                        self.config['model']['topk'][2]))
            else:
                print("error, layer num not match")
            for i in range(1, num_steps2):
                batch_rewards2[num_steps2 - i - 1] = batch_rewards2[num_steps2 - i - 1] + self.config['model'][
                                                                                                  'gamma'] * torch.mean(
                        batch_rewards2[num_steps2 - i], dim=-1)
            if len(self.config['model']['topk']) == 3:
                batch_rewards2[1] = torch.reshape(batch_rewards2[1],
                                                      (batch_rewards2[1].shape[0],
                                                       self.config['model']['topk'][0] * self.config['model']['topk'][
                                                           1]))
                batch_rewards2[2] = torch.reshape(batch_rewards2[2], (
                        batch_rewards2[2].shape[0], self.config['model']['topk'][0] * self.config['model']['topk'][1] *
                        self.config['model']['topk'][2]))
            else:
                print("error, layer num not match")

            all_loss_list = []
            actor_loss_list = []
            critic_loss_list = []
            news1_actor_loss = []
            news1_critic_loss = []
            for i in range(num_steps1):
                batch_reward1 = batch_rewards1[i]
                q_values_step1 = q_values_steps1[i]
                act_probs_step1 = act_probs_steps1[i]
                critic_loss1, actor_loss1 = self.model_anchor.step_update(act_probs_step1, q_values_step1,
                                                                              batch_reward1,
                                                                              1 - embedding_loss.detach(),
                                                                              1 - reasoning_loss.detach(),
                                                                              self.config['model']['alpha1'],
                                                                              self.config['model']['alpha2'])#计算actor和critic的loss
                news1_actor_loss.append(actor_loss1.mean())
                news1_critic_loss.append(critic_loss1.mean())
                actor_loss_list.append(actor_loss1.mean())
                critic_loss_list.append(critic_loss1.mean())
                all_loss_list.append(actor_loss1.mean())
                all_loss_list.append(critic_loss1.mean())
            news2_actor_loss = []
            news2_critic_loss = []
            for i in range(num_steps2):
                batch_reward2 = batch_rewards2[i]
                q_values_step2 = q_values_steps2[i]
                act_probs_step2 = act_probs_steps2[i]
                critic_loss2, actor_loss2 = self.model_anchor.step_update(act_probs_step2, q_values_step2,
                                                                              batch_reward2,
                                                                              1 - embedding_loss.detach(),
                                                                              1 - reasoning_loss.detach(),
                                                                              self.config['model']['alpha1'],
                                                                              self.config['model']['alpha2'])
                news2_actor_loss.append(actor_loss2.mean())
                news2_critic_loss.append(critic_loss2.mean())
                actor_loss_list.append(actor_loss2.mean())
                critic_loss_list.append(critic_loss2.mean())
                all_loss_list.append(actor_loss2.mean())
                all_loss_list.append(critic_loss2.mean())

            self.optimizer_anchor.zero_grad()
            if all_loss_list != []:
                loss = torch.stack(all_loss_list).sum()  # sum up all the loss
                loss.backward()
                self.optimizer_anchor.step()
                anchor_all_loss = anchor_all_loss + loss.data

            torch.cuda.empty_cache()

            break

        torch.save(self.model_anchor.state_dict(), './out/saved/models/AnchorKG/checkpoint_anchor.pt')
        torch.save(self.model_recommender.state_dict(),
                   './out/saved/models/AnchorKG/checkpoint_recommender.pt')
        torch.save(self.model_reasoner.state_dict(),
                   './out/saved/models/AnchorKG/checkpoint_reasoner.pt')

        print("anchor all loss: " + str(anchor_all_loss))
        print("embedding all loss: " + str(embedding_all_loss))
        print("reasoning all loss: " + str(reasoning_all_loss))

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
        start_list = list(range(0, len(self.val_data['label']), self.config['data_loader']['batch_size']))
        for start in start_list:
            if start + self.config['data_loader']['batch_size'] <= len(self.val_data['label']):
                end = start + self.config['data_loader']['batch_size']
                act_probs_steps1, q_values_steps1, act_probs_steps2, q_values_steps2, step_rewards1, step_rewards2, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2 = self.model_anchor(
                    self.val_data['item1'][start:end], self.val_data['item2'][start:end])
                embedding_predict = self.model_recommender(self.val_data['item1'][start:end], self.val_data['item2'][start:end], anchor_graph1, anchor_graph2)[0]
                reasoning_predict = self.model_reasoner(self.val_data['item1'][start:end], self.val_data['item2'][start:end], anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2)[0]
                predict = 0.5*embedding_predict + 0.5*reasoning_predict
                y_pred.extend(predict.cpu().data.numpy())
            else:
                act_probs_steps1, q_values_steps1, act_probs_steps2, q_values_steps2, step_rewards1, step_rewards2, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2 = self.model_anchor(
                    self.val_data['item1'][start:], self.val_data['item2'][start:])
                embedding_predict = self.model_recommender(self.val_data['item1'][start:], self.val_data['item2'][start:], anchor_graph1, anchor_graph2)[0]
                reasoning_predict = self.model_reasoner(self.val_data['item1'][start:], self.val_data['item2'][start:], anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2)[0]
                predict = 0.5 * embedding_predict + 0.5 * reasoning_predict
                y_pred.extend(predict.cpu().data.numpy())

        truth = self.val_data['label']
        auc_score = cal_auc(truth, y_pred)
        print("auc socre: "+str(auc_score))
        return auc_score

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state_anchor = self.model_anchor.state_dict()
        state_recommender = self.model_recommender.state_dict()
        state_reasoner = self.model_reasoner.state_dict()
        filename_anchor = str(self.checkpoint_dir / 'checkpoint-anchor-epoch{}.pth'.format(epoch))
        torch.save(state_anchor, filename_anchor)
        self.logger.info("Saving checkpoint: {} ...".format(filename_anchor))
        filename_recommender = str(self.checkpoint_dir / 'checkpoint-recommender-epoch{}.pth'.format(epoch))
        torch.save(state_recommender, filename_recommender)
        self.logger.info("Saving checkpoint: {} ...".format(filename_recommender))
        filename_reasoner = str(self.checkpoint_dir / 'checkpoint-reasoner-epoch{}.pth'.format(epoch))
        torch.save(state_reasoner, filename_reasoner)
        self.logger.info("Saving checkpoint: {} ...".format(filename_reasoner))


    def train(self):
        """
            Full training logic
        """
        logger_train = get_logger("train")

        # warm up training stage
        if self.config['trainer']['warm_up']:
            logger_train.info("warm up training")
            warmup_model = Net(self.config, self.entity_id_dict, self.doc_feature_embedding, self.entity_embedding, self.device).to(self.device)
            criterion = nn.BCELoss()
            optimizer_warmup = optim.Adam(warmup_model.parameters(), lr=self.config['optimizer']['lr'],
                                          weight_decay=self.config['optimizer']['weight_decay'])
            for epoch in range(10):
                warmup_model.train()
                warmup_all_loss = 0
                for step, batch in enumerate(self.warmup_train_dataloader):
                    predict_value = warmup_model(batch['item2'], batch['item1'])[0]
                    warmup_loss = criterion(predict_value, batch['label'].to(self.device).float())
                    warmup_all_loss = warmup_all_loss + warmup_loss
                    optimizer_warmup.zero_grad()
                    warmup_loss.backward()
                    optimizer_warmup.step()

                warmup_model.eval()
                y_pred = []
                start_list = list(range(0, len(self.warmup_test_data['label']), self.config['data_loader']['batch_size']))
                for start in start_list:
                    if start + self.config['data_loader']['batch_size'] <= len(self.warmup_test_data['label']):
                        end = start + self.config['data_loader']['batch_size']
                        predict = warmup_model(
                            self.warmup_test_data['item2'][start:end], self.warmup_test_data['item1'][start:end])[0]
                        y_pred.extend(predict.cpu().data.numpy())
                    else:
                        predict = warmup_model(
                            self.warmup_test_data['item2'][start:], self.warmup_test_data['item1'][start:])[0]
                        y_pred.extend(predict.cpu().data.numpy())

                truth = self.warmup_test_data['label']
                auc_score = cal_auc(truth, y_pred)
                logger_train.info('Warmup epoch:%d AUC:%.4f' % (epoch, auc_score))
            torch.save(warmup_model.state_dict(), self.config['savepath'] + "/warmup_model")

        logger_train.info("anchor graph training")
        valid_scores = []
        early_stopping = EarlyStopping(patience=self.config['trainer']['early_stop'], verbose=True)
        for epoch in range(self.start_epoch, self.epochs+1):
            self._train_epoch(epoch)

            valid_socre = self._valid_epoch(epoch)
            valid_scores.append(valid_socre)

            early_stopping(valid_socre, self.model_anchor)
            if early_stopping.early_stop:
                logger_train.info("Early stopping")

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            predict_anchor_graph = False
            if predict_anchor_graph:
                anchor_graph_nodes = []
                anchor_Data = get_anchor_graph_data(self.config['data']['datapath']+self.config['data']['doc_feature_entity_file'])
                start_list = list(range(0, len(anchor_Data['label']), self.config['data_loader']['batch_size']))
                for start in start_list:
                    if start + self.config['data_loader']['batch_size'] <= len(anchor_Data['label']):
                        end = start + self.config['data_loader']['batch_size']

                        anchor_graph_nodes.extend(self.model_anchor.get_anchor_graph_list(
                            self.model_anchor(anchor_Data['item1'][start:end], anchor_Data['item2'][start:end])[6],
                            len(anchor_Data['item1'][start:end])))
                    else:
                        anchor_graph_nodes.extend(self.model_anchor.get_anchor_graph_list(
                            self.model_anchor(anchor_Data['item1'][start:], anchor_Data['item2'][start:])[6],
                            len(anchor_Data['item1'][start:])))

                fp_anchor_file = open("./out/anchor_file_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                for i in range(len(anchor_Data['item1'])):
                    fp_anchor_file.write(
                        anchor_Data['item1'][i] + '\t' + ' '.join(list(set(anchor_graph_nodes[i]))) + '\n')
                fp_anchor_file = open("./out/anchor_file2_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                for i in range(len(anchor_Data['item1'])):
                    fp_anchor_file.write(anchor_Data['item1'][i] + '\t' + ' '.join(list(anchor_graph_nodes[i])) + '\n')




