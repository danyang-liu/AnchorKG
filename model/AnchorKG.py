from base.base_model import BaseModel
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class Net(BaseModel):
    def __init__(self, config, entityid_dict, doc_feature_embedding, device=torch.device('cpu')):
        super(Net, self).__init__()

        self.device=device
        self.config = config
        self.doc_feature_embedding = doc_feature_embedding
        self.entityid_dict = entityid_dict

        self.actor_l1 = nn.Linear(self.config['embedding_size']*3, self.config['embedding_size'])
        self.actor_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.actor_l3 = nn.Linear(self.config['embedding_size'],1)

        #self.critic_l1 = nn.Linear(self.config['embedding_size']*3, self.config['embedding_size'])
        self.critic_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.critic_l3 = nn.Linear(self.config['embedding_size'], 1)

        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        #self.softmax = torch.nn.Softmax(dim=0)

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.doc_feature_embedding[newsid]).to(self.device))
        return torch.stack(news_embeddings)

    def forward(self, state_input, action_input):
        if len(state_input.shape) < len(action_input.shape):
            if len(action_input.shape) == 3:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], state_input.shape[2])
            else:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], action_input.shape[2], state_input.shape[3])

        # Actor
        actor_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = self.elu(self.actor_l2(actor_x))
        act_probs = self.sigmoid(self.actor_l3(actor_out))

        # Critic
        critic_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        q_actions = self.sigmoid(self.critic_l3(critic_out))

        return act_probs, q_actions

class AnchorKG(BaseModel):

    def __init__(self, config, doc_entity_dict, entity_doc_dict, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, hit_dict, entity_id_dict, neibor_embedding, neibor_num, device=torch.device('cpu')):
        super(AnchorKG, self).__init__()
        self.device=device
        self.config = config
        self.doc_entity_dict = doc_entity_dict
        self.entity_doc_dict = entity_doc_dict
        self.doc_feature_embedding = doc_feature_embedding
        self.hit_dict = hit_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.MAX_DEPTH = 3
        self.entity_id_dict = entity_id_dict
        self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        self.neibor_num = neibor_num.to(device)

        self.elu = nn.ELU()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_compress = nn.Sequential(
                                nn.Linear(self.config['doc_embedding_size'], self.config['embedding_size']),
                                nn.ELU(),
                                nn.Linear(self.config['embedding_size'], self.config['embedding_size']),
                                nn.Tanh()
                            ).to(device)
        self.entity_compress =  nn.Sequential(
                                    nn.Linear(100, self.config['embedding_size']),
                                    nn.Tanh(),
                                ).to(device)
        self.relation_compress = nn.Sequential(
                                    nn.Linear(100, self.config['embedding_size']),
                                    nn.Tanh(),
                                ).to(device)
        #self.innews_relation = nn.Embedding(1,self.config['embedding_size']).to(device)

        self.anchor_embedding_layer = nn.Linear(self.config['embedding_size'],self.config['embedding_size']).to(device)#todo *2 diyige
        self.anchor_weighs1_layer1 = nn.Linear(self.config['embedding_size'], self.config['embedding_size']).to(device)
        self.anchor_weighs1_layer2 = nn.Linear(self.config['embedding_size'], 1).to(device)

        self.policy_net = Net(self.config, self.entity_id_dict, self.doc_feature_embedding, device).to(device)
        #self.target_net = Net(self.config, self.entity_id_dict, self.doc_feature_embedding, device).to(device)

    def get_neiborhood_news_embedding_batch(self, news_embedding, entityids):#(batch,5,768),即每个entity的doc neiborhood embedding的均值
        neibor_news_embedding_avg = self.neibor_embedding(entityids.to('cpu')).to(self.device)
        neibor_num = []
        for i in range(len(entityids)):
            neibor_num.append(torch.index_select(self.neibor_num, 0, entityids[i]))
        neibor_num = torch.stack(neibor_num)
        if len(neibor_news_embedding_avg.shape) > len(news_embedding.shape):
            news_embedding = torch.unsqueeze(news_embedding, 1)
            neibor_num = torch.unsqueeze(neibor_num, 2)
            news_embedding = news_embedding.expand(news_embedding.shape[0] ,neibor_news_embedding_avg.shape[1] ,news_embedding.shape[2])
            neibor_num = neibor_num.expand(neibor_num.shape[0], neibor_num.shape[1],news_embedding.shape[2])
        neibor_news_embedding_avg = torch.div((neibor_news_embedding_avg - news_embedding), neibor_num)
        return neibor_news_embedding_avg

    def get_anchor_graph_list(self, anchor_graph_nodes, batch_size):
        anchor_graph_list = []
        for i in range(batch_size):
            anchor_graph_list.append([])
        for i in range(len(anchor_graph_nodes)):
            for j in range(len(anchor_graph_nodes[i])):
                anchor_graph_list[j].extend(list(map(lambda x:str(x), anchor_graph_nodes[i][j].data.cpu().numpy())))
        return anchor_graph_list

    def get_sim_reward_batch(self, news_embedding_batch, neibor_news_embedding_avg_batch):
        if len(neibor_news_embedding_avg_batch.shape) > len(news_embedding_batch.shape):
            news_embedding_batch = torch.unsqueeze(news_embedding_batch, 1)
            news_embedding_batch = news_embedding_batch.expand(news_embedding_batch.shape[0] ,neibor_news_embedding_avg_batch.shape[1] ,news_embedding_batch.shape[2])
        cos_rewards = self.cos(news_embedding_batch, neibor_news_embedding_avg_batch)
        return cos_rewards

    def get_hit_rewards_batch(self, newsid_batch, state_id_input_batch):
        hit_rewards = torch.zeros([len(newsid_batch), len(state_id_input_batch[0])], dtype=torch.float32)
        for i in range(len(newsid_batch)):
            for j in range(len(state_id_input_batch[i])):
                idx = state_id_input_batch[i][j].item()
                if idx in self.entity_doc_dict and newsid_batch[i] in self.hit_dict:
                    entity_neibor = set(self.entity_doc_dict[idx]).discard(newsid_batch[i])#当前entity出现过的doc
                    news_hit_neibor = self.hit_dict[newsid_batch[i]]#与当前doc相似的doc
                    if entity_neibor != None and len(entity_neibor & news_hit_neibor)>0:
                        hit_rewards[i][j] = 1.0

        return hit_rewards.to(self.device)

    def get_batch_rewards_step(self, hit_reward, sim_reward):
        reward =0.5*hit_reward + (1-0.5)*sim_reward
        return reward

    def get_reward(self, newsid, news_embedding, state_input):
        entity_value = state_input
        neibor_news_embedding_avg = self.get_neiborhood_news_embedding_batch(news_embedding, entity_value)
        sim_reward = self.get_sim_reward_batch(news_embedding, neibor_news_embedding_avg)
        hit_reward = self.get_hit_rewards_batch(newsid, entity_value)
        reward = self.get_batch_rewards_step(hit_reward, sim_reward)
        return reward

    def get_news_entities_batch(self, newsids):#当前news里包含的entity
        news_entities = torch.zeros(len(newsids), self.config['news_entity_num'], dtype=torch.long)
        news_relations = torch.zeros(len(newsids), self.config['news_entity_num'], dtype=torch.long)#专门用一个innews_relation去学未必效果好
        for i in range(len(newsids)):
            news_entities[i] = self.doc_entity_dict[newsids[i]]
            # news_relations.append([0 for k in range(self.config['news_entity_num'])])
        return news_entities, news_relations

    def get_news_embedding_batch(self, newsids):#(batch, 768)
        news_embeddings = torch.zeros([len(newsids), self.config['doc_embedding_size']])
        for i, newsid in enumerate(newsids):
            news_embeddings[i] = self.doc_feature_embedding[newsid]
        return news_embeddings.to(self.device)

    def get_next_action(self, state_id_input_batch):
        next_action_id = torch.zeros([len(state_id_input_batch), len(state_id_input_batch[0]), 20], dtype=torch.long)
        next_action_r_id = torch.zeros([len(state_id_input_batch), len(state_id_input_batch[0]), 20], dtype=torch.long)
        for i in range(len(state_id_input_batch)):
            for j in range(len(state_id_input_batch[i])):
                idx = state_id_input_batch[i][j].item()
                if idx in self.entity_adj:
                    next_action_id[i][j] = self.entity_adj[idx]
                    next_action_r_id[i][j] = self.relation_adj[idx]
        return next_action_id, next_action_r_id

    def get_advantage(self, curr_reward,  q_value):
        advantage = curr_reward - q_value
        return advantage

    def get_actor_loss(self, act_probs_step, advantage):
        actor_loss = -act_probs_step * advantage.detach()
        return actor_loss
        # actor_loss = act_probs_step - advantage
        # actor_loss = torch.pow(actor_loss, 2)

    def get_critic_loss(self, advantage):
        critic_loss = torch.pow(advantage, 2)
        return critic_loss

    def step_update(self, act_probs_step, q_values_step, step_reward, embeddding_reward, reasoning_reward,
                    alpha1=0.9, alpha2 = 0.1):
        embeddding_reward = torch.unsqueeze(embeddding_reward, dim=1)
        reasoning_reward = torch.unsqueeze(reasoning_reward, dim=1)
        embeddding_reward = embeddding_reward.expand(step_reward.shape[0], step_reward.shape[1])
        reasoning_reward = reasoning_reward.expand(step_reward.shape[0], step_reward.shape[1])
        curr_reward = alpha2 * step_reward + (1-alpha2)*(alpha1*embeddding_reward + (1-alpha1)*reasoning_reward)#本轮的total reward
        advantage = self.get_advantage(curr_reward, q_values_step) #curr_reward - q_values_step
        actor_loss = self.get_actor_loss(torch.log(act_probs_step), advantage)#self.get_actor_loss(act_probs_step, q_values_step)#self.get_actor_loss(torch.log(act_probs_step), advantage) # -act_probs_step * advantage #problem2
        critic_loss = advantage.pow(2)#self.get_critic_loss(advantage) #advantage.pow(2)

        return critic_loss, actor_loss

    def get_anchor_nodes(self, weights, q_values, action_id_input, relation_id_input, topk):
        if len(weights.shape) <= 3:
            weights =torch.unsqueeze(weights, 1)
            q_values = torch.unsqueeze(q_values, 1)
            action_id_input = torch.unsqueeze(action_id_input, 1)
            relation_id_input = torch.unsqueeze(relation_id_input, 1)

        weights = weights.squeeze(-1)
        q_values = q_values.squeeze(-1)
        m = Categorical(weights)
        acts_idx = m.sample(sample_shape=torch.Size([topk]))
        acts_idx = acts_idx.permute(1,2,0)
        shape0 = acts_idx.shape[0]
        shape1 = acts_idx.shape[1]
        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2])#(batch,topk)

        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2])
        q_values = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1], action_id_input.shape[2])
        relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1], relation_id_input.shape[2])
        state_id_input_value = action_id_input.gather(1, acts_idx)#被选中的entity的id,(batch,topk)
        relation_id_selected = relation_id_input.gather(1, acts_idx)#被选中的relation的id,(batch,topk)
        weights = weights.gather(1, acts_idx)
        q_values = q_values.gather(1, acts_idx)
        weights = weights.reshape(shape0, shape1 *  weights.shape[1])#被选中的(r,e)对应的概率值，(batch,topk)
        q_values = q_values.reshape(shape0, shape1 * q_values.shape[1])#被选中的(r,e)动作对应的价值函数估计(batch,topk)
        state_id_input_value = state_id_input_value.reshape(shape0, shape1 *  state_id_input_value.shape[1])
        relation_id_selected = relation_id_selected.reshape(shape0, shape1 *  relation_id_selected.shape[1])
        return weights, q_values, state_id_input_value, relation_id_selected

    def get_news_embedding_input(self, entity_ids, news_embeddings):
        entity_ids_index = entity_ids._indices()[0]
        news_embedding_batch = news_embeddings(entity_ids_index)
        return news_embedding_batch

    def get_state_input(self, news_embedding, depth, anchor_graph, history_entity_1, history_relation_1):
        if depth == 0:
            state_embedding = torch.cat([news_embedding, torch.zeros([news_embedding.shape[0], 128], dtype=torch.float32).to(self.device)], dim=-1)
        else:
            history_entity_embedding = self.entity_compress(self.entity_embedding(history_entity_1.to('cpu')).to(self.device))
            history_relation_embedding = self.relation_compress(self.relation_embedding(history_relation_1.to('cpu')).to(self.device))
            state_embedding_new = history_relation_embedding + history_entity_embedding
            state_embedding_new = torch.mean(state_embedding_new, dim=1, keepdim=False)
            state_embedding = torch.cat([news_embedding, state_embedding_new], dim=-1)
        return state_embedding

    def get_neighbors(self, entities):
        neighbor_entities = []
        neighbor_relations = []
        for entity_batch in entities:
            neighbor_entities.append([])
            neighbor_relations.append([])
            for entity in entity_batch:
                if type(entity) == int:
                    neighbor_entities[-1].append(self.entity_adj[entity])
                    neighbor_relations[-1].append(self.adj_relation[entity])
                else:
                    neighbor_entities[-1].append([])
                    neighbor_relations[-1].append([])
                    for entity_i in entity:
                        neighbor_entities[-1][-1].append(self.entity_adj[entity_i])
                        neighbor_relations[-1][-1].append(self.adj_relation[entity_i])

        return torch.LongTensor(neighbor_entities), torch.LongTensor(neighbor_relations)

    def get_anchor_embedding(self, anchor_graph):
        anchor_graph_nodes = []
        for i in range(len(anchor_graph)):
            for j in range(len(anchor_graph[i])):
                anchor_graph_nodes.append(anchor_graph[i][j])
        anchor_graph_nodes = torch.tensor(anchor_graph_nodes)
        neibor_entities, neibor_relations = self.get_neighbors(anchor_graph_nodes)
        neibor_entities_embedding = self.entity_compress(self.entity_embedding(neibor_entities).to(self.device))
        neibor_relations_embedding = self.relation_compress(self.relation_embedding(neibor_relations).to(self.device))
        anchor_embedding = torch.cat([anchor_graph_nodes, torch.sum(neibor_entities_embedding+neibor_relations_embedding, dim=-2)])
        anchor_embedding = self.tanh(self.anchor_embedding_layer(anchor_embedding))
        anchor_embedding_weight = self.softmax(self.anchor_weighs1_layer2(self.elu(self.anchor_weighs1_layer1(anchor_embedding))))
        anchor_embedding = torch.sum(anchor_embedding * anchor_embedding_weight, dim=-2) #todo -2
        return anchor_embedding

    def predict_anchor(self, newsid, news_feature):

        self.doc_entity_dict[newsid] = news_feature[0]
        self.doc_feature_embedding = news_feature[1]
        prediction = self.forward([newsid], [newsid]).cpu().data.numpy()
        anchor_nodes = prediction[6]
        anchor_relation = prediction[8]
        return anchor_nodes, anchor_relation

    def forward(self, news1):
        #news1
        depth = 0
        history_entity_1 = []
        history_relation_1 = []

        anchor_graph1 = []
        anchor_relation1 = []
        act_probs_steps1 = []#策略网络确定的动作概率
        step_rewards1 = []#每一步的即时reward
        q_values_steps1 = []#价值网络确定的动作价值

        news_embedding_origin = self.get_news_embedding_batch(news1)#(batch, 768)
        news_embedding = self.news_compress(news_embedding_origin)#(batch, 128)
        
        action_id, relation_id = self.get_news_entities_batch(news1)#当前news里包含的实体和关系,这里关系恒取id=0, cpu
        action_embedding = self.entity_compress(self.entity_embedding(action_id).to(self.device))#(batch, 20, 128)
        relation_embedding = self.relation_compress(self.relation_embedding(relation_id).to(self.device))
        action_embedding = action_embedding + relation_embedding#动作集合的表征,(batch, 20, 128)
        
        state_input = self.get_state_input(news_embedding, depth, anchor_graph1, history_entity_1, history_relation_1)#(batch,256)

        while (depth < self.MAX_DEPTH):
            act_probs, q_values = self.policy_net(state_input, action_embedding)#output: (batch, 20, 1), (batch, 20, 1)
            topk = self.config['topk'][depth]
            anchor_act_probs, anchor_q_values, anchor_nodes, anchor_relations = self.get_anchor_nodes(act_probs, q_values, action_id.to(self.device), relation_id.to(self.device), topk)#做动作
            history_entity_1 = anchor_nodes#anchor_nodes即最新加入的实体
            history_relation_1 = anchor_relations
            depth = depth + 1
            state_input = self.get_state_input(news_embedding, depth, anchor_graph1, history_entity_1, history_relation_1)#下一个状态

            act_probs_steps1.append(anchor_act_probs)
            q_values_steps1.append(anchor_q_values)
            actionid_lookup, action_rid_lookup = self.get_next_action(anchor_nodes)#下一步可以扩展的(r,e)集合, (batch, 5, 20) / (batch, 5*3, 20)
            action_id = actionid_lookup
            relation_id = action_rid_lookup
            action_embedding = self.entity_compress(self.entity_embedding(action_id).to(self.device)) + self.relation_compress(self.relation_embedding(relation_id).to(self.device))

            anchor_graph1.append(anchor_nodes)#gpu
            anchor_relation1.append(anchor_relations)
            step_reward = self.get_reward(news1, news_embedding_origin, anchor_nodes)
            step_rewards1.append(step_reward)

        return act_probs_steps1, q_values_steps1, step_rewards1, anchor_graph1, anchor_relation1 #仅act_probs_steps1, q_values_steps1有有效梯度

    def warm_train(self, batch):
        loss_fn = nn.BCELoss()
        news_embedding = self.get_news_embedding_batch(batch['item1'])
        news_embedding = self.news_compress(news_embedding)
        path_node_embeddings = self.entity_compress(self.entity_embedding(batch['paths']).to(self.device))#(batch, depth, embedding_size)
        path_edge_embeddings = self.relation_compress(self.relation_embedding(batch['edges']).to(self.device))#(batch, depth, embedding_size)
        path_embeddings = path_node_embeddings + path_edge_embeddings#(batch, depth, 128)

        batch_act_probs=[]
        history_entity=[]
        history_relation=[]
        for i in range(self.MAX_DEPTH):
            state_input = self.get_state_input(news_embedding, i, [], history_entity, history_relation)#(batch,256)
            act_probs, _ = self.policy_net(state_input, path_embeddings[:,i,:])#(batch, 1)
            batch_act_probs.append(act_probs)
            history_entity = batch['paths'][:, i:i+1]#此处遵从forward里的实现，即只用最新加入的节点来进行下一个state_input的计算
            history_relation = batch['edges'][:, i:i+1]
        
        batch_act_probs = torch.cat(batch_act_probs, dim=1)#(batch, depth)
        indices = batch['label']>=-0.5
        labels = batch['label'][indices].to(self.device)
        predicts = batch_act_probs[indices.to(self.device)]
        loss = loss_fn(predicts, labels)

        return loss, predicts, labels
        

