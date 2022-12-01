from model.base_model import BaseModel
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class Net(BaseModel):
    def __init__(self, config, entityid_dict, doc_feature_embedding):
        super(Net, self).__init__()
        self.config = config
        self.doc_feature_embedding = doc_feature_embedding
        self.entityid_dict = entityid_dict

        self.actor_l1 = nn.Linear(self.config['embedding_size']*4, self.config['embedding_size'])
        self.actor_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.actor_l3 = nn.Linear(self.config['embedding_size'],1)

        #self.critic_l1 = nn.Linear(self.config['embedding_size']*3, self.config['embedding_size'])
        self.critic_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.critic_l3 = nn.Linear(self.config['embedding_size'], 1)

        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-2)

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
        act_probs = self.softmax(self.actor_l3(actor_out))#out: (batch, 20, 1),(batch, 5, 20, 1),(batch, 15, 20, 1)

        # Critic
        critic_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        values = self.critic_l3(critic_out).mean(dim=-2)#out: (batch,1), (batch,5,1), (batch,15,1)

        return act_probs, values

class AnchorKG(BaseModel):

    def __init__(self, config, doc_entity_dict, entity_doc_dict, doc_feature_embedding, entity_adj, relation_adj, hit_dict, entity_id_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, device=torch.device('cpu')):
        super(AnchorKG, self).__init__()
        self.device=device
        self.config = config
        self.doc_entity_dict = doc_entity_dict
        self.entity_doc_dict = entity_doc_dict
        self.doc_feature_embedding = doc_feature_embedding
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.hit_dict = hit_dict
        self.entity_id_dict = entity_id_dict
        self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        self.neibor_num = neibor_num.to(device)

        self.MAX_DEPTH = 3
        self.cos = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-2)

        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_compress = nn.Sequential(
                                nn.Linear(self.config['doc_embedding_size'], self.config['embedding_size']),
                                nn.ELU(),
                                nn.Linear(self.config['embedding_size'], self.config['embedding_size']),
                                nn.Tanh()
                            ).to(device)
        self.entity_compress =  nn.Sequential(
                                    nn.Linear(self.config['entity_embedding_size'], self.config['embedding_size'], bias=False),
                                    nn.Tanh(),
                                ).to(device)
        self.relation_compress = nn.Sequential(
                                    nn.Linear(self.config['entity_embedding_size'], self.config['embedding_size'], bias=False),
                                    nn.Tanh(),
                                ).to(device)
        self.innews_relation = nn.Embedding(1, self.config['embedding_size']).to(device)
        self.anchor_embedding_layer = nn.Sequential(
                                        nn.Linear(self.config['embedding_size']*2, self.config['embedding_size'], bias=False),
                                        nn.Tanh(),
                                    ).to(device)
        self.anchor_layer = nn.Sequential(
                                nn.Linear(self.config['embedding_size'], self.config['embedding_size'], bias=False),
                                nn.ELU(),
                                nn.Linear(self.config['embedding_size'], 1, bias=False),
                            ).to(device)

        self.policy_net = Net(self.config, self.entity_id_dict, self.doc_feature_embedding).to(device)

    def get_news_embedding_batch(self, newsids):#(batch, 768)
        news_embeddings = torch.zeros([len(newsids), self.config['doc_embedding_size']])
        for i, newsid in enumerate(newsids):
            news_embeddings[i] = self.doc_feature_embedding[newsid]
        return news_embeddings.to(self.device)
    
    def get_news_entities_batch(self, newsids):#entity contained in current news
        news_entities = torch.zeros(len(newsids), self.config['news_entity_num'], dtype=torch.long)
        news_relations = torch.zeros(len(newsids), self.config['news_entity_num'], dtype=torch.long)
        for i in range(len(newsids)):
            news_entities[i] = self.doc_entity_dict[newsids[i]]
        return news_entities, news_relations
    
    def get_state_input(self, news_embedding, depth, anchor_graph, history_entity, history_relation):
        if depth == 0:
            state_embedding = torch.cat([news_embedding, torch.zeros([news_embedding.shape[0], 128*2], dtype=torch.float32).to(self.device)], dim=-1)
        else:
            history_entity_embedding = self.entity_compress(self.entity_embedding(history_entity.to(self.entity_embedding.weight.device)).to(self.device))
            history_relation_embedding = self.relation_compress(self.relation_embedding(history_relation.to(self.relation_embedding.weight.device)).to(self.device)) if depth>1 else self.innews_relation(history_relation)
            state_embedding_new = history_relation_embedding + history_entity_embedding
            state_embedding_new = torch.mean(state_embedding_new, dim=1, keepdim=False)
            anchor_embedding = self.get_anchor_graph_embedding(anchor_graph)
            state_embedding = torch.cat([news_embedding, anchor_embedding, state_embedding_new], dim=-1)
        return state_embedding
    
    def get_anchor_graph_embedding(self, anchor_graph):
        anchor_graph_nodes  = torch.cat(anchor_graph, dim=-1).to(self.entity_adj.device)
        anchor_graph_nodes_embedding = self.entity_compress(self.entity_embedding(anchor_graph_nodes).to(self.device))
        neibor_entities, neibor_relations = self.get_neighbors(anchor_graph_nodes)#first-order neighbors for each entity
        neibor_entities_embedding = self.entity_compress(self.entity_embedding(neibor_entities).to(self.device))
        neibor_relations_embedding = self.relation_compress(self.relation_embedding(neibor_relations).to(self.device))
        anchor_embedding = torch.cat([anchor_graph_nodes_embedding, torch.sum(neibor_entities_embedding+neibor_relations_embedding, dim=-2)], dim=-1)

        anchor_embedding = self.anchor_embedding_layer(anchor_embedding)#(batch, 50, 128)
        anchor_embedding_weight = self.softmax(self.anchor_layer(anchor_embedding))#(batch, 50, 1)
        anchor_embedding = torch.sum(anchor_embedding * anchor_embedding_weight, dim=-2)
        return anchor_embedding
    
    def get_neighbors(self, entities):
        neighbor_entities = self.entity_adj[entities]
        neighbor_relations = self.relation_adj[entities]
        return neighbor_entities, neighbor_relations

    def get_anchor_nodes(self, weights, action_id_input, relation_id_input, topk):
        if len(weights.shape) <= 3:
            weights =torch.unsqueeze(weights, 1)
            action_id_input = torch.unsqueeze(action_id_input, 1)
            relation_id_input = torch.unsqueeze(relation_id_input, 1)

        weights = weights.squeeze(-1)
        m = Categorical(weights)
        acts_idx = m.sample(sample_shape=torch.Size([topk]))#存在同一位置的重复采样
        acts_idx = acts_idx.permute(1,2,0)
        shape0 = acts_idx.shape[0]
        shape1 = acts_idx.shape[1]
        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2])#(batch,topk)

        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2])
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1], action_id_input.shape[2])
        relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1], relation_id_input.shape[2])

        weights = weights.gather(1, acts_idx)
        state_id_input_value = action_id_input.gather(1, acts_idx)#selected entity id,(batch,topk)
        relation_id_selected = relation_id_input.gather(1, acts_idx)#selected relation id,(batch,topk)
        
        weights = weights.reshape(shape0, shape1 *  weights.shape[1])#probility for selected (r,e) ,(batch,5), (batch, 15) , (batch, 30)
        state_id_input_value = state_id_input_value.reshape(shape0, shape1 *  state_id_input_value.shape[1])
        relation_id_selected = relation_id_selected.reshape(shape0, shape1 *  relation_id_selected.shape[1])
        return weights, state_id_input_value, relation_id_selected

    def get_reward(self, newsid, news_embedding, anchor_nodes):
        neibor_news_embedding_avg = self.get_neiborhood_news_embedding_batch(news_embedding, anchor_nodes)#如果没有其他news呢？
        sim_reward = self.get_sim_reward_batch(news_embedding, neibor_news_embedding_avg)
        hit_reward = self.get_hit_rewards_batch(newsid, anchor_nodes)
        reward = 0.5*hit_reward + (1-0.5)*sim_reward
        return reward
    
    def get_neiborhood_news_embedding_batch(self, news_embedding, anchor_nodes):#(batch,5,768), doc neiborhood embedding avg for each entity
        neibor_news_embedding_avg = self.neibor_embedding(anchor_nodes.to(self.neibor_embedding.weight.device)).to(self.device)
        neibor_num = self.neibor_num[anchor_nodes]
        neibor_news_embedding_avg = torch.div((neibor_news_embedding_avg - news_embedding[:,None,:]), neibor_num[:,:,None])
        return neibor_news_embedding_avg # (batch, 5/15/30, 768)

    def get_sim_reward_batch(self, news_embedding_batch, neibor_news_embedding_avg_batch):
        cos_rewards = self.cos(news_embedding_batch[:,None,:], neibor_news_embedding_avg_batch)
        return cos_rewards #(batch, 5/15/30)
    
    def get_hit_rewards_batch(self, newsid_batch, anchor_nodes):
        hit_rewards = torch.zeros([len(newsid_batch), len(anchor_nodes[0])], dtype=torch.float32)
        for i in range(len(newsid_batch)):
            for j in range(len(anchor_nodes[i])):
                idx = anchor_nodes[i][j].item()
                if idx in self.entity_doc_dict and newsid_batch[i] in self.hit_dict:
                    entity_neibor = set(self.entity_doc_dict[idx])
                    entity_neibor.discard(newsid_batch[i])#entity neiborhood news
                    news_hit_neibor = self.hit_dict[newsid_batch[i]]#similarity doc
                    if len(entity_neibor & news_hit_neibor)>0:
                        hit_rewards[i][j] = 1.0
        return hit_rewards.to(self.device)

    def forward(self, news):
        depth = 0
        history_entity = []
        history_relation = []

        anchor_graph = []
        anchor_relation = []
        act_probs_steps = []#action probility of policy net 
        state_values_steps = []#state_values of target net
        rewards_steps = []#immediate reward

        news_embedding_origin = self.get_news_embedding_batch(news)#(batch, 768)
        news_embedding = self.news_compress(news_embedding_origin)#(batch, 128)
        
        action_id, relation_id = self.get_news_entities_batch(news)#entities and relations in current news, relation id==0, cpu
        action_embedding = self.entity_compress(self.entity_embedding(action_id).to(self.device))#(batch, 20, 128)
        relation_embedding = self.innews_relation(relation_id.to(self.device))        #self.relation_compress(self.relation_embedding(relation_id).to(self.device))
        action_embedding = action_embedding + relation_embedding#(batch, 20, 128)
        
        state_input = self.get_state_input(news_embedding, depth, anchor_graph, history_entity, history_relation)#(batch,256)

        while (depth < self.MAX_DEPTH):
            act_probs, state_values = self.policy_net(state_input, action_embedding)
            topk = self.config['topk'][depth]
            anchor_act_probs, anchor_nodes, anchor_relations = self.get_anchor_nodes(act_probs, action_id.to(self.device), relation_id.to(self.device), topk)#take action
            
            history_entity = anchor_nodes#newly adds entities
            history_relation = anchor_relations
            depth = depth + 1
            act_probs_steps.append(anchor_act_probs)
            state_values_steps.append(state_values)
            anchor_graph.append(anchor_nodes)#gpu
            anchor_relation.append(anchor_relations)

            state_input = self.get_state_input(news_embedding, depth, anchor_graph, history_entity, history_relation)#next state

            action_id, relation_id = self.get_neighbors(anchor_nodes)#(r,e) space for next action, (batch, 5, 20) / (batch, 5*3, 20)
            action_embedding = self.entity_compress(self.entity_embedding(action_id).to(self.device)) + self.relation_compress(self.relation_embedding(relation_id).to(self.device))

            step_reward = self.get_reward(news, news_embedding_origin, anchor_nodes)
            rewards_steps.append(step_reward)

        return act_probs_steps, state_values_steps, rewards_steps, anchor_graph, anchor_relation #only act_probs_steps, state_values_steps have gradients

    def warm_train(self, batch):
        loss_fn = nn.BCELoss()
        news_embedding = self.get_news_embedding_batch(batch['item1'])
        news_embedding = self.news_compress(news_embedding)
        path_node_embeddings = self.entity_compress(self.entity_embedding(batch['paths']).to(self.device))#(batch, depth, embedding_size)
        path_edge_embeddings = self.relation_compress(self.relation_embedding(batch['edges'][:,1:]).to(self.device))#(batch, depth-1, embedding_size)
        innews_relation_embedding = self.innews_relation(torch.zeros([batch['paths'].shape[0],1], dtype=torch.long).to(self.device))#(batch, 1, embedding_size)
        path_edge_embeddings = torch.cat([innews_relation_embedding, path_edge_embeddings], dim=1)
        path_embeddings = path_node_embeddings + path_edge_embeddings#(batch, depth, 128)

        batch_act_probs=[]
        anchor_graph = []
        history_entity=[]
        history_relation=[]
        for i in range(self.MAX_DEPTH):
            state_input = self.get_state_input(news_embedding, i, anchor_graph, history_entity, history_relation)#(batch,256)
            act_probs, _ = self.policy_net(state_input, path_embeddings[:,i,:])#(batch, 1)
            batch_act_probs.append(act_probs)
            anchor_graph.append(batch['paths'][:,i:i+1].to(self.device))
            history_entity = batch['paths'][:, i:i+1].to(self.device)#same as forward func，state_input only depends on newly added nodes
            history_relation = batch['edges'][:, i:i+1].to(self.device)
        
        batch_act_probs = torch.cat(batch_act_probs, dim=1)#(batch, depth)
        indices = batch['label']>=-0.5
        labels = batch['label'][indices].to(self.device)
        predicts = batch_act_probs[indices.to(self.device)]
        loss = loss_fn(predicts, labels)

        return loss, predicts, labels
        

    # def predict_anchor(self, newsid, news_feature):
    #     self.doc_entity_dict[newsid] = news_feature[0]
    #     self.doc_feature_embedding = news_feature[1]
    #     prediction = self.forward([newsid], [newsid]).cpu().data.numpy()
    #     anchor_nodes = prediction[6]
    #     anchor_relation = prediction[8]
    #     return anchor_nodes, anchor_relation
    # def get_news_embedding_input(self, entity_ids, news_embeddings):
    #     entity_ids_index = entity_ids._indices()[0]
    #     news_embedding_batch = news_embeddings(entity_ids_index)
    #     return news_embedding_batch
    # def get_anchor_graph_list(self, anchor_graph_nodes, batch_size):
    #     anchor_graph_list = []
    #     for i in range(batch_size):
    #         anchor_graph_list.append([])
    #     for i in range(len(anchor_graph_nodes)):
    #         for j in range(len(anchor_graph_nodes[i])):
    #             anchor_graph_list[j].extend(list(map(lambda x:str(x), anchor_graph_nodes[i][j].data.cpu().numpy())))
    #     return anchor_graph_list


    # def get_advantage(self, curr_reward,  q_value):
    #     advantage = curr_reward - q_value
    #     return advantage

    # def get_actor_loss(self, act_probs_step, advantage):
    #     actor_loss = -act_probs_step * advantage.detach()
    #     return actor_loss
    #     # actor_loss = act_probs_step - advantage
    #     # actor_loss = torch.pow(actor_loss, 2)

    # def get_critic_loss(self, advantage):
    #     critic_loss = torch.pow(advantage, 2)
    #     return critic_loss

    # def step_update(self, act_probs_step, q_values_step, step_reward, embeddding_reward, reasoning_reward, alpha1=0.9, alpha2 = 0.1):
    #     embeddding_reward = torch.unsqueeze(embeddding_reward, dim=1)
    #     reasoning_reward = torch.unsqueeze(reasoning_reward, dim=1)
    #     embeddding_reward = embeddding_reward.expand(step_reward.shape[0], step_reward.shape[1])
    #     reasoning_reward = reasoning_reward.expand(step_reward.shape[0], step_reward.shape[1])
    #     curr_reward = alpha2 * step_reward + (1-alpha2) * (alpha1*embeddding_reward + (1-alpha1)*reasoning_reward)#本轮的total reward
    #     advantage = self.get_advantage(curr_reward.detach(), q_values_step) #curr_reward - q_values_step
    #     actor_loss = self.get_actor_loss(torch.log(act_probs_step), advantage)#self.get_actor_loss(act_probs_step, q_values_step)#self.get_actor_loss(torch.log(act_probs_step), advantage) # -act_probs_step * advantage #problem2
    #     critic_loss = advantage.pow(2)#self.get_critic_loss(advantage) #advantage.pow(2)

    #     return critic_loss, actor_loss