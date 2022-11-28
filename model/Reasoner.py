import torch
import torch.nn as nn
from utils import *
import networkx as nx
from model.base_model import BaseModel

class Reasoner(BaseModel):

    def __init__(self, config, doc_feature_embedding, entity_embedding, relation_embedding, device=torch.device('cpu')):
        super(Reasoner, self).__init__()
        self.device = device
        self.config = config
        self.sigmoid = nn.Sigmoid()
        self.doc_feature_embedding = doc_feature_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.innews_relation = nn.Embedding(1, self.config['embedding_size']).to(device)
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
        self.gru = torch.nn.GRU(2*self.config['embedding_size'], self.config['embedding_size'], batch_first=True).to(device)
        self.gru_layer = nn.Sequential(
                            nn.Linear(self.config['embedding_size'], self.config['embedding_size']),
                            nn.ELU(),
                            nn.Linear(self.config['embedding_size'],1),
                        ).to(device)

    def get_overlap_entities(self, anchor_graph1, anchor_graph2):
        overlap_entity_num = []
        anchor_graph1_num = []
        anchor_graph2_num = []
        for i in range(len(anchor_graph1)):
            anchor_graph1_set = set()
            anchor_graph2_set = set()
            for anchor_node in anchor_graph1[i]:
                anchor_graph1_set.add(int(anchor_node))
            for anchor_node in anchor_graph2[i]:
                anchor_graph2_set.add(int(anchor_node))
            anchor_graph1_set.discard(0)
            anchor_graph2_set.discard(0)
            overlap_entity_num.append(len(anchor_graph1_set & anchor_graph2_set))
            anchor_graph1_num.append(len(anchor_graph1_set))
            anchor_graph2_num.append(len(anchor_graph2_set))
        return torch.tensor(overlap_entity_num).to(self.device), torch.tensor(anchor_graph1_num).to(self.device), torch.tensor(anchor_graph2_num).to(self.device), overlap_entity_num

    def get_anchor_graph_list(self, anchor_graph_layers, batch_size):
        anchor_graph_list_flat = []#flattened，(batch,50)
        anchor_graph_list = []#hierarchical，(batch,[5,15,30])
        for i in range(batch_size):
            anchor_graph_list_flat.append([])
            anchor_graph_list.append([[],[],[]])
        for i in range(len(anchor_graph_layers)):#3
            for j in range(len(anchor_graph_layers[i])):#batch
                for k in range(len(anchor_graph_layers[i][j])):#5/15/30
                    anchor_graph_list[j][i].append(int(anchor_graph_layers[i][j][k].data.cpu().numpy()))
                    anchor_graph_list_flat[j].append(int(anchor_graph_layers[i][j][k].data.cpu().numpy()))
        return anchor_graph_list_flat, anchor_graph_list

    def get_reasoning_paths(self, news1, news2, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2, overlap_entity_num_cpu):
        reasoning_paths = []
        reasoning_edges = []
        for i in range(len(news1)):
            reasoning_paths.append([])
            reasoning_edges.append([])
            if overlap_entity_num_cpu[i]>0:
                subgraph = nx.Graph()
                for index1 in range(self.config['topk'][0]):
                    if anchor_graph1[i][0][index1] != 0:
                        subgraph.add_edge(news1[i], anchor_graph1[i][0][index1], weight=0)
                    if anchor_graph2[i][0][index1] != 0:
                        subgraph.add_edge(news2[i], anchor_graph2[i][0][index1], weight=0)
                    for index2 in range(self.config['topk'][1]):
                        if anchor_graph1[i][1][index1*self.config['topk'][1]+index2] != 0:
                            subgraph.add_edge(anchor_graph1[i][0][index1], anchor_graph1[i][1][index1*self.config['topk'][1]+index2], weight=anchor_relation1[1][i][index1*self.config['topk'][1]+index2])
                        if anchor_graph2[i][1][index1*self.config['topk'][1]+index2] != 0:
                            subgraph.add_edge(anchor_graph2[i][0][index1], anchor_graph2[i][1][index1*self.config['topk'][1]+index2], weight=anchor_relation2[1][i][index1*self.config['topk'][1]+index2])
                        for index3 in range(self.config['topk'][2]):
                            if anchor_graph1[i][2][index1*self.config['topk'][1]*self.config['topk'][2]+index2*self.config['topk'][2]+index3] != 0:
                                subgraph.add_edge(anchor_graph1[i][1][index1*self.config['topk'][1]+index2], anchor_graph1[i][2][index1*self.config['topk'][1]*self.config['topk'][2]+index2*self.config['topk'][2]+index3], weight=anchor_relation1[2][i][index1*self.config['topk'][1]*self.config['topk'][2]+index2*self.config['topk'][2]+index3])
                            if anchor_graph2[i][2][index1*self.config['topk'][1]*self.config['topk'][2]+index2*self.config['topk'][2]+index3] != 0:
                                subgraph.add_edge(anchor_graph2[i][1][index1*self.config['topk'][1]+index2], anchor_graph2[i][2][index1*self.config['topk'][1]*self.config['topk'][2]+index2*self.config['topk'][2]+index3], weight=anchor_relation2[2][i][index1*self.config['topk'][1]*self.config['topk'][2]+index2*self.config['topk'][2]+index3])
                
                for path in nx.all_simple_paths(subgraph, source=news1[i], target=news2[i], cutoff=6):
                    reasoning_paths[-1].append(path[1:-1])
                    reasoning_edges[-1].append([])
                    for j in range(1, len(path)-2):
                        reasoning_edges[-1][-1].append(int(subgraph[path[j]][path[j+1]]['weight']))
            if len(reasoning_paths[-1]) == 0:
                reasoning_paths[-1].append([0])
                reasoning_edges[-1].append([])
        return reasoning_paths, reasoning_edges

    def forward(self, news1, news2, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2):

        anchor_graph_list1_flat, anchor_graph_list1 = self.get_anchor_graph_list(anchor_graph1, len(news1))
        anchor_graph_list2_flat, anchor_graph_list2 = self.get_anchor_graph_list(anchor_graph2, len(news2))
        overlap_entity_num, anchor_graph1_num, anchor_graph2_num, overlap_entity_num_cpu = self.get_overlap_entities(anchor_graph_list1_flat, anchor_graph_list2_flat)
        reasoning_paths, reasoning_edges = self.get_reasoning_paths(news1, news2, anchor_graph_list1, anchor_graph_list2, anchor_relation1, anchor_relation2, overlap_entity_num_cpu)
        batch_predict = []
        batch_path_scores = []
        for i in range(len(reasoning_paths)):
            paths = reasoning_paths[i]
            edges = reasoning_edges[i]
            news_embeeding_1 = self.news_compress(self.doc_feature_embedding[news1[i]].to(self.device)).unsqueeze(dim=0)
            news_embedding_2 = self.news_compress(self.doc_feature_embedding[news2[i]].to(self.device)).unsqueeze(dim=0)
            path_scores=[]
            for j in range(len(paths)):
                path_node_embeddings = self.entity_compress(self.entity_embedding(torch.tensor(paths[j], dtype=torch.long)).to(self.device))
                path_edge_embeddings = self.relation_compress(self.relation_embedding(torch.tensor(edges[j], dtype=torch.long)).to(self.device))
                path_node_embeddings = torch.cat((news_embeeding_1, path_node_embeddings, news_embedding_2), dim=0) #(path_len, embedding_size)
                path_edge_embeddings = torch.cat((self.innews_relation(torch.tensor([0]).to(self.device)), path_edge_embeddings, self.innews_relation(torch.tensor([0]).to(self.device)), torch.zeros([1, self.config['embedding_size']]).to(self.device)), dim=0) #(path_len, embedding_size)
                path_node_embeddings = torch.unsqueeze(path_node_embeddings, 0)#(1, path_len, embedding_size)
                path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 0)#(1, path_len, embedding_size)
                output, _ = self.gru(torch.cat((path_node_embeddings, path_edge_embeddings), dim=2))#(1, path_len, embedding_size)
                path_score = self.gru_layer(torch.squeeze(output)[-1])#[1]
                path_scores.append(path_score)
            path_scores = torch.stack(path_scores, dim=0) #(path_num, 1)
            batch_path_scores.append(path_scores)
            predict = torch.logsumexp(path_scores, dim=0)#[1]#path_scores.shape[0]
            batch_predict.append(predict)

        path_predicts = torch.cat(batch_predict, dim=0) #[batch_size]
        predicts_qua = torch.div(path_predicts, torch.log(torch.e+anchor_graph1_num+anchor_graph2_num))
        predicts_num = torch.div(overlap_entity_num, torch.log(torch.e+anchor_graph1_num+anchor_graph2_num))
        predicts = self.sigmoid(0.8*predicts_qua + 0.2*predicts_num)

       
        return predicts, reasoning_paths, reasoning_edges, batch_predict, batch_path_scores
    
    # def get_path_score(self, reasoning_paths):
    #     predict_scores = []
    #     for paths in reasoning_paths:
    #         predict_scores.append([0])
    #         for path in paths:
    #             path_node_embeddings = self.entity_compress(self.entity_embedding(torch.tensor(path)).to(self.device))
    #             if len(path_node_embeddings.shape) == 1:
    #                 path_node_embeddings = torch.unsqueeze(path_node_embeddings, 0)
    #             path_node_embeddings = torch.unsqueeze(path_node_embeddings, 1)
    #             output, h_n = self.gru(path_node_embeddings)
    #             path_score = (self.gru_layer(torch.squeeze(output[-1])))
    #             predict_scores[-1].append(path_score)
    #         predict_scores[-1] = torch.sum(predict_scores[-1]).float()
    #     return torch.stack(predict_scores).to(self.device)
    # gt_indices = torch.gt(predicts, 1.0)
    # lt_indices = torch.lt(predicts, 0.0)
    # gt_predicts = predicts[gt_indices]
    # lt_predicts = predicts[lt_indices]
    # if len(gt_predicts) > 0 or len(lt_predicts) > 0:
    #     print('error')
    #     print("gt_predicts", gt_predicts)
    #     print("lt_predicts", lt_predicts)
    #     print(path_predicts[gt_indices])
    #     print(predicts_qua[gt_indices])
    #     print(predicts_num[gt_indices])
    #     print(path_predicts[lt_indices])
    #     print(predicts_qua[lt_indices])
    #     print(predicts_num[lt_indices])
    #     exit(0)