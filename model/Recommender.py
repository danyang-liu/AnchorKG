import torch
import torch.nn as nn
from base.base_model import BaseModel

class Recommender(BaseModel):

    def __init__(self, config, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, device=torch.device('cpu')):
        super(Recommender, self).__init__()
        self.device = device
        self.config = config
        self.doc_feature_embedding = doc_feature_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj

        self.elu = nn.ELU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-2)
        self.mlp_layer1 = nn.Linear(self.config['model']['embedding_size']*2, self.config['model']['embedding_size'])
        self.mlp_layer2 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.news_compress1 = nn.Linear(self.config['model']['doc_embedding_size'], self.config['model']['embedding_size'])
        self.news_compress2 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.anchor_embedding_layer = nn.Linear(self.config['model']['embedding_size']*2, self.config['model']['embedding_size'])
        self.anchor_weights_layer1 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.anchor_weights_layer2 = nn.Linear(self.config['model']['embedding_size'],1)

        self.entity_compress = nn.Linear(100, self.config['model']['embedding_size'])
        self.relation_compress = nn.Linear(100, self.config['model']['embedding_size'])

        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.doc_feature_embedding[newsid]).to(self.device))
        return torch.stack(news_embeddings)

    def get_neighbors(self, entities):
        neighbor_entities = []
        neighbor_relations = []
        for entity_batch in entities:
            neighbor_entities.append([])
            neighbor_relations.append([])
            for entity in entity_batch:
                if type(entity) == int:
                    neighbor_entities[-1].append(self.entity_adj[entity])
                    neighbor_relations[-1].append(self.relation_adj[entity])
                else:
                    neighbor_entities[-1].append([])
                    neighbor_relations[-1].append([])
                    for entity_i in entity:
                        neighbor_entities[-1][-1].append(self.entity_adj[entity_i])
                        neighbor_relations[-1][-1].append(self.relation_adj[entity_i])

        return torch.tensor(neighbor_entities).to(self.device), torch.tensor(neighbor_relations).to(self.device)

    def get_anchor_graph_embedding(self, anchor_graph):
        anchor_graph_nodes = []#(batch, 50),把图上所有节点的id都放进去
        for i in range(len(anchor_graph[1])):
            anchor_graph_nodes.append([])
            for j in range(len(anchor_graph)):
                anchor_graph_nodes[-1].extend(anchor_graph[j][i].tolist())

        anchor_graph_nodes_embedding = self.tanh(self.entity_compress(self.entity_embedding(torch.tensor(anchor_graph_nodes).to(self.device))))
        neibor_entities, neibor_relations = self.get_neighbors(anchor_graph_nodes)#为每个结点找一阶邻居
        neibor_entities_embedding = self.tanh(self.entity_compress(self.entity_embedding(neibor_entities)))
        neibor_relations_embedding = self.tanh(self.relation_compress(self.relation_embedding(neibor_relations)))
        anchor_embedding = torch.cat([anchor_graph_nodes_embedding, torch.sum(neibor_entities_embedding+neibor_relations_embedding, dim=-2)], dim=-1)
        anchor_embedding = self.tanh(self.anchor_embedding_layer(anchor_embedding))#(batch, 50, 128)
        anchor_embedding_weight = self.softmax(self.anchor_weights_layer2(self.elu(self.anchor_weights_layer1(anchor_embedding))))#(batch, 50, 1)
        anchor_embedding = torch.sum(anchor_embedding * anchor_embedding_weight, dim=-2)
        return anchor_embedding

    def forward(self, news1, news2, anchor_graph1, anchor_graph2):
        news_embedding1 = self.get_news_embedding_batch(news1)
        news_embedding2 = self.get_news_embedding_batch(news2)
        news_embedding1 = self.tanh(self.news_compress2(self.elu(self.news_compress1(news_embedding1))))
        news_embedding2 = self.tanh(self.news_compress2(self.elu(self.news_compress1(news_embedding2))))
        anchor_embedding1 = self.get_anchor_graph_embedding(anchor_graph1)
        anchor_embedding2 = self.get_anchor_graph_embedding(anchor_graph2)
        news_embedding1 = torch.cat([news_embedding1,anchor_embedding1], dim=-1)
        news_embedding2 = torch.cat([news_embedding2,anchor_embedding2], dim=-1)

        news_embedding1 = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(news_embedding1))))
        news_embedding2 = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(news_embedding2))))
        predict = self.sigmoid((self.cos(news_embedding1, news_embedding2)+1)/2)
        return predict, news_embedding1