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
        self.mlp_layer1 = nn.Linear(self.config['embedding_size']*2, self.config['embedding_size']).to(device)
        self.mlp_layer2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size']).to(device)
        self.news_compress1 = nn.Linear(self.config['doc_embedding_size'], self.config['embedding_size']).to(device)
        self.news_compress2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size']).to(device)
        self.anchor_embedding_layer = nn.Linear(self.config['embedding_size']*2, self.config['embedding_size']).to(device)
        self.anchor_weights_layer1 = nn.Linear(self.config['embedding_size'], self.config['embedding_size']).to(device)
        self.anchor_weights_layer2 = nn.Linear(self.config['embedding_size'],1).to(device)

        self.entity_compress = nn.Linear(100, self.config['embedding_size']).to(device)
        self.relation_compress = nn.Linear(100, self.config['embedding_size']).to(device)

        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()

    def get_news_embedding_batch(self, newsids):
        news_embeddings = torch.zeros([len(newsids), self.config['doc_embedding_size']])
        for i, newsid in enumerate(newsids):
            news_embeddings[i] = self.doc_feature_embedding[newsid]
        return news_embeddings.to(self.device)

    def get_neighbors(self, entities):
        neighbor_entities = torch.zeros([len(entities), len(entities[0]), 20], dtype=torch.long)
        neighbor_relations = torch.zeros([len(entities), len(entities[0]), 20], dtype=torch.long)
        for i, entity_batch in enumerate(entities):
            for j, entity in enumerate(entity_batch):
                assert type(entity) == int
                neighbor_entities[i][j] = self.entity_adj[entity]
                neighbor_relations[i][j] = self.relation_adj[entity]
                
        return neighbor_entities, neighbor_relations #(batch, 5+5*3+5*3*2, 20)

    def get_anchor_graph_embedding(self, anchor_graph):
        anchor_graph_nodes = []#(batch, 50),把图上所有节点的id都放进去
        for i in range(len(anchor_graph[1])):
            anchor_graph_nodes.append([])
            for j in range(len(anchor_graph)):
                anchor_graph_nodes[-1].extend(anchor_graph[j][i].tolist())

        anchor_graph_nodes_embedding = self.tanh(self.entity_compress(self.entity_embedding(torch.tensor(anchor_graph_nodes)).to(self.device)))
        neibor_entities, neibor_relations = self.get_neighbors(anchor_graph_nodes)#为每个结点找一阶邻居
        neibor_entities_embedding = self.tanh(self.entity_compress(self.entity_embedding(neibor_entities).to(self.device)))
        neibor_relations_embedding = self.tanh(self.relation_compress(self.relation_embedding(neibor_relations).to(self.device)))
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