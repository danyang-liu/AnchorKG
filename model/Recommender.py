import torch
import torch.nn as nn
from model.base_model import BaseModel

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

        self.softmax = nn.Softmax(dim=-2)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
                        nn.Linear(self.config['embedding_size']*2, self.config['embedding_size']),
                        nn.ELU(),
                        nn.Linear(self.config['embedding_size'], self.config['embedding_size']),
                        nn.Tanh(),
                    ).to(device)
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
        self.anchor_embedding_layer = nn.Sequential(
                                        nn.Linear(self.config['embedding_size']*2, self.config['embedding_size'], bias=False),
                                        nn.Tanh(),
                                    ).to(device)
        self.anchor_layer = nn.Sequential(
                                nn.Linear(self.config['embedding_size'], self.config['embedding_size'], bias=False),
                                nn.ELU(),
                                nn.Linear(self.config['embedding_size'],1, bias=False),
                            ).to(device)

    def get_news_embedding_batch(self, newsids):
        news_embeddings = torch.zeros([len(newsids), self.config['doc_embedding_size']])
        for i, newsid in enumerate(newsids):
            news_embeddings[i] = self.doc_feature_embedding[newsid]
        return news_embeddings.to(self.device)

    def get_neighbors(self, entities):
        neighbor_entities = self.entity_adj[entities]
        neighbor_relations = self.relation_adj[entities]
        return neighbor_entities, neighbor_relations

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

    def forward(self, news1, news2, anchor_graph1, anchor_graph2):
        news_embedding1 = self.get_news_embedding_batch(news1)
        news_embedding2 = self.get_news_embedding_batch(news2)
        news_embedding1 = self.news_compress(news_embedding1)
        news_embedding2 = self.news_compress(news_embedding2)
        anchor_embedding1 = self.get_anchor_graph_embedding(anchor_graph1)
        anchor_embedding2 = self.get_anchor_graph_embedding(anchor_graph2)
        news_embedding1 = torch.cat([news_embedding1, anchor_embedding1], dim=-1)
        news_embedding2 = torch.cat([news_embedding2, anchor_embedding2], dim=-1)

        news_embedding1 = self.mlp(news_embedding1)
        news_embedding2 = self.mlp(news_embedding2)
        predict = self.sigmoid(self.cos(news_embedding1, news_embedding2))
        return predict, news_embedding1