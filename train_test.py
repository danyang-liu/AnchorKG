from utils.util import *
import hnswlib
from utils.metrics import *
from model.AnchorKG import *
from torch import nn, optim
from model.Reasoner import Reasoner
from model.Recommender import Recommender
from trainer.trainer import Trainer

def train(data, config):

    _, _, _, _, _, doc_feature_embedding, entity_adj, relation_adj, entity_id_dict, kg_env, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, hit_dict = data
    device, _ = prepare_device(config['n_gpu'])
    model_anchor = AnchorKG(config, doc_entity_dict, entity_doc_dict, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, hit_dict, entity_id_dict, neibor_embedding, neibor_num, device)
    model_recommender = Recommender(config, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, device)
    model_reasoner = Reasoner(config, entity_embedding, relation_embedding, device)


    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer_anchor = optim.Adam(model_anchor.parameters(), lr=0.1*config['lr'], weight_decay=config['weight_decay'])
    optimizer_recommender = optim.Adam(model_recommender.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer_reasoner = optim.Adam(model_reasoner.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    trainer = Trainer(config, model_anchor, model_recommender, model_reasoner, criterion, optimizer_anchor, optimizer_recommender, optimizer_reasoner, device, data)

    trainer.train()

def test(data, config):
    logger = config.get_logger('test')

    warmup_train_dataloader, warm_up_test_data, train_dataloader, Val_data, Test_data, doc_feature_embedding, entity_adj, relation_adj, entity_id_dict, kg_env, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, hit_dict = data
    device, deviceids = prepare_device(config['n_gpu'])
    model_anchor = AnchorKG(config, doc_entity_dict, entity_doc_dict, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, hit_dict, entity_id_dict, neibor_embedding, neibor_num, device)
    model_recommender = Recommender(config, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, device)
    model_reasoner = Reasoner(config, entity_embedding, relation_embedding, device)
    #load model
    model_anchor.load_state_dict(torch.load('./out/saved/models/AnchorKG/1103_152727/checkpoint-anchor.pt'))
    model_recommender.load_state_dict(torch.load('./out/saved/models/AnchorKG/1103_152727/checkpoint-recommender.pt'))
    model_reasoner.load_state_dict(torch.load('./out/saved/models/AnchorKG/1103_152727/checkpoint-reasoner.pt'))
   
    #test
    model_anchor.eval()
    model_recommender.eval()
    model_reasoner.eval()

    # get all news embeddings
    doc_list = list(Test_data.keys())
    logger.info('len(doc_list) : {}'.format(len(doc_list)))
    start_list = list(range(0, len(doc_list), config['batch_size']))
    doc_embedding = []
    doc_embedding_dict = {}
    for start in start_list:
        end = start + config['batch_size']
        _, _, _, anchor_graph1, _ = model_anchor(doc_list[start:end])
        doc_embedding.extend(model_recommender(doc_list[start:end], doc_list[start:end], anchor_graph1, anchor_graph1)[1].cpu().data.numpy())
        
    # knn search topk
    ann = hnswlib.Index(space='cosine', dim=128)
    ann.init_index(max_elements=len(doc_list), ef_construction=200, M=16)
    for i in range(len(doc_list)):
        doc_embedding_dict[doc_list[i]] = doc_embedding[i]
        ann.add_items(doc_embedding[i], i)
    ann.set_ef(100)
    predict_dict = {}
    for doc in Test_data:
        doc_embedding = doc_embedding_dict[doc]
        labels, distances = ann.knn_query(doc_embedding, k=10)
        predict_dict[doc] = list(map(lambda x: doc_list[x], labels[0]))
    # compute metric
    avg_precision, avg_recall, avg_ndcg, avg_hit, invalid_users = evaluate(predict_dict, Test_data)
    logger.info('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))
    



