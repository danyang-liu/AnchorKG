from utils.util import *
import hnswlib
from utils.metrics import *
from model.AnchorKG import *
from torch import nn, optim
from model.Reasoner import Reasoner
from model.Recommender import Recommender
from trainer.trainer import Trainer

def train_test(data, config):



    warmup_train_dataloader, warm_up_test_data, train_dataloader, Val_data, Test_data, doc_feature_embedding, entity_adj, relation_adj, entity_id_dict, kg_env, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, hit_dict = data
    device, deviceids = prepare_device(config['n_gpu'])
    model_anchor = AnchorKG(config, doc_entity_dict, entity_doc_dict, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, kg_env, hit_dict, entity_id_dict, neibor_embedding, neibor_num).to(device)
    model_recommender = Recommender(config, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj).to(device)
    model_reasoner = Reasoner(config, kg_env, entity_embedding, relation_embedding).to(device)


    criterion = nn.BCEWithLogitsLoss(reduce=False)
    optimizer_anchor = optim.Adam(model_anchor.parameters(), lr=0.1*config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    optimizer_recommender = optim.Adam(model_recommender.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    optimizer_reasoner = optim.Adam(model_reasoner.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])

    trainer = Trainer(config, model_anchor, model_recommender, model_reasoner, criterion, optimizer_anchor, optimizer_recommender, optimizer_reasoner, device, data)

    trainer.train()

    #test
    model_anchor.eval()
    model_recommender.eval()
    model_reasoner.eval()

    # get all news embeddings
    doc_list = list(Test_data.keys())
    print("length: " + str(len(doc_list)))
    start_list = list(range(0, len(doc_list), config['data_loader']['batch_size']))
    doc_embedding = []
    doc_embedding_dict = {}
    for start in start_list:
        if start +config['data_loader']['batch_size'] <= len(doc_list):
            end = start + config['data_loader']['batch_size']
            act_probs_steps1, q_values_steps1, act_probs_steps2, q_values_steps2, step_rewards1, step_rewards2, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2 = model_anchor(
                doc_list[start:end], doc_list[start:end])
            doc_embedding.extend(model_recommender(doc_list[start:end], doc_list[start:end], anchor_graph1, anchor_graph2)[
                                     1].cpu().data.numpy())
        else:
            act_probs_steps1, q_values_steps1, act_probs_steps2, q_values_steps2, step_rewards1, step_rewards2, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2 = model_anchor(
                doc_list[start:], doc_list[start:])
            doc_embedding.extend(model_recommender(doc_list[start:], doc_list[start:], anchor_graph1, anchor_graph2)[
                                     1].cpu().data.numpy())
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
    evaluate(predict_dict, Test_data)



