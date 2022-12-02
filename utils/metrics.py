import numpy as np
from math import log2
from sklearn.metrics import roc_auc_score


def cal_auc(labels, preds):
    auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
    return auc

def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 1:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid], set(test_user_products[uid])
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / log2(i + 2)
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / log2(i + 2)
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    return avg_precision, avg_recall, avg_ndcg, avg_hit, invalid_users
    
