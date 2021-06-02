doc_feature_dict = {}
fp_doc_feature1 = open("/data/RAGNRec_data/data/mind_dataset/doc_feature_entity(+body).tsv", 'r', encoding='utf-8')
for line in fp_doc_feature1:
    linesplit = line.strip().split('\t')
    if linesplit[0] not in doc_feature_dict:
        doc_feature_dict[linesplit[0]] = set()
    if len(linesplit)> 1:
        for i in range(len(linesplit)-1):
            doc_feature_dict[linesplit[0]].add(linesplit[i+1])
fp_doc_feature1.close()
fp_doc_feature2 = open("/data/RAGNRec_data/data/mind_dataset/doc_feature_entity.tsv", 'r', encoding='utf-8')
for line in fp_doc_feature2:
    linesplit = line.strip().split('\t')
    if linesplit[0] not in doc_feature_dict:
        doc_feature_dict[linesplit[0]] = set()
    if len(linesplit) > 1:
        for i in range(len(linesplit)-1):
            doc_feature_dict[linesplit[0]].add(linesplit[i+1])
fp_doc_feature2.close()

fp_doc_feature = open("/data/RAGNRec_data/data/mind_dataset/doc_feature_entity_all.tsv", 'w', encoding='utf-8')
for news in doc_feature_dict:
    fp_doc_feature.write(news+'\t'+'\t'.join(list(doc_feature_dict[news]))+'\n')
fp_doc_feature.close()

