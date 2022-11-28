# Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning

This repository contains the source code of the paper: Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning

![framework](./framework.png)

## Dataset:

The original data we used is from the public news dataset : [MIND](https://msnews.github.io). We build an item2item dataset based on the method in the paper.

Files in data folder:

-   `./data/`
    -   `kg/wikidata-graph/`
        - `wikidata-graph.tsv` knowledge graph triples from Wikidata
        - `entity2id.txt` entity label to index
        - `relation2id.txt` relation label to index
        - `entity2vecd100.vec` entity embedding from TransE
        - `relation2vecd100.vec` relation embedding from TransE
    -   `mind/`
        - `behaviors.tsv` the impression logs and users' news click hostories
        - `news.tsv` the detailed information of news articles involved in the behaviors.tsv file
    -   `item2item/`
        - `all_news.tsv` all news used for training, validating, testing
        - `doc_feature_embedding.tsv` document embedding from sentence-bert
        - `doc_feature_entity.tsv` entities mentioned in documents
        - `pos_train.tsv` positive item pairs in train data
        - `pos_valid.tsv` positive item pairs in valid data
        - `pos_test.tsv` positive item pairs in test data
        - `random_neg_sample_train.tsv` item2item train data
        - `random_neg_sample_valid.tsv` item2item valid data
    -  `kprn/`
        - `train_data.json` train data for KPRN
        - `valid_data.json` valid data for KPRN
        - `predict_train.json` warm up train data for anchorKG
        - `predict_valid.json` warm up valid data for anchorKG

## Requirements:
```
python == 3.9.13
torch == 1.12.0
sklearn == 1.1.2
numpy == 1.23.4
hnswlib == 0.4.0
networkx == 2.8.7
nni == 2.8
sentence_transformers == 2.2.2
tqdm == 4.64.1
```
## How to run the code:

1. Dataset download and process

    `$ python data_process.py`

    > The config file is ./config/data_config.json
    
    > If the download speed is too slow, you can refer to followng links for dataset download and put it under the corresponding folder before running the code.
    * [MIND_large_train](https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip): ./data/mind/train/
    * [MIND_large_valid](https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip): ./data/mind/valid/
    * [MIND_small_train](https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip): ./data/mind/train/
    * [MIND_small_valid](https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip): ./data/mind/valid/
    * [Knowledge Graph](https://kredkg.blob.core.windows.net/wikidatakg/kg.zip): ./data/kg/


2. Kprn training
    
    `$ python KPRN_train.py`

    > The config file is ./config/KPRN_config.json

3. Warmup training + AnchorKG training
    
    `$ python main.py`

    > The config file is ./config/anchorkg_config.json

## Automatic hyper-parameter tuning

We integrates with NNI module for tuning the hyper-parameters automatically. You can tune the KPRN training stage, warmpup training stage, anchorKG training stage respectively. For easy usage, you can run the following code:

`$ nnictl create --config ./nni_config.yaml --port 9074`
    
You can configure the nni_config.yaml for your own usage. For more details about NNI, please refer to [NNI Documentation](https://nni.readthedocs.io/zh/stable/)