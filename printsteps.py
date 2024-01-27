import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
###Data Preprossesing
##function to convert conll format to pandas
def read_conll(filename):
    df = pd.read_csv(filename,
                     na_values=['#'],
                    sep = ' ', header = None, keep_default_na = False,
                    names = ['TOKEN', 'POS', 'CHUNK', 'NE'],
                    quoting = 3, skip_blank_lines = False)
    df = df.dropna()
    df['SENTENCE'] = (df.TOKEN == '').cumsum()
    return df[df.TOKEN != '']
##use only TOKEN NE SENTENCE
def preprossesingData(data):
    data = data[["TOKEN", "NE", "SENTENCE"]].drop_duplicates().reset_index(drop=True)
    data['Sentence'] = data[['SENTENCE','TOKEN','NE']].groupby(['SENTENCE'])['TOKEN'].transform(lambda x: ' '.join(x))
    data['IOBTags'] = data[['SENTENCE','TOKEN','NE']].groupby(['SENTENCE'])['NE'].transform(lambda x: ','.join(x))
    #reform the data as sentences(tokenlists) and IOBTag lists
    data = data[["Sentence", "IOBTags"]].drop_duplicates().reset_index(drop=True)
    return data
###Data Preprocessing
trainingData = read_conll('data/train/en-train.conll')
testData= read_conll('data/test/en_test_withtags.conll')
frequencies = trainingData.NE.value_counts()
#NE tag counts
tags = {}
for tag, count in zip(frequencies.index, frequencies):
    if tag != "":
        if tag != "O":
            if tag[2:5] not in tags.keys():
                tags[tag[0:30]] = count
            else:
                tags[tag[0:30]] += count
        continue
#print tag counts sorted
print(sorted(tags.items(), key=lambda x: x[1], reverse=True))
#number of unique tags
id2label = {v: k for v, k in enumerate(trainingData.NE.unique())}
label2id = {k: v for v, k in enumerate(trainingData.NE.unique())}
print(label2id)
trainingData=preprossesingData(trainingData)
testData=preprossesingData(testData)
print(trainingData.iloc[41].Sentence)
print(trainingData.iloc[41].IOBTags)