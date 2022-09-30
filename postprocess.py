#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:00:49 2022

@author: nidzoni
"""


import pickle
modelname = "./model"

with open('./dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
with open('./test.pickle', 'rb') as f:
    test = pickle.load(f)
import torch
testset = torch.utils.data.Subset(dataset, test)


from transformers import BertForTokenClassification #AutoModelForPreTraining  
model = BertForTokenClassification.from_pretrained(modelname, num_labels=2)

from transformers import Trainer
evaler = Trainer(
    model=model
)
pred=evaler.predict(testset)

filename = './pred.pickle'
outfile = open(filename,'wb')
pickle.dump(pred,outfile)
outfile.close()



# with open('./pred.pickle', 'rb') as f:
#     pred = pickle.load(f)

import numpy as np
pred0=np.reshape(pred[0],(-1,2))
pred1=np.reshape(pred[1],(-1))
from sklearn.metrics import confusion_matrix, recall_score, f1_score
matrix=confusion_matrix( pred1, pred0.argmax(axis=1))
f1=f1_score( pred1, pred0.argmax(axis=1),average=None)
recall=recall_score( pred1, pred0.argmax(axis=1),average=None)                                                       

#type_ids=test[:]['token_type_ids']

wids=np.array([dataset.encodings.encodings[ii].word_ids for ii in test])
wids[wids==None]=-1
wids=wids.astype(int)
type_ids=np.array([dataset.encodings.encodings[ii].type_ids for ii in test],dtype=bool)
pre=pred[0].argmax(axis=2)
pre_list=[]
test_list=[]
for ii in range(wids.shape[0]):
    test_list.append(wids[ii][type_ids[ii]])
    pre_list.append(pre[ii][type_ids[ii]])
labels=[]
for ii in range(len(pre_list)):
    labels.append(np.array(range(test_list[ii].max()+1)))
    for jj in labels[ii]:
        labels[ii][jj]=np.uint(np.array(pre_list[ii])[np.where(test_list[ii]==jj)[0]].mean())
#ovo ide na mesto broj 20


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(modelname)  ## loading the tokenizer of that model
encs=[dataset.encodings['input_ids'][ii] for ii in test]
lis=[]
for seq in encs:
    lis.append(tokenizer.decode(seq,skip_special_tokens=True,split_into_words=True))

import pandas as pd
dset=pd.read_pickle('./rawdataset.pickle')
labels_original=np.array(dset.iloc[test]['labels'])
#selencs=list(encs[mask])
