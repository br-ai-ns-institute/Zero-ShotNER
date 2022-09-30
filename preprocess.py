#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:07:14 2022

@author: nidzoni
"""

import pandas as pd
import numpy as np

dsets=[]
dsets.append(pd.read_pickle('./datasets/n2c2.pkl'))
#dsets.append(pd.read_pickle('./datasets/BioRED.pkl'))
dsets.append(pd.read_pickle('./datasets/CDR.pkl'))
dsets.append(pd.read_pickle('./datasets/JNLPBA.pkl'))
#dsets.append(pd.read_pickle('./datasets/NCB.pkl'))


dset=pd.DataFrame()
columns=['class','text','labels']
for ii in range(len(dsets)):
    print(ii)
    dset=pd.concat([dset,dsets[ii][columns]])

dset.reset_index(drop=True,inplace=True)
dset.to_pickle('./rawdataset.pickle')


counts=dset['class'].value_counts()

from transformers import AutoTokenizer
#modelname='gsarti/biobert-nli'  
modelname='dmis-lab/biobert-v1.1'
tokenizer = AutoTokenizer.from_pretrained(modelname)  ## loading the tokenizer of that model
encodings = tokenizer(dset['class'].to_list(),dset['text'].to_list(), is_split_into_words=True,
            padding=True, truncation=True, add_special_tokens=True, return_offsets_mapping=False)#,max_length=512)


#bb=list(zip(encodings.encodings[0].offsets,encodings.encodings[0].tokens))
duz=len(encodings.encodings[0].tokens)

labels=[]
for ii,entry in dset.iterrows():   
    aa=dset['labels'][ii]
    cc=np.array(encodings.encodings[ii].tokens)
    ind=np.where(cc=='[SEP]')
    cc=np.array(  encodings.encodings[ii].word_ids[ind[0][0]+1:ind[0][1]], dtype=np.uint8  )
    bb=np.zeros(ind[0][0]+1,dtype=np.uint8)
    dd=np.zeros(duz-ind[0][1],dtype=np.uint8)
    lista10=np.concatenate((bb,aa[cc],dd))
    labels.append(lista10)
    if ii%10000==0:
        print(ii)
labels=np.array(labels)

# import pickle
# filename = './encodings.pickle'
# outfile = open(filename,'wb')
# pickle.dump(encodings,outfile)
# outfile.close()
# outfile = open('./labels.pickle','wb')
# pickle.dump(labels,outfile)
# outfile.close()

model_path = "./model"  
tokenizer.save_pretrained(model_path)


from dataset import TarsDataset
dataset = TarsDataset(encodings, labels)
import pickle
filename = './dataset.pickle'
outfile = open(filename,'wb')
pickle.dump(dataset,outfile)
outfile.close()