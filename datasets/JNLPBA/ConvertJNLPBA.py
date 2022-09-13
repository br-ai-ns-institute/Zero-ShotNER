#!/usr/bin/env python
# coding: utf-8

# Konvetrovanje JNLPBA dataset-a
# Ove verzija pravi vektore za sve klase - čak i ako nema entiteta ni nedne klase pravi se numericki vektor sa labelom te klase


import pandas as pd
import numpy as np


map_labels = {"O": 0,
        "B-DNA": 1,
        "I-DNA": 2,
        "B-RNA": 1,
        "I-RNA": 2,
        "B-cell_line":1,
        "I-cell_line":2,
        "B-cell_type":1,
        "I-cell_type":2,
        "B-protein" :1,
       "I-protein":2}
with open(r"C:\Users\Lenovo\OneDrive - IR INSTITUT ZA VESTACKU INTELIGENCIJU SRBIJE\Desktop\Bayer\Datasets\Biomedical_data\5.JNLPBA\training\Genia4ERtask1.iob2") as f:
    ids=list()
    texts=list()
    token_vectors=list()
    tag_vectors=list()
    num_vectors=list()
    klas_vectors=list()
    
    
    #kreiranje liste klasa iz rečnika map_labels
    klase=['DNA','RNA','cell_line','cell_type','protein']
    
    idd=0
    for x in f.read().split('\n\n'):
        try:
            
            # klase=list() # pronalazenje klasa za koje ce se kreirati posebni vektori npr:protein, DNA, RNA...
            # for y in x.split('\n'):
            #     token=y.split('\t')[0]
            #     tag=y.split('\t')[1]
            #     if tag != 'O':
            #         klasa=tag[2:]
            #         klase.append(klasa)
            
            # klase=set(klase) 
            idd+=1 # id je isti za vektore svih klasa za isti apstrakt
            klase_abstrakata=list() # koristim za dataframe da se vidi na koje klase se vektori odnose

            for kl in klase:
                token_vector=list()
                tag_vector=list()
                num_vector=list()
                klas_vector=list()
                klas_vector.append(kl)
                
                for y in x.split('\n'):
                    token=y.split('\t')[0]
                    tag=y.split('\t')[1]                   
                    token_vector.append(token)
                    tag_vector.append(tag)
                    
                    
                    #racunanje koja je klasa trenutnog tokena da bi je mapirali iz dictioary
                    if tag!='O':
                        klasa_tokena=tag[2:]
                    else:
                        klasa_tokena='O'
                    
                    if tag[2:]==kl:
                        num_tag=map_labels.get(tag)
                    else:
                        num_tag=0
                        
                    num_vector.append(num_tag)
                    
                   
                
                
                ids.append(idd)
                
                klas_vectors.append(kl)
                tag_vectors.append(tag_vector)
                token_vectors.append(token_vector)
                num_vectors.append(num_vector)
                
                
        except:
            break
                
                            

                            
    

#Isbacila sam kolonu Tag_vector jer Nikoli za treniranje ne treba
df=pd.DataFrame(list(zip(ids,klas_vectors,token_vectors, num_vectors)), columns=['ID','Klasa','Token_vector', 'Num_vector'])
df.set_index('ID')
df['Num_vector'] = df['Num_vector'].apply(lambda x: np.array(x))
df['Num_vector'] = df['Num_vector'].apply(lambda x: x.astype(np.uint8))
result1 = df.dtypes
#print(df['Num_vector'][0].dtype)
print(df['Num_vector'][0])
df.to_excel('ceoDataset\JNLPBA.xlsx')
df.to_pickle('ceoDataset\JNLPBA.pkl')








