#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pickle
with open('./dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

from torch.utils.data import random_split, Subset

# train,valid,test=random_split(dataset,[round(0.8*len(dataset)),round(0.1*len(dataset)),round(0.1*len(dataset))])
# filename = './train.pickle'
# outfile = open(filename,'wb')
# pickle.dump(train.indices,outfile)
# outfile.close()

# filename = './test.pickle'
# outfile = open(filename,'wb')
# pickle.dump(test.indices,outfile)
# outfile.close()

# filename = './valid.pickle'
# outfile = open(filename,'wb')
# pickle.dump(valid.indices,outfile)
# outfile.close()


with open('./train.pickle', 'rb') as f:
      train = pickle.load(f)
with open('./valid.pickle', 'rb') as f:
      valid = pickle.load(f)
train = Subset(dataset, train)
valid = Subset(dataset, valid)



modelname='dmis-lab/biobert-v1.1'
#modelname = "./model"

from transformers import BertForTokenClassification #AutoModelForPreTraining
model = BertForTokenClassification.from_pretrained(modelname, num_labels=2)

from transformers import Trainer, TrainingArguments
# these are training settings
training_args = TrainingArguments(
    bf16=True,
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=32,   # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_strategy='steps',
    logging_steps=2000,
    save_steps=5000,
    evaluation_strategy='steps',
    eval_delay=40000,
    eval_steps = 5000, # Evaluation and Save happens every 10 steps
    save_total_limit = 4, # Only last 5 models are saved. Older ones are deleted.
    load_best_model_at_end=True
    )


trainer = Trainer(
    model=model,                         # the instantiated �~_�~W Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train,         # training dataset
    eval_dataset=valid            # evaluation dataset
    )
import time
t0=time.time()
trainer.train()
tt=time.time()-t0

model_path = "./model"  
model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)
print('proba')

