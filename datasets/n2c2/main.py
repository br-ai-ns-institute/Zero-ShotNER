# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pickle

import numpy as np
import pandas as pd

from bart_file_reader import read_bert_files_create_dataset,tokenize_brat_text_for_ner
import nltk
import csv

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    nltk.download('punkt')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    documents = read_bert_files_create_dataset('data/training_20180910/')
    texts, annotations = tokenize_brat_text_for_ner(documents)
    ann_types = []
    for annotation in annotations:
        for ann in annotation:
            if ann not in ann_types and ann!='O':
                ann_types.append(ann)
    csvfile =  open('dataset.csv', 'w', newline='')
    csv_writer = csv.writer(csvfile, delimiter=';',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
    big_array = []
    texts_arr = []
    ids = []
    classes = []
    labels = []
    id = 0
    for type in ann_types:
        for i in range(0,len(texts)):
            token_line = []
            ann_line = []
            for j in range(0,len(texts[i])):
                token_line.append(texts[i][j])
                if(type==annotations[i][j]):
                    ann_line.append(1)
                else:
                    ann_line.append(0)

            if len(texts[i])<3 and 1 not in ann_line:
                continue
            classes.append(type)
            labels.append(ann_line)
            texts_arr.append(token_line)
            ids.append(id)
            id = id +1
            ann_line = np.array(ann_line,dtype=np.uint8)
            big_array.append([type,token_line,ann_line])
            csv_writer.writerow([type,token_line,ann_line])
    csvfile.close()
    df = pd.DataFrame(list(zip(ids, classes, texts_arr, labels)),
                      columns=['ids', 'class', 'text', 'labels'])
    df.set_index('ids')

    df.insert(0, 'cor', 'n2c2')  # dodajemo kolonu na prvom mestu, cor(naziv kolone), naziv korpusa (JNLPBA)

    df['labels'] = df['labels'].apply(lambda x: np.array(x))  # konvertovanje liste u niz
    df['labels'] = df['labels'].apply(lambda x: x.astype(np.uint8))  # forsiranje tipa unisgnedInt
    df['class'] = df['class'].apply(lambda x: list([x]))  # promena tipa kolone iz stringa u listu od jednog stringa

    result1 = df.dtypes
    # print(df['Num_vector'][0].dtype)
#    print(df['labels'][0])
    df.to_excel('n2c2.xlsx')
    df.to_pickle('dataset_n2c2.pkl')
    print("Done.")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
