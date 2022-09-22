import utilities as utilis
from nltk.tokenize.util import align_tokens
import nltk
import tqdm

def generate_relationship_dataset(documents):
    for doc in documents:
        pass

def read_txt_files(path):
    files = utilis.get_files_from_folder(path)
    docs = []
    for file in files:
        text = ''
        if '.txt' in file:
            base_name = file.replace('.txt', '')
            f = open(path + file, 'r', encoding='utf-8')
            text = f.read()
            f.close()
            docs.append({'file_name':base_name,'text':text})
    return docs


def read_bert_files_create_dataset(path):
    """
    BRAT Rapid annotation file format reader for input to machine learning (or rule-based) training input.
    This is a prototype and does not support broken annotations (made of multiple segments).
    :param path: Path where the files are
    :return: a list of documents. Each document is a tuple of (text,terms,relations). tearms and relations are list of tuples.
    A list of terms contains tuple (trm_id,ner_name,start,end,string), while a list of relations contains tuples (trm_id,rel_name,rel_arg1,rel_arg2)
    """
    files = utilis.get_files_from_folder(path)
    docs = []
    for file in files:
        if '.txt' in file:
            base_name = file.replace('.txt','')
            f = open(path+file,'r',encoding='utf-8')
            text = f.read().replace('``',"").replace('``',"")
            f.close()
            f2 = open(path+base_name+'.ann','r',encoding='utf-8')
            anns = f2.readlines()
            terms = []
            relations = []
            for ann in anns:
                trm = ann.split('\t')
                trm_id = trm[0]
                if 'T' in trm_id:
                    metadata = trm[1].split(' ')
                    ner_name = metadata[0]
                    start = metadata[1]
                    end = metadata[2]
                    string = trm[2].replace('\n','')
                    terms.append((trm_id,ner_name,start,end,string))
                if 'R' in trm_id:
                    metadata = trm[1].split(' ')
                    rel_name = metadata[0]
                    rel_arg1 = metadata[1].replace('Arg1:','')
                    rel_arg2 = metadata[2].replace('Arg2:', '').replace('\n','')
                    relations.append((trm_id,rel_name,rel_arg1,rel_arg2))
            document = (text,terms,relations)
            docs.append(document)
    return docs



def tokenize_text(text):
    token_sequences = []
    new_text = str(text).replace("\"", "'")
    new_text = str(new_text).replace("`", "'")
    #new_text = str(new_text).replace("``", "")
    #new_text = str(new_text).replace('``',"")
    new_text = str(new_text).replace("\'", "'")
    new_text = new_text.replace("''", "")
    #tokens = nltk.tokenize.word_tokenize(new_text)
    tokens = utilis.custom_word_tokenize(new_text.replace('``',""))
    try:
        token_spans = align_tokens(tokens, new_text)
    except:
        return []
    token_sequence = []
    for i in range(0, len(tokens)):
        token_txt = text[token_spans[i][0]:token_spans[i][1]]
        token_sequence.append(token_txt)
        if token_txt == '.' or token_txt == '!' or token_txt == '?':
            token_sequences.append(token_sequence)
            token_sequence = []
        if i == len(tokens)-1:
            token_sequences.append(token_sequence)
    return token_sequences

def tokenize_brat_text_for_ner(documents):
    token_sequences =[]
    annotations_sequences = []
    for doc in tqdm.tqdm(documents):
        text = doc[0]
        text = text.replace("\"", "'")
        text = text.replace("`", "'")
        text = text.replace("``", "")
        #text = text.replace("''", "")
        tokens = nltk.tokenize.word_tokenize(text)
        try:
            token_spans = align_tokens(tokens,text)
        except:
            continue
        token_sequence = []
        annotation_sequence = []
        for i in range(0,len(tokens)):
            token_tag = 'O'

            for tag in doc[1]:
                end = 0
                if len(tag)>3 and tag[3] != None and ';' in tag[3]:
                    end = tag[3].split(';')[1]
                else:
                    end = tag[3]
                if int(tag[2])<=token_spans[i][0] and int(end)>=token_spans[i][1]:
                    if tag[1] == 'GENE-Y' or tag[1] == 'GENE-N':
                    #if tag[1] == 'Disease':
                        #continue
                        token_tag = 'GENE'
                    else:
                        #continue
                        token_tag = tag[1]
            token_txt = text[token_spans[i][0]:token_spans[i][1]]
            token_sequence.append(token_txt)
            annotation_sequence.append(token_tag)
            if token_txt =='.' or token_txt =='!' or token_txt =='?':
                token_sequences.append(token_sequence)
                annotations_sequences.append(annotation_sequence)
                token_sequence = []
                annotation_sequence = []
    return token_sequences,annotations_sequences
