import os
import csv
import unicodedata as ud

import tqdm
from nltk.tokenize.util import align_tokens
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk import sent_tokenize
import re

from Abbreviations import AbbreviationProcessor

_treebank_word_tokenizer = TreebankWordTokenizer()
import nltk
def tokenize_fa(documents):
    """
              Tokenization function. Returns list of sequences

              :param documents: list of texts
              :type language: list

              """
    sequences = []
    sequence = []
    for doc in documents:
        if len(sequence) > 0:
            sequences.append(sequence)
        sequence = []
        text = doc
        text = text.replace("\"", "'")
        text = text.replace("`", "'")
        #text = text.replace("``", "")
        text = text.replace("''", "")
        tokens = custom_span_tokenize(text)
        for token in tokens:
            token_txt = text[token[0]:token[1]]
            found = False
            if found == False:
                token_tag = "O"
                # token_tag_type = "O"
            sequence.append((token_txt, token_tag))
            if token_txt == "." or token_txt == "?" or token_txt == "!":
                sequences.append(sequence)
                sequence = []
        sequences.append(sequence)
    return sequences

def custom_span_tokenize(text, language='english', preserve_line=True):
    """
            Returns a spans of tokens in text.

            :param text: text to split into words
            :param language: the model name in the Punkt corpus
            :type language: str
            :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
            :type preserver_line: bool
            """
    tokens = custom_word_tokenize(text)
    tokens = ['"' if tok in ['``', "''"] else tok for tok in tokens]
    return align_tokens(tokens, text)

def custom_word_tokenize(text, language='english', preserve_line=False):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into words
    :param text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
    :type preserver_line: bool
    """
    tokens = []
    sentences = [text] if preserve_line else nltk.sent_tokenize(text, language)
    for sent in sentences:
        for token in _treebank_word_tokenizer.tokenize(sent):
#            if "-" in token:
#                m = re.compile("(\d+)(-)([a-zA-z-]+)")
#                g = m.match(token)
#                if g:
#                    for group in g.groups():
#                        tokens.append(group)
#                else:
#                    tokens.append(token)
            if "/" in token:
                m = re.compile("([a-zA-z0-9]+)(/)([a-zA-z0-9]+)")
                g = m.match(token)
                if g:
                    for group in g.groups():
                        tokens.append(group)
                else:
                    tokens.append(token)
            else:
                tokens.append(token)
    return tokens

def create_folder(folder):
    """
    Creates a folder, if it does not exists.

    :param folder: Folder path to be created
    :return: True if successful, False otherwise
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    return True

def get_files_from_folder(folder):
    """
    A list of files that are inside a given folder

    :param folder:   folder to take files from
    :return:
    """
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return onlyfiles

def read_csv_file(file,delimiter = ',',quotechar = '"'):
    """
    Reads csv or tsv and returns rows
    :param file: file name, this fieald is mandatory
    :param delimiter: delimiter character, optional
    :param quotechar: quote character, optional
    :return: a list of rows
    """
    output_rows = []
    with open(file, newline='',encoding='utf-8') as csvfile:
        abstract_reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        for row in abstract_reader:
            output_rows.append(row)
    yield output_rows
latin_letters = {}

def read_plan_text_dataset(path: str):
    """
    Reads set of files in folder and creates list of documents and list of sentences in that document set
    :param path: Path to the documents
    :return: list of documents and list of sentences in these documents
    """
    files = get_files_from_folder(path)
    output_data = []
    sentences = []
    print('reading files and transforming to sentences')
    for file in tqdm.tqdm(files):
        f = open(path+'/'+file,'r',encoding='utf-8')
        data = f.read()
        output_data.append(data)
        sents = sent_tokenize(data)
        for sent in sents:
            se2 = sent.split('\n')
            for s in se2:
                if s == '':
                    continue
                sentences.append(s)
    return output_data, sentences

def read_plan_text_dataset_into_data_structure(path: str,abbr = False):
    """
    Reads set of files in folder and creates list of documents with ids and sentences
    :param path: Path to the documents
    :param abbr: Whether to expand abbreviations
    :return: list of documents with ids and sentences
    """
    if abbr:
        abbreviation_processor = AbbreviationProcessor()
    files = get_files_from_folder(path)
    data_struc = []
    print('reading files and transforming to sentences')
    for file in tqdm.tqdm(files):
        sentences = []
        f = open(path+'/'+file,'r',encoding='utf-8')
        data = f.read()
        if abbr:
            new_data = abbreviation_processor.expand_abbreviations(data)
        else:
            new_data = data
        sents = sent_tokenize(new_data)
        for sent in sents:
            se2 = sent.split('\n')
            for s in se2:
                if s == '':
                    continue
                sentences.append(s)
        data_struc.append({'id':file,'sentences':sentences})
    return data_struc


def is_latin(uchr):
    """
    Takes one character and checks whether that character is a latin character
    :param uchr: one character
    :return: True if the character is in latin set of 26 letters, False otherwise (note: for special characters,
    such as dot (.) or comma (,), it returns false
    """
    try:
        return latin_letters[uchr]
    except KeyError:
        return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def only_roman_chars(unistr):
    """
    Checks whether the letters in a given string are only latin characters
    :param unistr: string
    :return: True if all characters are latin/roman and False otherwise
    """
    return all(is_latin(uchr)
               for uchr in unistr
               if uchr.isalpha())
