
import random
import csv
import os
import string
import numpy as np

CORPUS_NAME = "all-seasons.csv"
MODULE_PATH = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(MODULE_PATH, CORPUS_NAME)


# Corpus Loading --------------------------------------------------------------
def load_south_park_corpus():
    corpus = []
    with open(CORPUS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpus.append(row)
    
    return [attribute_line(line) for line in corpus] 


def attribute_line(line):
    return "{}".format(line['Line'])
    #return "{}: {}".format(line['Character'], line['Line'])


def clean_corpus(corpus):
    ascii_chars = set(string.printable)
    has_only_ascii = lambda s : all((c in ascii_chars for c in s))
    return [s for s in corpus if has_only_ascii(s)] 
    


# Corpus Expansion ------------------------------------------------------------
def random_line_concat(corpus, concat_p = 0.2, seed = None):
    concat_lines = []
    
    random.seed(seed)
    for line in corpus:
        roll = random.random() 
        if roll < concat_p:
            concat_line = random.choice(corpus)
            concat_lines.append(line + concat_line)
    
    concat_lines.extend(corpus)
    return concat_lines 

def random_line_concat_iterations(corpus, iters = 5, concat_p = 0.2, seed = None):
    concat_corpus = corpus
    for i in range(iters):
        concat_corpus = random_line_concat(concat_corpus, concat_p, seed)
    random.shuffle(concat_corpus)
    return concat_corpus


# Line Operations -------------------------------------------------------------
def split_line_for_training(line, sample_length = None):
    inputs = []
    outputs = []
    
    if sample_length is None:
        for i in range(1,len(line)):
            inputs.append(line[:i])
            outputs.append(line[i])
    else:
        for i in range(len(line) - sample_length):
            inputs.append(line[i:i+sample_length])
            outputs.append(line[i+sample_length])

    return inputs, outputs



# Dataset Operations ----------------------------------------------------------
def convert_to_dataset(corpus, sample_length = None):
    inputs = []
    outputs = []
    for line in corpus:
        line_inputs, line_outputs = split_line_for_training(line, sample_length)
        inputs.extend(line_inputs)
        outputs.extend(line_outputs)
    
    return inputs, outputs


def shuffle_dataset(dataset):
    inputs, outputs = dataset
    dataset = list(zip(inputs, outputs))
    random.shuffle(dataset)
    inputs, outputs = tuple(zip(*dataset))
    return np.asarray(inputs), np.asarray(outputs)


def split_dataset(dataset, train_p = 0.9):
    inputs, outputs = dataset
    num_train = int(len(outputs) * train_p)
    train = (inputs[:num_train], outputs[:num_train])
    test = (inputs[num_train:], outputs[num_train:])
    return train, test



# Utils ------------------------------------------------------------------
def print_dataset_shape(dataset, name = 'Dataset'):
    inputs, outputs = dataset
    print("{} Shapes: {}  -->  {}".format(name, inputs.shape, outputs.shape))




# Deprecated ------------------------------------------------------------------

def concat_entire_corpus(corpus):
    entire_corpus = ""
    for s in corpus:
        entire_corpus += s
    return entire_corpus

def prepare_dataset_for_training(corpus, sample_length):
    inputs = []
    outputs = []
    
    entire_corpus = concat_entire_corpus(corpus)
    for i in range(len(entire_corpus) - sample_length):
        line_input = entire_corpus[i:i+sample_length]
        line_output = entire_corpus[i+sample_length]
        inputs.append(line_input)
        outputs.append(line_output)

    return inputs, outputs









