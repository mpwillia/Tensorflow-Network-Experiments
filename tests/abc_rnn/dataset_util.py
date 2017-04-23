import numpy as np
import random
import string
from pprint import pprint
from char_encoding import Encoding

def create_abc_dataset(size = None, seq_len = 2, pos_inflation = 5, train_p = 0.9, neg_to_pos = 2.0):
    null_term = '\0'
    ordered_alphabet = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)
    wrap_alphabet = ordered_alphabet + ordered_alphabet[:seq_len]
    alphabet = set(ordered_alphabet)

    if pos_inflation is None or pos_inflation <= 0:
        pos_inflation = 1
    
    print("Alphabet:")
    print(ordered_alphabet)
    print

    pos = []      
    neg = []

    for i in range(len(alphabet)):
        input_str = wrap_alphabet[i:i+seq_len]
        output_char = wrap_alphabet[i+seq_len]
        pos.append((input_str, output_char))

        rev_input_str = list(reversed(input_str[1:] + [output_char]))
        rev_output_char = input_str[0] 
        pos.append((rev_input_str, rev_output_char))

        for perm in permute(input_str[0], ordered_alphabet, seq_len):
            perm = list(perm)
            if perm != input_str and perm != rev_input_str:
                neg.append((perm, null_term))

    
    print("Positive Example") 
    pprint(pos[:2])
    print
    print("Negative Example") 
    pprint(neg[:2])
    print


    # pos inflate
    pos_inf = []
    for i in range(pos_inflation):
        pos_inf.extend(pos)  
   

    print("Base Pos Samples : {}".format(len(pos)))
    print("Infl Pos Samples : {}".format(len(pos_inf))) 

    print("Base Neg Samples : {}".format(len(neg))) 
    if neg_to_pos is not None:
        neg = random.sample(neg, int(len(pos_inf)*neg_to_pos))
        print("Neg to Pos Samples : {}".format(len(neg))) 
    
    print
    print("Final Pos Samples : {}".format(len(pos_inf)))
    print("Final Neg Samples : {}".format(len(neg)))
    
    dataset = pos_inf + neg
    
    if size is not None:
        print("\nTotal Dataset Size  : {}".format(len(dataset)))
        dataset = dataset[:size]   
        print("Final Dataset Size  : {}\n".format(len(dataset)))
    else:
        print("\nDataset Size  : {}\n".format(len(dataset)))

    inputs, outputs = zip(*dataset)
    return (inputs, outputs), Encoding(dataset)


def permute(s, choices, seq_len):
    for c in choices:
        perm = s + c
        if len(perm) < seq_len:
            for perm in permute(perm, choices, seq_len):
                yield perm
        else:
            yield perm



def encode_dataset(dataset, encoding):
    inputs, outputs = dataset
    inputs = np.asarray([encoding.encode_str(s) for s in inputs])
    outputs = np.asarray([encoding.encode_one_hot(c) for c in outputs])
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


def print_dataset_shape(dataset, name = 'Dataset'):
    inputs, outputs = dataset
    print("{} Shapes: {}  -->  {}".format(name, inputs.shape, outputs.shape))

