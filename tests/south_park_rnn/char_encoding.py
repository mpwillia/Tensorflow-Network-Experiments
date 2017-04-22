
import numpy as np
import string


class Encoding(object):
    def __init__(self, data = None):
        
        self.alphabet = set(string.printable) - set(['\t', '\r', '\b', '\x0b', '\x0c'])
        if data is not None:
            self.alphabet = set()
            def find_strs(data):
                for item in data:
                    if type(item) == str:
                        self.alphabet.update(list(item))
                    else:
                        find_strs(item)
            find_strs(data)

        self.encode_map = dict()
        self.decode_map = dict()

        for val, char in enumerate(self._ordered_alphabet()):
            self.encode_map[char] = val
            self.decode_map[val] = char
    
    def _ordered_alphabet(self):
        return sorted(list(self.alphabet))


    # Encoding ----------------------------------------------------------------
    def encode_char(self, c):
        return self.encode_map[c]

    def encode_str(self, s):
        return [self.encode_map[c] for c in s]

    def encode(self, s):
        if len(s) == 1: 
            return self.encode_char(s)
        else:
            return self.encode_str(s)


    # Decoding ----------------------------------------------------------------
    def decode(self, val):
        return self.decode_map[val]


    # One Hot Encoding ----------------------------------------------------------------
    def encode_char_one_hot(self, c):
        val = self.encode_char(c)
        one_hot = np.zeros(len(self.alphabet))
        one_hot[self.encode_char(c)] = 1
        return one_hot

    def encode_str_one_hot(self, s):
        return np.asarray([self.encode_char_one_hot(c) for c in s])

    def encode_one_hot(self, s):
        if len(s) == 1:
            return self.encode_char_one_hot(s)
        else:
            return self.encode_str_one_hot(s)

    def __str__(self):
        return str(self._ordered_alphabet())

    def __len__(self):
        return len(self.alphabet)


def get_corpus_alphabet(corpus):
    #alphabet = set(string.ascii_letters)
    alphabet = set()
    for line in corpus:
        alphabet.update(list(line))
    return alphabet
    

def get_corpus_char_encoding(corpus):
    #alphabet = set(string.ascii_letters)
    alphabet = set()
    for line in corpus:
        alphabet.update(list(line))
    
    encode = dict()
    decode = dict()
    for val, char in enumerate(alphabet):
        encode[char] = val
        decode[val] = char
    
    return encode, decode



def char_to_one_hot(c, encoding):
    one_hot = np.zeros(len(encoding))
    one_hot[encoding[c]] = 1
    return np.asarray(one_hot)

def encode_str(s, encoding):
    return [encoding[c] for c in s]

def encode_char(c, encoding):
    return encoding[c] 



