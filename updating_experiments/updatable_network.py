

import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl

from neural_network import Network

from dataset_util import split_by_class

class UpdatableNetwork(Network):
    def __init__(self, *args, **kwargs):
        super(UpdatableNetwork, self).__init__(*args, **kwargs)


    def fit(self, train_data, optimizer, loss, epochs, **kwargs):
        args = (train_data, optimizer, loss, epochs)
        fit_return = super(UpdatableNetwork, self).fit(*args, **kwargs)
    
        return fit_return

    def find_representative_examples(train_data):
        pass
         
    
