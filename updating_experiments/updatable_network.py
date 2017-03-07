

import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl

from neural_network.network_util import batch_dataset
from neural_network import Network

from dataset_util import split_by_class, concat_datasets, sample_dataset
from dataset_util import Dataset

class UpdatableNetwork(Network):
    def __init__(self, input_shape, layers, update_method = 'sample_previous', 
                 recall_p = 0.25, *args, **kwargs):
    #def __init__(self, *args, **kwargs):
        super(UpdatableNetwork, self).__init__(input_shape, layers, *args, **kwargs)
        self.update_method = update_method
        
        self.known_train_data = None
        self.recall_p = recall_p
        if self.recall_p < 0: 
            self.recall_p = 0.0
        elif self.recall_p >= 1.0:
            raise ValueError("Recall percentage cannot be greater than or equal to 1.0")
        print("Update Method: {}".format(self.update_method))
        print("Recall Percent : {:7.2%}".format(self.recall_p))
    
    def close(self):
        super(UpdatableNetwork, self).close()
        self.known_train_data = None

    def set_recall_p(self, recall_p):
        self.recall_p = recall_p  

    def fit(self, train_data, optimizer, loss, epochs, **kwargs):
        args = (train_data, optimizer, loss, epochs)
        fit_return = super(UpdatableNetwork, self).fit(*args, **kwargs)
        
        if self.update_method is not None:
            if self.update_method == 'sample_previous':
                train_data = Dataset(*self._reshape_dataset(train_data)) 
                print("Storing training data!")
                if self.known_train_data is None:
                    self.known_train_data = train_data 
                else:
                    self.known_train_data = concat_datasets(self.known_train_data, train_data)
                print("Stored training data size: {:d}".format(len(self.known_train_data.labels)))

        return fit_return
    
    def _batch_for_train(self, dataset, batch_size, include_progress = False):
        if self.known_train_data is None:
            return batch_dataset(dataset, batch_size, include_progress)
        else: 
            recall_size = int(batch_size * self.recall_p)
            if recall_size <= 0 and self.recall_p > 0.0:
                recall_size = 1

            actual_batch_size = batch_size - recall_size
            if actual_batch_size < 1:
                actual_batch_size = 1
            recall_size = batch_size - actual_batch_size

            if recall_size <= 0:
                return batch_dataset(dataset, batch_size, include_progress)
            else:
                return self._batch_with_known(dataset, actual_batch_size, recall_size, include_progress)
    
    def _batch_with_known(self, dataset, batch_size, recall_size, include_progress = False):
        for batch in batch_dataset(dataset, batch_size, include_progress):
            chunk_dataset = Dataset(batch[0], batch[1])
            extra = tuple(batch[2:])
            known_dataset = sample_dataset(self.known_train_data, recall_size)
            recall_dataset = concat_datasets(chunk_dataset, known_dataset)
            yield (recall_dataset.images, recall_dataset.labels) + extra

    def _run_training_epoch(self, *args, **kwargs):
        super(UpdatableNetwork, self)._run_training_epoch(*args, **kwargs)

    def find_representative_examples(train_data):
        pass
         
    
