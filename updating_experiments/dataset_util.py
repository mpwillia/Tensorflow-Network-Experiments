
import random
import numpy as np
from collections import namedtuple
Dataset = namedtuple('Dataset', ['images', 'labels'])
Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])

def split_by_class(dataset):
    classes = set((one_hot_to_label(ohl) for ohl in dataset.labels))
    by_class = []
    for label in sorted(list(classes)):
        by_class.append(filter_dataset(dataset, [label])) 
    return by_class

def sample_dataset(dataset, num_samples, seed = None):
    images, labels = split_dataset(dataset)
    zipped = list(zip(images, labels))
    samples = random.sample(zipped, num_samples)
    sample_images, sample_labels = tuple(zip(*samples))
    
    print("Sample Size: {:d}".format(len(sample_labels)))

    return Dataset(np.asarray(sample_images), np.asarray(sample_labels)) 

def filter_datasets(datasets, labels = None):
    if labels is None: return None
    labels = set(labels)

    trn_filtered = filter_dataset(datasets.train)
    val_filtered = filter_dataset(datasets.validation)
    tst_filtered = filter_dataset(datasets.test)
    
    return Datasets(trn_filtered, val_filtered, tst_filtered)

def filter_dataset(dataset, labels):
    labels = set(labels)
    dataset_images, dataset_ohl = split_dataset(dataset)
    dataset_labels = one_hot_to_label(dataset_ohl)
    dataset_zipped = list(zip(dataset_images, dataset_ohl, dataset_labels))
    dataset_filtered = [(img, ohl) for img, ohl, label in dataset_zipped if label in labels] 
    filt_images, filt_ohl = zip(*dataset_filtered)
    return Dataset(np.asarray(filt_images), np.asarray(filt_ohl))

def one_hot_to_label(one_hot_encoding):
    return np.argmax(one_hot_encoding, axis = 1)


def split_dataset(dataset):
    return dataset.images, dataset.labels

