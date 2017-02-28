
import random
import numpy as np
from collections import namedtuple
Dataset = namedtuple('Dataset', ['images', 'labels'])
Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])

def test():
    imgs_a = np.asarray([[111,112], [121,122], [131,132]])
    labels_a = np.asarray([11, 12, 13])
    
    imgs_b = np.asarray([[211,212], [221,222], [231,232]])
    labels_b = np.asarray([21, 22, 23])

    ds_a = Dataset(imgs_a, labels_a)
    ds_b = Dataset(imgs_b, labels_b)
    
    dss_a = Datasets(ds_a, None, None)
    dss_b = Datasets(ds_b, None, None)

    #print("Dataset A: {}".format(ds_a))
    #print("Dataset B: {}".format(ds_b))
    
    ds_c = concat_datasets(ds_a,ds_b)
    print("Concat Dataset Objects")
    print(ds_c)

    dss_c = concat_datasets(dss_a,dss_b)
    print("Concat Datasets Objects")
    print(dss_c)

def concat_datasets(*args):
    
    if len(args) <= 0: return None

    # check if they're all Dataset or Datasets
    if all((type(ds) is Dataset for ds in args)):
        concat_imgs = np.concatenate([images for images,labels in args])
        concat_labels = np.concatenate([labels for images,labels in args])
        return Dataset(concat_imgs, concat_labels)
    elif all((type(ds) is Datasets for ds in args)):
        # they're all Datasets
        def concat_subset(getter):
            return concat_datasets(*[getter(ds) for ds in args if getter(ds) is not None])
        
        trn = concat_subset(lambda x : x.train)
        val = concat_subset(lambda x : x.validation)
        tst = concat_subset(lambda x : x.test)
        return Datasets(trn, val, tst)
    else:
        raise ValueError("Expected arguments to be either all Dataset or all Datasets objects!")

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


if __name__ == "__main__":
    test() 

