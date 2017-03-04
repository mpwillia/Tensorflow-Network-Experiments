

from ..common import fit_net, get_mnist_fit_params
from ...dataset_util import filter_dataset, split_dataset
from ...dataset_util import Dataset, Datasets


def run(mnist, net):
    
    #mnist = setup_drip_datasets(mnist)

    val_data = split_dataset(mnist.validation)
    test_data = split_dataset(mnist.test)

    epochs = 1
    verbose = True
    kwargs = get_mnist_fit_params()
    kwargs['verbose'] = verbose
   
    for label in range(10):
        print("Training for just label {}".format(label))
        label_trn_data = filter_dataset(mnist.train, [label])
        net, label_fit_results = fit_net(net, epochs, label_trn_data, val_data, test_data, **kwargs)

    
    print("Finished")
    return label_fit_results


def setup_drip_datasets(datasets):
    
    trn_by_label = []
    for label in range(10):
        trn_by_label.append(filter_dataset(datasets.train, [label])) 

    val = datasets.validation
    tst = datasets.test
    
    return Dataset(trn_by_label)


