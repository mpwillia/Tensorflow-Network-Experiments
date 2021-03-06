
#try:
#    import cPickle as pickle
#except:
#    import pickle
import dill as pickle
import os

def save(net, name = None, path = './'):
    if name is None:
        name = net.network_name
    
    vars_name = name + ".ckpt"
    net_name = name + ".pkl"

    vars_path = os.path.join(path, vars_name)
    net_path = os.path.join(path, net_name)
    
    print("Saving Network")
    net.save_variables(vars_path)
    
    with open(net_path, 'w') as f:
        pickle.dump(net, f)


def load(name, path = './'):
    vars_name = name + ".ckpt"
    net_name = name + ".pkl"

    vars_path = os.path.join(path, vars_name)
    net_path = os.path.join(path, net_name)
    

    with open(net_path, 'r') as f:
        print("Loading Network")
        net = pickle.load(f)
    
    net.load_variables(vars_path)

    return net


