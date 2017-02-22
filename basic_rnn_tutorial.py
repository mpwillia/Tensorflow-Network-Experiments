
"""
Following Tutorial Located At:
    http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
"""

import numpy as np
from random import shuffle
import tensorflow as tf

def main():

    dataset = make_dataset()
    train_data, test_data = split_dataset(dataset)

    test_rnn(train_data, test_data)

def test_rnn(train_data, test_data):
    
    train_input, train_output = train_data
    test_input, test_output = test_data

    data = tf.placeholder(tf.float32, [None, 20,1])
    target = tf.placeholder(tf.float32, [None, 21])
    
    num_hidden = 24
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

    val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    
    print("Starting train")
    batch_size = 1000
    no_of_batches = int(len(train_input)/batch_size)
    epoch = 1000
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(minimize,{data: inp, target: out})
        print "Epoch - ",str(i)
    incorrect = sess.run(error,{data: test_input, target: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

    print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
    sess.close()

def split_dataset(dataset, num_examples = 10000):
    inputs, outputs = dataset
    train = (inputs[:num_examples], outputs[:num_examples])
    test = (inputs[num_examples:], outputs[num_examples:])
    return train, test


def make_dataset():
 
    train_input = ['{0:020b}'.format(i) for i in range(2**20)]
    shuffle(train_input)
    train_input = [map(int,i) for i in train_input]
    ti  = []
    for i in train_input:
        temp_list = []
        for j in i:
                temp_list.append([j])
        ti.append(np.array(temp_list))
    train_input = ti

    train_output = []
     
    for i in train_input:
        count = 0
        for j in i:
            if j[0] == 1:
                count+=1
        temp_list = ([0]*21)
        temp_list[count]=1
        train_output.append(temp_list)
    
    return train_input, train_output

if __name__ == "__main__":
    main()
