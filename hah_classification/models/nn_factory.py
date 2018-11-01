import tensorflow as tf
from tensorflow.contrib import rnn


def init_embedding(vocab_size, embedding_size, embedding_path=None, name='embedding'):
    embedding = tf.get_variable(name, [vocab_size, embedding_size])
    return embedding


def rnn_factory(num_units, layer_num, cell_type='lstm', input_keep_prob=1.0, output_keep_prob=1.0):
    if cell_type.lower() == 'lstm':
        cell_func = rnn.BasicLSTMCell
    elif cell_type.lower() == 'gru':
        cell_func = rnn.GRUCell

    else:
        cell_func = rnn.RNNCell

    cells = [cell_func(num_units) for _ in range(layer_num)]
    drop_func = lambda cell: rnn.DropoutWrapper(cell,
                                                input_keep_prob=input_keep_prob,
                                                output_keep_prob=output_keep_prob)
    cell = rnn.MultiRNNCell(list(map(drop_func, cells)))
    return cell

def muti_layer_conv(inputs, filter_num, skip=1, layer_num=1):
    out = inputs
    for i in range(layer_num):
        with tf.variable_scope('CNN_%s'%(i+1)):
            out = tf.layers.conv1d(out, filter_num, skip, padding='same')
            if i > 0:
                out = inputs + out

    # out = tf.tanh(out)
    return out


def muti_class_attention(inputs, class_num, concat_inputs=None):
    attention = tf.nn.softmax(tf.layers.conv1d(inputs, class_num, 1, padding='same', activation=tf.nn.relu), axis=2)
    inputs_reshape = tf.transpose(inputs, [0, 2, 1])
    attention_out = tf.matmul(inputs_reshape, attention)
    outputs = []
    for output in tf.unstack(attention_out, axis=2):
        if concat_inputs is not None:
            output = tf.concat([output, concat_inputs], axis=1)
        outputs.append(output)

    return outputs


def fc(inputs, class_num):
    # fc = tf.layers.dense(inputs, class_num, activation=tf.nn.relu)
    output = tf.layers.dense(inputs, class_num)
    return output
