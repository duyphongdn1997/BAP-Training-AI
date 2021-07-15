import numpy as np
import io
import re
import tensorflow as tf
import time


def read_text(URL):
    with io.open(URL, 'r', encoding='utf8') as f:
        text = f.read()
    # Character's collection
    return text


def c2i(sub_txt, dict_int):
    """
    This method used to convert character to index mapping
    :param sub_txt: The paragraph that wanted to encoding
    :param dict_int: The vocabulary
    :return: The encode of sub_txt
    """
    encode = np.array([dict_int[c] for c in sub_txt], dtype=np.int32)  # encode data
    return encode


def i2c(arr, dict_char):
    """
    This method used to convert a list of index to list of character
    :param arr: List of index
    :param dict_char: The vocabulary
    :return:
    """
    word = []
    for i in arr:
        word.append(dict_char[i])
    return repr(''.join(word))


# def i2c(sub_int, dict_char):
#    ret = ''
#    for idx in sub_int:
#       ret += dict_char[idx]
#    return ret

def split_input_target(chunk):
    """
    This method used to create the input and target in pair.
    :param chunk:
    :return: The pair input - target
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# Create training examples / targets
def handle_data(data, seq_len):
    """
    this function to create data from row data

    :param data: row data with int type
    :param seq_len: max len of input and output sequence
    :return: data for training
    """
    # data4epoch = len(data) // (seq_len+1)
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    return dataset


def print_data(data, number, dict_char):
    for input_example, target_example in data.take(number):
        print('Input data: ', repr(''.join(i2c(input_example.numpy(), dict_char))))
        print('Target data:', repr(''.join(i2c(target_example.numpy(), dict_char))))


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def lossValue(labels, predicted, batch_size, seq_len):
    # predictedLabel = np.argmax(predicted, axis=2)
    loss = 0
    BATCH_SIZE = batch_size
    seq_len = seq_len

    for i in range(BATCH_SIZE):
        for s in range(seq_len):
            predicted[i, s] = softmax(predicted[i, s])
            loss += np.log(predicted[i, s, labels[i, s]])

    return -loss / (BATCH_SIZE * seq_len)


def built_model(cellType, vocab_size, embedding_dim, rnn_units, BATCH_SIZE):

    if cellType == "LSTM":
        print("Using CuDNNLSTM")
    elif cellType == "GRU":
        print("Using CuDNNGRU")
    else:
        print("Using SimpleRNN")
        rnn = tf.keras.layers.SimpleRNN

    # define model
    model = "Model here in"
    return model
