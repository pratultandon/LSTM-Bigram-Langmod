"""
rnn_lm.py

Implementation of an RNN LSTM Cell based Language Model, with Embeddings.
"""
from __future__ import division

import numpy as np
import tensorflow as tf

RAW_FILENAME = "huckfin.txt"

class RNNLM():
    def __init__(self, embedding_size, lstm_size, num_steps, vocab_size, batch_size, learning_rate):
        """
        Instantiate an RNN Language Model, with the necessary hyperparameters and other
        relevant variables.

        :param embedding_size: Size of the word embeddings.
        :param lstm_size: Size of the LSTM cell.
        :param num_steps: Size of input window.
        :param vocab_size: Size of the vocabulary. 
        :param learning_rate: Learning Rate.
        :param batch_size: Batch Size.
        """
        # Initialize hyperparameters and other useful variables
        self.embedding_size, self.lstm_size, self.num_steps = embedding_size, lstm_size, num_steps
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, [self.batch_size, num_steps])
        self.Y = tf.placeholder(tf.int32, [self.batch_size, num_steps])
        self.keep_prob = tf.placeholder(tf.float32)

        # Instantiate Network Parameters
        self.instantiate_weights()

        # Build the Logits
        self.logits, self.new_state = self.inference()

        # Build the Loss Computation
        self.loss_val = self.loss()

        # Build the Training Operation
        self.train_op = self.train()

    def instantiate_weights(self):
        """
        Instantiate the network Variables, for the Embedding, LSTM, and Output Layers.
        """
        # Embedding
        self.E = self.weight_variable([self.vocab_size, self.embedding_size], 'E')

        # LSTM Cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=0.0, state_is_tuple=True)
        self.lstm = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        self.initial_state = self.lstm.zero_state(self.batch_size, tf.float32)

        # Output Layer
        self.output_W = self.weight_variable([self.lstm_size, self.vocab_size], 'Output_W')
        self.output_B = self.bias_variable([self.vocab_size], 'Output_B')

    def inference(self):
        """
        Build the inference computation graph for the model, going from the input to the output
        logits (before final softmax activation).
        """
        # Get embeddings
        emb = tf.nn.embedding_lookup(self.E, self.X)

        # Apply dropout (Relevant only for training)
        emb_dropout = tf.nn.dropout(emb, self.keep_prob)

        # Get outputs from LSTM
        outputs, state = tf.nn.dynamic_rnn(self.lstm, emb_dropout, initial_state = self.initial_state)

        # Get logits
        logits = tf.matmul(tf.reshape(outputs, [self.batch_size*self.num_steps, self.lstm_size]), self.output_W) + self.output_B
        return logits, state

    def loss(self):
        """
        Build the cross-entropy loss computation graph.
        """
        loss_value = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.Y, [-1])], [tf.ones([self.batch_size * self.num_steps])])
        return tf.reduce_sum(loss_value)/self.batch_size

    def train(self):
        """
        Build the training operation, using the cross-entropy loss and an Adam Optimizer.
        """
        opt = tf.train.AdamOptimizer(self.learning_rate)
        return opt.minimize(self.loss_val)

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial, name=name)

def read(raw_filename):
    """
    Read and parse the file, building the vectorized representations of the data.
    The first 90 percent of the file is uses for training, and the rest for testing.

    :param raw_filename: Filename of Project Gutenburg .txt file
    :return: Tuple of train_data, test_data, and vocab_size
    """
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    import nltk as nltk
    decoded_file = open(raw_filename).read().decode('utf8')
    tokenized_file = nltk.word_tokenize(decoded_file)

    split_point = int(len(tokenized_file)*0.9)
    train_data = tokenized_file[0:split_point]
    test_data = tokenized_file[split_point:]

    from collections import defaultdict
    freq = defaultdict(int)
    for tok in train_data:
        if tok not in vocabulary:
            vocabulary[tok] = vocab_size
            vocab_size += 1
        freq[tok] += 1
    vocabulary["*UNK*"] = vocab_size
    vocab_size += 1

    train_data_temp = []
    for tok in train_data:
        if freq[tok] <= 5:
            train_data_temp.append("*UNK*")
        else:
            train_data_temp.append(tok)
    train_data = train_data_temp

    test_data_temp = []
    for tok in test_data:
        if tok in vocabulary and freq[tok] >= 5:
            test_data_temp.append(tok)
        else:
            test_data_temp.append("*UNK*")
    test_data = test_data_temp

    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)

    # Vectorize, and return output tuple.
    train_data = map(lambda x: vocabulary[x], train_data)
    test_data = map(lambda x: vocabulary[x], test_data)
    return np.array(train_data), np.array(test_data), vocab_size

def batch_producer(raw_data, batch_size, num_steps):
    """
    Helper method to generate and return tuples for x, y that are passed into the model using
    the feed_dict for any testing/training operation.

    :param raw_data: Data to be split into batches.
    :param batch_size: Size of a batch.
    :param num_steps: Size of I/O window for our model.
    :return: A generator for tuples (x,y) that are passed into our model. (Covering all of the raw data)
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    # Create generator
    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)
  
# Main Training and Testing Block
if __name__ == "__main__":
    # Create vocabulary, vectorize training and testing data
    train_data, test_data, voc_sz = read(RAW_FILENAME)
    bsz = 30
    # Launch Tensorflow Session
    print 'Launching Session!'
    with tf.Session() as sess:
        # Instantiate Model
        model = RNNLM(50, 256, 20, voc_sz, bsz, 1e-4)
        # Initialize all Variables
        sess.run(tf.initialize_all_variables())
        print 'Starting Training!'
        for i in range(20):
            train_generator = batch_producer(train_data, bsz, model.num_steps)
            loss, counter = 0.0, 0
            for x, y in train_generator:
                curr_loss, new_state, _ = sess.run([model.loss_val, model.new_state, model.train_op],
                                        feed_dict={model.X: x,
                                                   model.Y: y,
                                                   model.keep_prob: 0.5})
                loss, counter = loss + curr_loss, counter + model.num_steps
                if counter % 100 == 0:
                    print 'Counter {} Train Perplexity:'.format(counter), np.exp(loss/counter), curr_loss
                model.initial_state = new_state
        print 'Starting Testing!'
        test_generator = batch_producer(test_data, bsz, model.num_steps)
        loss1, counter1 = 0.0, 0
        for x, y in test_generator:
            curr_loss, new_state = sess.run([model.loss_val, model.new_state],
                                    feed_dict={model.X: x,
                                               model.Y: y,
                                               model.keep_prob:1})
            
            loss1, counter1 = loss1 + curr_loss, counter1 + model.num_steps
            if counter1 % 100 == 0:
                print 'Counter {} Test Perplexity:'.format(counter1), np.exp(loss1/counter1), curr_loss
            model.initial_state = new_state

