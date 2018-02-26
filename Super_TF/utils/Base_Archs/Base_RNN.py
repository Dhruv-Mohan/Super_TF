import tensorflow as tf
from utils.Architect import Architect
from abc import abstractmethod

class Base_RNN(object):
    """Base RNN network class, inherited by all rnn neural networks"""
    Type = 'Sequence'

    def __init__(self, kwargs):
        super().__init__()
        self.input_seq_placeholder = tf.placeholder(tf.int32, shape=[None, kwargs['Padded_length']], name='Input_Seq')
        self.target_seq_placeholder = tf.placeholder(tf.int32, shape=[None, kwargs['Padded_length']], name='Target_Seq')
        self.mask_placeholder = tf.placeholder(tf.int32, shape=[None, kwargs['Padded_length']], name='Seq_Mask')
        self.build_params = kwargs
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.state_placeholder = tf.placeholder(tf.string, name='State')
        self.output = None
        self.loss = []

    @abstractmethod
    def build_net(self):
        pass

    def set_output(self):
        self.output = self.build_net()