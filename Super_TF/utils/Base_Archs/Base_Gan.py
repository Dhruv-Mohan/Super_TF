import tensorflow as tf
from utils.Architect import Architect
from abc import abstractmethod

class Base_Gan(Architect):
    """Base gan class, inherited by all Gan models"""
    Type = 'GAN'
    def __init__(self, kwargs):
        super().__init__()
        self.gen_input_placeholder = tf.placeholder(tf.float32, \
            shape=[None, kwargs['Image_width'], kwargs['Image_height'], kwargs['Image_cspace']], name='Input')
        self.build_params = kwargs
        self.gen_dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.gen_state_placeholder = tf.placeholder(tf.string, name="State")
        self.dis_dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.dis_state_placeholder = tf.placeholder(tf.string, name="State")

    @abstractmethod
    def generator(self, kwargs):
        pass

    @abstractmethod
    def discriminator(self, kwargs):
        pass


