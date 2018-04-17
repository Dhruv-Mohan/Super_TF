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
        self.gen_dropout_prob_placeholder = tf.placeholder(tf.float32, name='Gen_Dropout')
        self.gan_state_placeholder = tf.placeholder(tf.bool, name="State")
        self.dis_dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dis_Dropout')

    @abstractmethod
    def generator(self, kwargs):
        pass

    @abstractmethod
    def discriminator(self, kwargs):
        pass

    def construct_control_dict(self, Type='TEST'):
        if Type.upper() in 'TRAIN':
            return {self.gen_dropout_prob_placeholder: self.build_params['Gen_Dropout'],\
                self.dis_dropout_prob_placeholder: self.build_params['Dis_Dropout'],\
                self.gan_state_placeholder: True}

        elif Type.upper() in 'TEST':
            return {self.gen_dropout_prob_placeholder: 1,\
                self.dis_dropout_prob_placeholder: 1,\
                self.gan_state_placeholder: False}
    
    def Construct_IO_dict(self):
        pass