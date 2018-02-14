import tensorflow as tf
from utils.Architect import Architect
from abc import abstractmethod

class Base_Segnet(Architect):
    """Base Segmentation network class, inherited by segmentation networks"""
    Type = 'Segmentation'

    def __init__(self, kwargs):
        super().__init__()
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Image_width'], kwargs['Image_height'],
                                                       kwargs['Image_cspace']], name='Input')
        self.output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Image_width'], kwargs['Image_height'],
                                                       kwargs['Classes']], name='Output')
        if kwargs['Classes'] is 1:
            self.sigmoid_loss = True
        else:
            self.sigmoid_loss = False

        self.build_params = kwargs
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.state_placeholder = tf.placeholder(tf.string, name='State')
        self.output = None

        self.loss = []

        self.train_step = None

        self.accuracy = None

    @abstractmethod
    def build_net(self):
        pass

    def construct_control_dict(self, Type='TEST'):
        if Type.upper() in 'TRAIN':
            return {self.dropout_placeholder: self.build_params['Dropout'], self.state_placeholder: self.build_params['State']}

        elif Type.upper() in 'TEST':
            return {self.dropout_placeholder: 1, self.state_placeholder: self.build_params['State']}

    def set_output(self):
        self.output = self.build_net()

    def set_accuracy_op(self):
        #onus of pre-proc probs lies on the network
        intersection = tf.reduce_sum(self.output * self.output_placeholder, axis=1) #per instance
        union = tf.reduce_sum(self.output, axis=1) + tf.reduce_sum(self.output_placeholder, axis=1)
        self.accuracy = tf.reduce_mean((2*intersection)/union)
        tf.summary.scalar('Dice_Coeff', self.accuracy)

    def set_train_ops(self, optimizer):
        loss = tf.add_n(self.loss, 'Loss_accu')
        self.train_step = optimizer.minimize(loss, global_step=self.global_step)

    def construct_IO_dict(self, batch):
        return {self.input_placeholder: batch[0], self.output_placeholder: batch[1]} #Input image and output image

    def predict(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        predict_io_dict = {self.input_placeholder: kwargs['Input_Im']}
        predict_feed_dict = {**predict_io_dict, **self.test_dict}
        return session.run([self.output], feed_dict=predict_feed_dict)

    def construct_loss(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

