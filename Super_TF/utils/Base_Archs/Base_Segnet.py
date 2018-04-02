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
            self.sigmoid = True
        else:
            self.sigmoid = False

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
        output = tf.reshape(self.output, shape=(-1, self.build_params['Image_width'] * self.build_params['Image_height']))
        output_placeholder = tf.reshape(self.output_placeholder, shape=(-1, self.build_params['Image_width'] * self.build_params['Image_height']))
        sigmoid = tf.nn.sigmoid(output)
        sigmoid = tf.ceil(sigmoid- 0.5 + 1e-10)
        intersection = tf.reduce_sum(sigmoid * output_placeholder, axis=1) + 1e-10 #per instance 
        union = tf.reduce_sum(sigmoid, axis=1) + tf.reduce_sum(output_placeholder, axis=1) + 1e-10
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
        #default loss is dice loss
        if self.output is None:
            self.set_output()
        output = tf.reshape(self.output, shape=(-1, self.build_params['Image_width'] * self.build_params['Image_height']))
        eps = tf.constant(1e-5, name='Eps')
        output = tf.nn.sigmoid(output + eps)
        output_placeholder = tf.reshape(self.output_placeholder, shape=(-1, self.build_params['Image_width'] * self.build_params['Image_height']))
        intersection = tf.reduce_sum(output*output_placeholder, axis=1) + 1e-5
        union = tf.reduce_sum(output, axis=1) + tf.reduce_sum(output_placeholder, axis=1) + 1e-5
        D_C = 1 - (2*intersection) / union
        D_L = tf.reduce_mean(D_C)
        tf.summary.scalar('Dice_loss', D_L)
        self.loss.append(D_L)
        tf.summary.image(name='Input image', tensor=self.input_placeholder)
        tf.summary.image(name='Mask', tensor=self.output_placeholder)
        tf.summary.image(name='Output', tensor=tf.nn.sigmoid(self.output))

    def train(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']
        batch = kwargs['data']
        #batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.construct_IO_dict(batch)
        train_dict = self.construct_control_dict(Type='Train')
        train_feed_dict = {**IO_feed_dict, **train_dict}
        session.run([self.train_step], feed_dict=train_feed_dict)

    def test(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']
        batch = kwargs['data']
        #batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.construct_IO_dict(batch)
        test_dict = self.construct_control_dict(Type='Test')
        test_feed_dict = {**IO_feed_dict, **test_dict}

        if self.accuracy is not None:
            summary, _ = session.run([kwargs['merged'], self.accuracy], feed_dict=test_feed_dict)

        else:
            summary = session.run([kwargs['merged']], feed_dict=test_feed_dict)[0]

        return summary

