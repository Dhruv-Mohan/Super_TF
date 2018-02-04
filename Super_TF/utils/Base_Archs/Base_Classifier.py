import tensorflow as tf
from utils.Architect import Architect
from abc import abstractmethod

class Base_Classifier(Architect):
    """Base classification network class, inherited by all classificaiton neural networks """
    Type = 'Classification'

    def __init__(self, kwargs):
        super().__init__()
        self.input_placeholder = tf.placeholder(shape=[None, kwargs['Image_width'] * kwargs['Image_height'] *
                                                       kwargs['Image_cspace']], name='Input')
        self.output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Classes']], name='Output')
        self.build_params = kwargs
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.state_placeholder = tf.placeholder(tf.string, name='State')
        self.output = None

        self.CBE_loss = None

        self.test_dict = None
        self.train_dict = None
        self.IO_dict = None

        self.train_step = None
        self.train_step = None

        self.accuracy = None

    @abstractmethod
    def build_net(self):
        pass

    def construct_control_dict(self, Type='TEST'):
        if Type.upper() is 'TRAIN':
            return{self.dropout_placeholder: self.build_params['Dropout'],\
                   self.state_placeholder: self.build_params['State']}

        elif Type.upper() is 'TEST':
            return{self.dropout_placeholder: 1, \
                   self.state_placeholder: self.build_params['State']}

    def set_output(self):
        self.output = self.build_net()

    def set_accuracy_op(self):
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.output_placeholder, 1))
        false_images = tf.boolean_mask(self.model_dict['Reshaped_input'], tf.logical_not(correct_prediction))
        tf.summary.image(name='False images', tensor=false_images)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def set_train_ops(self, optimizer):
        self.train_step = self.optimizer.minimize(self.CBE_loss, global_step=self.global_step)

    def construct_IO_dict(self, batch):
        return {self.input_placeholder: batch[0], self.output_placeholder: batch[1]}

    def predict(self, kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        predict_io_dict = {self.input_placeholder: kwargs['Input_Im']}
        predict_feed_dict = {**predict_io_dict, **self.test_dict}
        return session.run([self.output], feed_dict=predict_feed_dict)

    def construct_loss(self):
        self.CBE_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_placeholder, logits=self.output))

    def train(self, kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.construct_IO_dict(batch)
        train_feed_dict = {**IO_feed_dict, **self.train_dict}
        session.run([self.train_step], feed_dict=train_feed_dict)

    def test(self, kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.Construct_IO_dict(batch)
        test_feed_dict = {**IO_feed_dict, **self.test_dict}

        if self.accuracy is not None:
            summary, _ = session.run([kwargs['merged'], self.accuracy], feed_dict={test_feed_dict})

        else:
            summary, = session.run([kwargs['merged']], feed_dict={test_feed_dict})