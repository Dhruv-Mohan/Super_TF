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


    def set_accuracy_op(self):
        pass


    def set_train_ops(self):
        pass

    def construct_IO_dict(self, batch):
        return  {self.input_placeholder: batch[0], self.dis_class_placeholder: batch[1], self.gen_class_placeholder:
            self.gen_random_lab(batch[1])}

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



    def train(self):
        pass


    def test(self):
        pass