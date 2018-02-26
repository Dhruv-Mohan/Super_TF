import tensorflow as tf
from utils.Architect import Architect
from abc import abstractmethod

class Base_RNN(Architect):
    """Base RNN network class, inherited by all rnn neural networks"""
    Type = 'Sequence'

    def __init__(self, kwargs):
        super().__init__()
        self.input_placeholder = None
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
        self.Predict_op = tf.argmax(self.output, 1)

    def set_accuracy_op(self):
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.reshape(tf.cast(tf.reshape(self.target_seq_placeholder, shape=[-1]), tf.int64), [-1]))
        pre_acc = tf.to_float(correct_prediction) * tf.to_float(tf.reshape(self.mask_placeholder, [-1]))
        pre_acc = tf.reduce_sum(pre_acc)
        self.accuracy = tf.div(pre_acc,  tf.maximum(1.0,tf.reduce_sum(tf.to_float(tf.reshape(self.mask_placeholder, [-1])))))
        tf.reduce_sum(tf.to_float(tf.reshape(self.mask_placeholder, [-1])))
        tf.summary.scalar('accuracy', self.accuracy)

    def construct_IO_dict(self, batch):
        return {self.input_placeholder: batch[0], self.input_seq_placeholder: batch[1], self.target_seq_placeholder: batch[2], self.mask_placeholder: batch[3]}

    def construct_control_dict(self, Type='TEST'):
        if Type.upper() in 'TRAIN':
            return {self.dropout_placeholder: self.build_params['Dropout'], self.state_placeholder: self.build_params['State']}

        elif Type.upper() in 'TEST':
            return {self.dropout_placeholder: 1, self.state_placeholder: self.build_params['State']}

    def set_train_ops(self, optimizer):
        loss = tf.add_n(self.loss, 'Loss_accu')
        self.train_step = optimizer.minimize(loss, global_step=self.global_step)

    def predict(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        predict_io_dict = {self.input_placeholder: kwargs['Input_Im']}
        predict_feed_dict = {**predict_io_dict, **self.test_dict}
        return session.run([self.output], feed_dict=predict_feed_dict)


    def train(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.construct_IO_dict(batch)
        train_dict = self.construct_control_dict(Type='Train')
        train_feed_dict = {**IO_feed_dict, **train_dict}
        _, out = session.run([self.train_step, self.Predict_op], feed_dict=train_feed_dict)
        print(out)

    def test(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.construct_IO_dict(batch)
        test_dict = self.construct_control_dict(Type='Test')
        test_feed_dict = {**IO_feed_dict, **test_dict}

        if self.accuracy is not None:
            summary, _ = session.run([kwargs['merged'], self.accuracy], feed_dict=test_feed_dict)

        else:
            summary = session.run([kwargs['merged']], feed_dict=test_feed_dict)[0]

        return summary
