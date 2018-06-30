import tensorflow as tf
from utils.Architect import Architect
from abc import abstractmethod
import sys
from tensorflow.python.platform import tf_logging as logging
_slim_path = '/media/nvme/tfslim/models/research/slim'
sys.path.append(_slim_path)
slim = tf.contrib.slim

class Base_Classifier(Architect):
    """Base classification network class, inherited by all classification neural networks """
    Type = 'Classification'

    def __init__(self, kwargs):
        super().__init__()
        #self.input_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Image_height'], kwargs['Image_width'],kwargs['Image_cspace']], name='Input')
        #self.output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Classes']], name='Output')
        self.build_params = kwargs
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.state_placeholder = tf.placeholder(tf.string, name='State')
        self.output = None

        self.loss = []

        self.train_step = None
        self.iter = self.build_params['iter']
        self.handle = self.build_params['handle']
        self.train_iter = self.build_params['train_iter']
        self.test_iter = self.build_params['test_iter']
        self.input_placeholder , self.output_placeholder = self.train_iter.get_next()
        self.accuracy = tf.Variable(0.0, trainable=False)
        self.Init_test = False
        self.Init_train = False
        self.training = self.build_params['Training']
    @abstractmethod
    def build_net(self):
        pass

    def construct_control_dict(self, Type='TEST'):
        if Type.upper() in 'TRAIN':
            return {self.dropout_placeholder: self.build_params['Dropout'], self.state_placeholder: self.build_params['State'], self.handle:self.train_handle}

        elif Type.upper() in 'TEST':
            return {self.dropout_placeholder: 1, self.state_placeholder: self.build_params['State'], self.handle:self.test_handle}

    def set_output(self):
        self.output = self.build_net()

    def set_accuracy_op(self):
        acc_decay = 0.99
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.output_placeholder, 1))
        false_images = tf.boolean_mask(self.input_placeholder, tf.logical_not(correct_prediction))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.image(name='False images', tensor=false_images)
        tf.summary.scalar('accuracy', self.accuracy.assign(acc_decay * self.accuracy + (1 - acc_decay) * accuracy))

    def set_train_ops(self, optimizer):
        self.loss = tf.add_n(self.loss, 'Loss_accu')
        self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

    def construct_IO_dict(self, batch):
        return {self.input_placeholder: batch[0], self.output_placeholder: batch[1]}

    def predict(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        predict_io_dict = {self.input_placeholder: kwargs['Input_Im']}
        predict_feed_dict = {**predict_io_dict, **self.test_dict}
        return session.run([self.output], feed_dict=predict_feed_dict)

    def construct_loss(self):
        if self.output is None:
            self.set_output()
            cbe_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_placeholder,\
                                                                                 logits=self.output))
        self.loss.append(cbe_loss)
        tf.summary.scalar('loss', cbe_loss)

    def train(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        #global_step_tensor = slim.get_global_step() or slim.create_global_step()
        self.construct_loss()
        if not self.Init_train:
            self.train_handle = session.run(self.train_iter.string_handle())
            self.Init_train = True
            self.fine_tune()
        self.set_accuracy_op()
        self.loss = tf.add_n(self.loss, 'Loss_accu')
        tf.summary.scalar('loss/Total_loss', self.loss)
        global_step_tensor = slim.create_global_step()
        learning_rate = tf.train.exponential_decay(0.01, global_step_tensor, decay_steps=10000, decay_rate=0.94, staircase=False)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1, beta1=0.9, beta2=0.99)
        #batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        #IO_feed_dict = self.construct_IO_dict(batch)
        train_op = slim.learning.create_train_op(self.loss, optimizer, summarize_gradients=False)


        #train_feed_dict = self.construct_control_dict(Type='Train')
        logging.set_verbosity(1)
        slim.learning.train(
            train_op= train_op,
            logdir = self.build_params['Save_dir'] + 'logs/',
            number_of_steps=100000,
            save_summaries_secs=30,
            save_interval_secs=3600,
            global_step = global_step_tensor)
        #train_feed_dict = {**IO_feed_dict, **train_dict}
        _, loss = session.run([self.train_step, self.loss], feed_dict=train_feed_dict)
        print(loss)

    def test(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']

        #batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        #IO_feed_dict = self.construct_IO_dict(batch)
        if not self.Init_test:
            self.test_handle = session.run(self.test_iter.string_handle())
            self.Init_test = True
        test_feed_dict = self.construct_control_dict(Type='Test')
        #test_feed_dict = {**IO_feed_dict, **test_dict}

        if self.accuracy is not None:
            summary, _ = session.run([kwargs['merged'], self.accuracy], feed_dict=test_feed_dict)

        else:
            summary = session.run([kwargs['merged']], feed_dict=test_feed_dict)[0]

        return summary