from utils.builder import Builder
import tensorflow as tf
from tensorflow import nn
from tensorflow.python.layers.core import Dense
from utils.Base_Archs.Base_RNN import Base_RNN



class MDM(Base_RNN):
    """MDM for facial landmark tracking"""

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.input_seq_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Patches'], 2], name='Input_Seq')
        self.target_seq_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Patches'], 2], name='Target_Seq')

    def build_net(self):
        with tf.name_scope('MDM'):
            with Builder(**self.build_params) as MDM_builder:

                #BUILD CONV MODEL HERE get image embeddings

                image_embeddings= 0
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(512, tf.tanh, tf.AUTO_REUSE)
                rnn_state = rnn_cell.zero_state(tf.shape(image_embeddings)[0], dtype=tf.float32)
                delta = tf.zeros((tf.shape(image_embeddings)[0], num_patches, 2))
                for i in range(4):
                    image_embeddings = self.get_image_embeddings()
                    predictions, rnn_state = rnn_cell(image_embeddings, rnn_state)
                    processed_preds = MDM_builder.FC_layer(predictions, filters=68*2, readout=True, name='Preds')
                    output_preds = tf.reshape(processed_preds, [tf.shape(image_embeddings)[0], 68, 2])
                    delta += output_preds
                    self.deltas.append(delta)


    def construct_IO_dict(self, batch):
        return {self.input_placeholder: batch[0], self.input_seq_placeholder: batch[1], self.target_seq_placeholder: batch[2]}

    def set_output(self):
        self.build_net()
        self.Predict_op = self.init_points + self.deltas[-1]

    def construct_loss(self):
        set_output()
        norm = tf.sqrt(tf.reduce_sum(((self.target_seq_placeholder[:, 36, :] - self.target_seq_placeholder[:, 45, :])**2), 1))
        for indx, delta in enumerate(deltas):
            norm_rms_loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(self.input_seq_placeholder + delta - self.target_seq_placeholder), 2)), 1) / (norm * 68))
            tf.summary.scalar('norm_loss_level_' + str(indx), norm_rms_loss)
            self.loss.append(norm_rms_loss)

