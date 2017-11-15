from utils.builder import Builder
import tensorboard as tf
from tensorflow import nn

def Build_Im2txt(kwargs):
        ''' IM2TXT'''
        with tf.name_scope('IM2TXT'):
            with Builder(**kwargs) as im2txt_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                state_placeholder = tf.placeholder(tf.string, name="State")
                input_reshape = im2txt_builder.Reshape_input(input_placeholder, width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])

                #Setting control params
                im2txt_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                lstm_cell = im2txt_builder.Lstm_cell();
                lstm_cell = im2txt_builder.Rnn_dropout(lstm_cell)
                zero_state = lstm_cell.zero_state(batch_size=input_placeholder.get_shape()[0], dtype=tf.float32)
                _, initial_stae = lstm_cell(image_embedding, zero_state)
                lstm_outputs, _ =nn.dynamic_rnn(cell=lstm_cell, inputs=seq_embedding, sequence_length=sequence_length, initial_stae=initial_stae, dtype=tf.float32)
                lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])
