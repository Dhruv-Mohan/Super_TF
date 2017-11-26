from utils.builder import Builder
import tensorflow as tf
from tensorflow import nn
from Model_builder.Architecture.Classification.Inception_resnet_v2a import Build_Inception_Resnet_v2a

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

                #show tell init
                initalizer = tf.random_uniform_initializer(minval=-0.08 , maxval=0.08)

                #Setting control params
                im2txt_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                #Image embeddings
                Build_Inception_Resnet_v2a(kwargs)
                inception_output = tf.get_collections(self.Model_name + '_Incepout')[0]
                image_embeddings = im2txt_builder.FC_layer(inception_output, filters=512)

                #Seq embeddings
                embeddings_map = tf.get_variable(name='Map', shape=[40,512], initializer=initalizer)
                seq_embeddings = tf.nn.embedding_lookup(embeddings_map, output_placeholder) #output placeholder is placeholder need to fit to input seq


                lstm_cell = im2txt_builder.Lstm_cell();
                lstm_cell = im2txt_builder.Rnn_dropout(lstm_cell)
                with tf.variable_scope("lstm") as lstm_scope:
                    zero_state = lstm_cell.zero_state(batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
                    _, initial_stae = lstm_cell(image_embedding, zero_state)

                    lstm_scope.reuse_variables()
                    sequence_length = tf.reduce_sum(sequence_mask, 1) #Add sequence_mask 
                    lstm_outputs, _ =nn.dynamic_rnn(cell=lstm_cell, inputs=seq_embedding, sequence_length=sequence_length, initial_stae=initial_stae, dtype=tf.float32, scope=lstm_scope)
                    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

                    logits = im2txt_builder.FC_layer(lstm_outputs, filters=40, readout=True)
                    #Target seq and losses next 
