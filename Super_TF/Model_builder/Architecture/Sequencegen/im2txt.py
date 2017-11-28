from utils.builder import Builder
import tensorflow as tf
from tensorflow import nn
from Model_builder.Architecture.Classification.Inception_resnet_v2a import Build_Inception_Resnet_v2a

def Build_Im2txt(kwargs):
        ''' IM2TXT'''
        with tf.name_scope('IM2TXT'):
            with Builder(**kwargs) as im2txt_builder:
                '''
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                #dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                #state_placeholder = tf.placeholder(tf.string, name="State")
                #input_reshape = im2txt_builder.Reshape_input(input_placeholder, width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])
                #Redundant feature extractor already creates this placeholder
                '''
                if kwargs['State'] is 'Train':
                    input_seq_placeholder = tf.placeholder(tf.int32, shape=[None, kwargs['Padded_length']], name='Input_Seq')
                    target_seq_placeholder = tf.placeholder(tf.int32, shape=[None, kwargs['Padded_length']], name='Target_Seq')
                elif kwargs['State'] is 'Test':
                    input_seq_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name='Input_Seq')
                    target_seq_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name='Target_Seq')

                mask_placeholder = tf.placeholder(tf.int32, shape=[None, kwargs['Padded_length']], name='Seq_Mask')
                Lstm_state_placeholder = tf.placeholder(tf.float32, shape=[])

                '''
                TODO:
                Get input_seq, mask and target seq from reader
                Init inception-resnet correctly and attach input from reader to input_placeholder of inception-resnet
                Understand and build deploy state
                Seperate implementation of loss and construction of network
                '''

                #reader will give input seq, mask and target seq
                #show tell init
                initalizer = tf.random_uniform_initializer(minval=-0.08 , maxval=0.08)


                #Building feature extractor
                Build_Inception_Resnet_v2a(kwargs)
                
                #Extracting necessary variables from feature extractor
                inception_output = tf.get_collections(self.Model_name + '_Incepout')[0]
                inception_state = tf.get_collections(self.Model_name + '_State')[0]
                inception_dropout = tf.get_collections(self.Model_name + '_Dropout_prob_ph')[0]

                #Setting control params
                im2txt_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)

                #Image embeddings
                image_embeddings = im2txt_builder.FC_layer(inception_output, filters=512)

                #Seq embeddings
                embeddings_map = tf.get_variable(name='Map', shape=[40,512], initializer=initalizer)
                seq_embeddings = tf.nn.embedding_lookup(embeddings_map, input_seq_placeholder) 


                lstm_cell = im2txt_builder.Lstm_cell();
                lstm_cell = im2txt_builder.Rnn_dropout(lstm_cell)
                with tf.variable_scope("lstm") as lstm_scope:
                    zero_state = lstm_cell.zero_state(batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
                    _, initial_stae = lstm_cell(image_embedding, zero_state)

                    lstm_scope.reuse_variables()
                    if kwargs['State'] is 'Test':
                        state_feed = tf.placeholder(dtype=tf.float32, shape=[None, sum(lstm_cell.state_size)], name='State_feed')
                        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
                        lstm_outputs, state_tuple = lstm_cell(inputs = tf.squeeze(seq_embeddings, axis=[1]), state=state_tuple)
                        concat_input = tf.concat(values= initial_stae, axis=1)
                        concat_state = tf.concat(values=state_tuple, axis=1)

                    elif kwargs['State'] is 'Train':
                        sequence_length = tf.reduce_sum(sequence_mask, 1) #Add sequence_mask 
                        lstm_outputs, _ =nn.dynamic_rnn(cell=lstm_cell, inputs=seq_embedding, sequence_length=sequence_length, initial_stae=initial_stae, dtype=tf.float32, scope=lstm_scope)

                    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

                    logits = im2txt_builder.FC_layer(lstm_outputs, filters=40, readout=True)
                    #Target seq and losses next 
                    if kwargs['State'] is 'Train':
                        targets = tf.reshape(target_seq_placeholder, [-1]) #flattening target seqs
                        weights = tf.to_float(tf.reshape(input_mask, [-1]))

                        with tf.name_scope('Softmax_CE_loss'):
                            seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                            batch_loss = tf.div(tf.reduce_sum(tf.multiply(seq_loss, weights)), tf.reduce_sum(weights))


                    tf.add_to_collection(kwargs['Model_name'] + '_Input_seq_ph', input_seq_placeholder)
                    tf.add_to_collection(kwargs['Model_name'] + '_Output_ph', target_seq_placeholder)
                    tf.add_to_collection(kwargs['Model_name'] + '_Mask_ph', mask_placeholder)
                    tf.add_to_collection(kwargs['Model_name'] + '_Output', logits)

                    if kwargs['State'] is 'Test':
                        tf.add_to_collection(kwargs['Model_name'] + '_Initial_state', concat_input)
                        tf.add_to_collection(kwargs['Model_name'] + '_Lstm_state_feed', state_feed)
                        tf.add_to_collection(kwargs['Model_name'] + '_Lstm_state', concat_state)

                    elif kwargs['State'] is 'Train':
                        tf.add_to_collection(kwargs['Model_name'] + '_Loss', batch_loss)

                    #Test output next

                    return 'Sequence'

