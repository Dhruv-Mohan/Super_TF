from utils.builder import Builder
import tensorflow as tf
from tensorflow import nn
from Model_builder.Architecture.Classification.Inception_resnet_v2a import Build_Inception_Resnet_v2a
from tensorflow.python.layers.core import Dense

def Build_Attn_lstm(kwargs):
        ''' Attn_lstm'''
        with tf.name_scope('Attn_lstm'):
            with Builder(**kwargs) as Attn_lstm_builder:
                '''
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                #dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                #state_placeholder = tf.placeholder(tf.string, name="State")
                #input_reshape = Attn_lstm_builder.Reshape_input(input_placeholder, width=kwargs['Image_width'], height=kwargs['Image_height'], colorspace= kwargs['Image_cspace'])
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
                with tf.name_scope('Feature_Extractor'):
                    inception_output_init_state = tf.get_collection(kwargs['Model_name'] + '_Incepout')[0]
                    inception_output = tf.get_collection(kwargs['Model_name'] + '_Incepout_attn')[0]
                    inception_state = tf.get_collection(kwargs['Model_name'] + '_State')[0]
                    inception_dropout = tf.get_collection(kwargs['Model_name'] + '_Dropout_prob_ph')[0]

                #Setting control params
                Attn_lstm_builder.control_params(Dropout_control=inception_dropout, State=inception_state)

                #Image embeddings
                with tf.name_scope('Lstm_Embeddings'):
                    #image_embeddings = Attn_lstm_builder.FC_layer(inception_output, filters=512, flatten=False)
                    image_embeddings = inception_output
                    image_embeddings_size= tf.shape(image_embeddings)
                    inital_image_embeddings =  Attn_lstm_builder.FC_layer(inception_output_init_state, filters=512) 
                    #Seq embeddings
                    embeddings_map = tf.get_variable(name='Map', shape=[40,512], initializer=initalizer)
                    seq_embeddings = tf.nn.embedding_lookup(embeddings_map, input_seq_placeholder) 


                    lstm_cell = Attn_lstm_builder.Lstm_cell_LayerNorm()
                    Bah_atten_mech = tf.contrib.seq2seq.BahdanauAttention(512, normalize=True, memory=image_embeddings)
                    lstm_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_cell, Bah_atten_mech, output_attention=False, attention_layer_size=512)

                    
                    #lstm_cell = Attn_lstm_builder.Lstm_cell();
                    #lstm_cell = Attn_lstm_builder.Rnn_dropout(lstm_cell)
                    
                with tf.variable_scope("lstm") as lstm_scope:
                    zero_state = lstm_cell.zero_state(batch_size=image_embeddings_size[0], dtype=tf.float32)
                    _, initial_stae = lstm_cell(inital_image_embeddings, zero_state)
                    next_state = initial_stae.clone()
                    print(initial_stae.cell_state)
                    lstm_scope.reuse_variables()
                    if kwargs['State'] is 'Test':
                        
                        state_feed = tf.placeholder(dtype=tf.float32, shape=[None, sum(lstm_cell.state_size.cell_state)], name='State_feed')
                        attn_feed = tf.placeholder(dtype=tf.float32, shape=[None, lstm_cell.state_size.attention], name='Attn_feed')
                        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
                        #prev_state = tf.contrib.seq2seq.AttentionWrapperState(cell_state= state_tuple, attention=attn_feed)
                        
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embeddings_map, start_tokens=tf.tile([36], [image_embeddings_size[0]]), end_token=37)
                        lstm_outputs, next_state = lstm_cell(inputs = tf.squeeze(seq_embeddings, axis=[1]), state=next_state)
                        state_tuple = next_state.cell_state
                        nxt_attn = next_state.attention
                        init_attn = initial_stae.attention
                        concat_input = tf.concat(values= initial_stae.cell_state , axis=1)
                        concat_state = tf.concat(values=state_tuple, axis=1)
                        
                    elif kwargs['State'] is 'Train':
                        sequence_length = tf.reduce_sum(mask_placeholder, 1) #Add sequence_mask 
                        #sequence_length = tf.constant(tf.ones(shape=[9])) *10
                        #lstm_outputs, _ =nn.dynamic_rnn(cell=lstm_cell, inputs=seq_embeddings, sequence_length=sequence_length, initial_state=initial_stae, dtype=tf.float32, scope=lstm_scope)
                        helper = tf.contrib.seq2seq.TrainingHelper(inputs=seq_embeddings, sequence_length=[10,10,10,10,10,10,10,10,10,10])

                    decoder = tf.contrib.seq2seq.BasicDecoder(cell=lstm_cell, helper=helper, initial_state=initial_stae)
                    lstm_outputs, a ,b = tf.contrib.seq2seq.dynamic_decode(decoder = decoder, impute_finished=True, maximum_iterations=12)
                    lstm_outputs = lstm_outputs.rnn_output
                    with tf.name_scope('Lstm_output'):
                        lstm_outputs = tf.reshape(lstm_outputs, [-1,512])

                        logits = Attn_lstm_builder.FC_layer(lstm_outputs, filters=40, readout=True)
                    #Target seq and losses next 
                    with tf.name_scope('Lstm_loss'):
                        if kwargs['State'] is 'Train':
                            targets = tf.reshape(target_seq_placeholder, [-1]) #flattening target seqs
                            weights = tf.to_float(tf.reshape(mask_placeholder, [-1]))

                            with tf.name_scope('Softmax_CE_loss'):
                                seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                                batch_loss = tf.div(tf.reduce_sum(tf.multiply(seq_loss, weights)), tf.maximum(tf.reduce_sum(weights),1))


                    tf.add_to_collection(kwargs['Model_name'] + '_Input_seq_ph', input_seq_placeholder)
                    tf.add_to_collection(kwargs['Model_name'] + '_Output_ph', target_seq_placeholder)
                    tf.add_to_collection(kwargs['Model_name'] + '_Mask_ph', mask_placeholder)
                    tf.add_to_collection(kwargs['Model_name'] + '_Output', logits)

                    if kwargs['State'] is 'Test':
                        tf.add_to_collection(kwargs['Model_name'] + '_Initial_state', concat_input)
                        tf.add_to_collection(kwargs['Model_name'] + '_Lstm_state_feed', state_feed)
                        tf.add_to_collection(kwargs['Model_name'] + '_Lstm_state', concat_state)
                        tf.add_to_collection(kwargs['Model_name'] + '_Lstm_state_feed', attn_feed)
                        tf.add_to_collection(kwargs['Model_name'] + '_Lstm_state', nxt_attn)
                        tf.add_to_collection(kwargs['Model_name'] + '_Initial_state', init_attn)

                    elif kwargs['State'] is 'Train':
                        tf.add_to_collection(kwargs['Model_name'] + '_Loss', batch_loss)

                    #Test output next

                    return 'Sequence'

