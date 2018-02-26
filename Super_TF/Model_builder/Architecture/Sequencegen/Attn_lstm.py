from utils.builder import Builder
import tensorflow as tf
from tensorflow import nn
from Model_builder.Architecture.Classification.Inception_resnet_v2a import Inception_resnet_v2a
from tensorflow.python.layers.core import Dense
from utils.Base_Archs.Base_RNN import Base_RNN

class Atrtn_lstm(Base_RNN):

    def build_net(self):
        with tf.name_scope('Attn_lstm'):
            with Builder(**self.build_params) as Attn_lstm_builder:

                #build ing feature extractor
                F_ext = Inception_resnet_v2a(self.build_params)
                F_vects = F_ext.Endpoints['Model_conv'] #feature vectors

                #show tell init
                initalizer = tf.random_uniform_initializer(minval=-0.08 , maxval=0.08)
                #Setting control params
                Attn_lstm_builder.control_params(Dropout_control=self.dropout_placeholder, State=self.state_placeholder)

                #Image embeddings
                with tf.name_scope('Lstm_Embeddings'):
                    #image_embeddings = Attn_lstm_builder.FC_layer(inception_output, filters=512, flatten=False)
                    #image_embeddings = F_vects
                    image_embeddings =  Attn_lstm_builder.FC_layer(F_vects, filters=1536) #why?
                    image_embeddings_size= tf.shape(image_embeddings)
                    
                    #Seq embeddings
                    embeddings_map = tf.get_variable(name='Map', shape=[40,512*2], initializer=initalizer)
                    seq_embeddings = tf.nn.embedding_lookup(embeddings_map, self.input_seq_placeholder) 

                    lstm_cell = Attn_lstm_builder.Lstm_cell_LayerNorm(1024)
                    Bah_atten_mech = tf.contrib.seq2seq.BahdanauAttention(512*2, normalize=True, memory=image_embeddings)
                    lstm_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_cell, Bah_atten_mech, output_attention=False, attention_layer_size=512)
                    top_cell =Attn_lstm_builder.Lstm_cell_LayerNorm(1024)

                    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell, top_cell])
                    
                    #lstm_cell = Attn_lstm_builder.Lstm_cell();
                    #lstm_cell = Attn_lstm_builder.Rnn_dropout(lstm_cell)
                    
                
                zero_state = lstm_cell.zero_state(batch_size=image_embeddings_size[0], dtype=tf.float32)
                _, initial_stae = lstm_cell(image_embeddings, zero_state)
                next_state = initial_stae
                #lstm_scope.reuse_variables()
                if self.build_params['State'] is 'Test':
                        
                        state_feed = tf.placeholder(dtype=tf.float32, shape=[None, 1024], name='State_feed')
                        attn_feed = tf.placeholder(dtype=tf.float32, shape=[None, 1024], name='Attn_feed')
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embeddings_map, start_tokens=tf.tile([36], [image_embeddings_size[0]]), end_token=37)

                elif self.build_params['State'] is 'Train':
                        sequence_length = tf.reduce_sum(mask_placeholder, 1) #Add sequence_mask 
                        helper = tf.contrib.seq2seq.TrainingHelper(inputs=seq_embeddings, sequence_length=tf.tile([self.build_params['Padded_length']], [10]))

                output_layer = Dense(units=40, use_bias=False, name='output_proj')
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=lstm_cell, helper=helper, initial_state=initial_stae, output_layer=output_layer)
                lstm_outputs, a ,b = tf.contrib.seq2seq.dynamic_decode(decoder = decoder, impute_finished=True, maximum_iterations=15, output_time_major=False)
                lstm_outputs = lstm_outputs.rnn_output
                print(lstm_outputs.get_shape().as_list())
                with tf.name_scope('Lstm_output'):
                        logits = tf.reshape(lstm_outputs, [-1,40])
                return  logits

    def construct_loss(self):
        if self.output is None:
            self.set_output()
        #Target seq and losses next 
        with tf.name_scope('Lstm_loss'):
            if self.build_params['State'] is 'Train':
                targets = tf.reshape(self.target_seq_placeholder, [-1]) #flattening target seqs
                weights = tf.to_float(tf.reshape(self.mask_placeholder, [-1]))
                with tf.name_scope('Softmax_CE_loss'):
                    seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                    batch_loss = tf.div(tf.reduce_sum(tf.multiply(seq_loss, weights)), tf.maximum(tf.reduce_sum(weights),1))
                    self.loss.append(batch_loss)