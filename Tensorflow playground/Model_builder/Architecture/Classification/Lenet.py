from utils.builder import Builder
import tensorflow as tf


def Build_Lenet(kwargs):
        ''' Add paper and brief description'''
        with tf.name_scope('LeNeT_Model'):
            #with Builder(Summary=True,Batch_size=50,Image_width=28,Image_height=28,Image_cspace=1) as lenet_builder:
            with Builder(**kwargs) as lenet_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, kwargs['Image_width']*kwargs['Image_height']*kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, kwargs['Classes']], name='Output')
                input_reshape = lenet_builder.Reshape_input(input_placeholder)
                
                conv1 = lenet_builder.Conv2d_layer(input_reshape, k_size=[5, 5])
                pool1 = lenet_builder.Pool_layer(conv1)

                conv2 = lenet_builder.Conv2d_layer(pool1, k_size=[5, 5], filters=64)
                pool2 = lenet_builder.Pool_layer(conv2)

                fc1 = lenet_builder.FC_layer(pool2);
                output = lenet_builder.FC_layer(fc1, filters=kwargs['Classes'], readout=True)

                #Logit Loss
                with tf.name_scope('Cross_entropy_loss'):
                    softmax_logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=output))

                tf.add_to_collection(kwargs['Model_name'] + '_Input_ph', input_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Input_reshape', input_reshape)
                tf.add_to_collection(kwargs['Model_name'] + '_Output_ph', output_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Output', output)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', softmax_logit_loss)

                return 'Classification'