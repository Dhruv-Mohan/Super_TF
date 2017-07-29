from utils.builder import Builder
import tensorflow as tf


class Factory(object):
    """Factory class to build DNN Architectures"""
    #Look into adding a datastructure to keep track of last layer added to the graph
    def get_model(self):
        #return self.Build_Lenet()
        print('Build_'+self.model_name+'()')
        return (eval('self.Build_'+self.model_name+'()'))

    def Build_Alexnet(self):
        with tf.name_scope('Alexnet_model'):
            with Builder(**self.kwargs) as alexnet_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')

                input_reshape = alexnet_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #FEATURE EXTRACTION
                conv1 = alexnet_builder.Conv2d_layer(input_reshape, stride=[1, 4, 4, 1], k_size=[11, 11], filters=96, padding='VALID')
                pool1 = alexnet_builder.Pool_layer(conv1, k_size=[1, 3, 3, 1])

                pad1 = alexnet_builder.Pad_layer(pool1, p_type='SYMMETRIC')
                conv2 = alexnet_builder.Conv2d_layer(pad1, k_size=[5, 5], filters=256, padding='VALID')
                pool2 = alexnet_builder.Pool_layer(conv2, k_size=[1, 3, 3, 1])

                pad2 = alexnet_builder.Pad_layer(pool2, p_size=[1, 1], p_type='SYMMETRIC')
                conv3 = alexnet_builder.Conv2d_layer(pad2, filters=384, padding='VALID')

                pad3 = alexnet_builder.Pad_layer(conv3, p_size=[1, 1], p_type='SYMMETRIC')
                conv4 = alexnet_builder.Conv2d_layer(pad3, filters=384, padding='VALID')

                pad4 = alexnet_builder.Pad_layer(conv4, p_size=[1, 1], p_type='SYMMETRIC')
                conv5 = alexnet_builder.Conv2d_layer(pad4, filters=256, padding='VALID')
                pool5 = alexnet_builder.Pool_layer(conv5, k_size=[1, 3, 3, 1])

                #DENSELY CONNECTED
                fc1 = alexnet_builder.FC_layer(pool5, filters=4096)
                drop1 = alexnet_builder.Dropout_layer(fc1, dropout_prob_placeholder)

                fc2 = alexnet_builder.FC_layer(drop1, filters=4096)
                drop2 = alexnet_builder.Dropout_layer(fc2, dropout_prob_placeholder)

                output = alexnet_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)

                return(input_placeholder, output_placeholder, output, dropout_prob_placeholder)
           
    def Build_Lenet(self):
        with tf.name_scope('LeNeT_Model'):
            #with Builder(Summary=True,Batch_size=50,Image_width=28,Image_height=28,Image_cspace=1) as lenet_builder:
            with Builder(**self.kwargs) as lenet_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                input_reshape = lenet_builder.Reshape_input(input_placeholder)
                
                conv1 = lenet_builder.Conv2d_layer(input_reshape, k_size=[5, 5])
                pool1 = lenet_builder.Pool_layer(conv1)

                conv2 = lenet_builder.Conv2d_layer(pool1, k_size=[5, 5], filters=64)
                pool2 = lenet_builder.Pool_layer(conv2)

                fc1 = lenet_builder.FC_layer(pool2);
                output = lenet_builder.FC_layer(fc1, filters=self.kwargs['Classes'], readout=True)
                return(input_placeholder, output_placeholder, output)







    def __init__(self, **kwargs):
        #TODO: WRITE ERROR HANDLER AND PARSER 
        self.model_name = kwargs['Model_name']
        self.summary = kwargs['Summary']
        self.kwargs = kwargs
        #Add more params as required



