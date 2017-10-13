from utils.builder import Builder
import tensorflow as tf


def Build_Lenet(self):
    ''' Add paper and brief description'''
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

                Lenet_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output}
                return(Lenet_dict)