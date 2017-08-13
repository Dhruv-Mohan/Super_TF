from utils.builder import Builder
import tensorflow as tf


class Factory(object):
    """Factory class to build DNN Architectures"""
    #Look into adding a datastructure to keep track of last layer added to the graph


    def get_model(self):
        #return self.Build_Lenet()
        print('Build_'+self.model_name+'()')
        return (eval('self.Build_'+self.model_name+'()'))


    def Build_vgg19(self):
        with tf.name_scope('Vgg_model'):
            with Builder(**self.kwargs) as vgg19_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                input_reshape = alexnet_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #FEATURE EXTRACTION
                conv1a = vgg19_builder.Conv2d_layer(input_reshape, filters=64)
                conv1b = vgg19_builder.Conv2d_layer(conv1a, filters=64)

                pool1 = vgg19_builder.Pool_layer(conv1b)

                conv2a = vgg19_builder.Conv2d_layer(pool1, filters=128)
                conv2b = vgg19_builder.Conv2d_layer(conv2a, filters=128)

                pool2 = vgg19_builder.Pool_layer(conv2b)

                conv3a = vgg19_builder.Conv2d_layer(pool2, filters=256)
                conv3b = vgg19_builder.Conv2d_layer(conv3a, filters=256)
                conv3c = vgg19_builder.Conv2d_layer(conv3b, filters=256)
                conv3d = vgg19_builder.Conv2d_layer(conv3c, filters=256)

                pool3 = vgg19_builder.Pool_layer(conv3d)

                conv4a = vgg19_builder.Conv2d_layer(pool3, filters=512)
                conv4b = vgg19_builder.Conv2d_layer(conv4a, filters=512)
                conv4c = vgg19_builder.Conv2d_layer(conv4b, filters=512)
                conv4d = vgg19_builder.Conv2d_layer(conv4c, filters=512)

                pool4 = vgg19_builder.Pool_layer(conv4d)

                conv5a = vgg19_builder.Conv2d_layer(pool4, filters=512)
                conv5b = vgg19_builder.Conv2d_layer(conv5a, filters=512)
                conv5c = vgg19_builder.Conv2d_layer(conv5b, filters=512)
                conv5d = vgg19_builder.Conv2d_layer(conv5c, filters=512)

                pool5 = vgg19_builder.Pool_layer(conv5d)

                #DENSELY CONNECTED
                fc1 = vgg19_builder.FC_layer(pool5, filters=4096)
                drop1 = vgg19_builder.Dropout_layer(fc1, dropout_prob_placeholder)

                fc2 = vgg19_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg19_builder.Dropout_layer(fc2, dropout_prob_placeholder)

                output = vgg19_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)

                VGG19_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder }
                return(VGG19_dict)



    def Build_vgg16(self):
        with tf.name_scope('Vgg_model'):
            with Builder(**self.kwargs) as vgg16_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                input_reshape = alexnet_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #FEATURE EXTRACTION
                conv1a = vgg16_builder.Conv2d_layer(input_reshape, filters=64)
                conv1b = vgg16_builder.Conv2d_layer(conv1a, filters=64)

                pool1 = vgg16_builder.Pool_layer(conv1b)

                conv2a = vgg16_builder.Conv2d_layer(pool1, filters=128)
                conv2b = vgg16_builder.Conv2d_layer(conv2a, filters=128)

                pool2 = vgg16_builder.Pool_layer(conv2b)

                conv3a = vgg16_builder.Conv2d_layer(pool2, filters=256)
                conv3b = vgg16_builder.Conv2d_layer(conv3a, filters=256)
                conv3c = vgg16_builder.Conv2d_layer(conv3b, filters=256)

                pool3 = vgg16_builder.Pool_layer(conv3c)

                conv4a = vgg16_builder.Conv2d_layer(pool3, filters=512)
                conv4b = vgg16_builder.Conv2d_layer(conv4a, filters=512)
                conv4c = vgg16_builder.Conv2d_layer(conv4b, filters=512)

                pool4 = vgg16_builder.Pool_layer(conv4c)

                conv5a = vgg16_builder.Conv2d_layer(pool4, filters=512)
                conv5b = vgg16_builder.Conv2d_layer(conv5a, filters=512)
                conv5c = vgg16_builder.Conv2d_layer(conv5b, filters=512)

                pool5 = vgg16_builder.Pool_layer(conv5c)

                #DENSELY CONNECTED
                fc1 = vgg16_builder.FC_layer(pool5, filters=4096)
                drop1 = vgg16_builder.Dropout_layer(fc1, dropout_prob_placeholder)

                fc2 = vgg16_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg16_builder.Dropout_layer(fc2, dropout_prob_placeholder)

                output = vgg16_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)

                VGG16_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder }
                return(VGG16_dict)



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
                pool1 = alexnet_builder.Pool_layer(conv1, k_size=[1, 3, 3, 1], padding='VALID')

                pad1 = alexnet_builder.Pad_layer(pool1, p_type='SYMMETRIC')
                conv2 = alexnet_builder.Conv2d_layer(pad1, k_size=[5, 5], filters=256, padding='VALID')
                pool2 = alexnet_builder.Pool_layer(conv2, k_size=[1, 3, 3, 1], padding='VALID')

                conv3 = alexnet_builder.Conv2d_layer(pool2, filters=384)

                conv4 = alexnet_builder.Conv2d_layer(conv3, filters=384)

                conv5 = alexnet_builder.Conv2d_layer(conv4, filters=256)
                pool5 = alexnet_builder.Pool_layer(conv5, k_size=[1, 3, 3, 1])

                #DENSELY CONNECTED
                fc1 = alexnet_builder.FC_layer(pool5, filters=4096)
                drop1 = alexnet_builder.Dropout_layer(fc1, dropout_prob_placeholder)

                fc2 = alexnet_builder.FC_layer(drop1, filters=4096)
                drop2 = alexnet_builder.Dropout_layer(fc2, dropout_prob_placeholder)

                output = alexnet_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)

                Alexnet_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder }
                return(Alexnet_dict)
           
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

                Lenet_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output}
                return(Lenet_dict)







    def __init__(self, **kwargs):
        #TODO: WRITE ERROR HANDLER AND PARSER 
        self.model_name = kwargs['Model_name']
        self.summary = kwargs['Summary']
        self.kwargs = kwargs
        #Add more params as required



