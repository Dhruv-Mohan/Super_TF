from utils.builder import Builder
import tensorflow as tf


class Factory(object):
    """Factory class to build DNN Architectures"""
    #Look into adding a datastructure to keep track of last layer added to the graph


    def get_model(self):
        #return self.Build_Lenet()
        print('Build_'+self.model_name+'()')
        return (eval('self.Build_'+self.model_name+'()'))
    

    def Build_Incpetion_Resnet_v2(self):
        with tf.name_scope('Inception_Resnet_v2_model'):
            with Builder(**self.kwargs) as inceprv2_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                train_state_placeholder = tf.placeholder(tf.bool, name="Train_State")
                input_reshape = inceprv2_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                inceprv2_builder.control_params(Dropout_control=dropout_prob_placeholder, Train_state=train_state_placeholder)
                
                #Construct functional building blocks
                def stem():
                    with tf.name_scope('Stem'):
                        conv1 = inceprv2_builder.Conv2d_layer(input_reshape, stride=[1,2,2,1], filters=32, padding='VALID', Batch_norm=True)
                        conv2 = inceprv2_builder.Conv2d_layer(conv1, stride=[1,1,1,1], filters=32, padding='VALID', Batch_norm=True)
                        conv3 = inceprv2_builder.Conv2d_layer(conv2, stride=[1,1,1,1], filters=64, padding='VALID', Batch_norm=True)
                        
                        conv4a_split1 = inceprv2_builder.Conv2d_layer(conv3, stride=[1,2,2,1], filters=96, padding='VALID', Batch_norm=True)
                        pool1b_split1 = inceprv2_builder.Pool_layer(conv3, stride=[1,2,2,1], padding='VALID')

                        concat1 = inceprv2_builder.Concat([conv4a_split1, pool1b_split1])

                        conv5a_split2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=64, Batch_norm=True)
                        conv6a_split2 = inceprv2_builder.Conv2d_layer(conv5a_split2, stride=[1, 1, 1, 1], k_size=[7, 1], filters=64, Batch_norm=True)
                        conv7a_split2 = inceprv2_builder.Conv2d_layer(conv6a_split2, stride=[1, 1, 1, 1], k_size=[1, 7], filters=64, Batch_norm=True)
                        conv8a_split2 = inceprv2_builder.Conv2d_layer(conv7a_split2, stride=[1, 1, 1, 1], filters=96, Batch_norm=True)

                        conv5b_split2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=64, Batch_norm=True)
                        conv6b_split2 = inceprv2_builder.Conv2d_layer(conv5b_split2, stride=[1, 1, 1, 1],filters=96, padding='VALID', Batch_norm=True)


    def Build_vgg19(self):
        with tf.name_scope('Vgg_model'):
            with Builder(**self.kwargs) as vgg19_builder:
                input_placeholder = tf.placeholder(tf.float32, \
                    shape=[None, self.kwargs['Image_width']*self.kwargs['Image_height']*self.kwargs['Image_cspace']], name='Input')
                output_placeholder = tf.placeholder(tf.float32, shape=[None, self.kwargs['Classes']], name='Output')
                dropout_prob_placeholder = tf.placeholder(tf.float32, name='Dropout')
                input_reshape = vgg19_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                vgg19_builder.control_params(Dropout_control=dropout_prob_placeholder)

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
                drop1 = vgg19_builder.Dropout_layer(fc1)

                fc2 = vgg19_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg19_builder.Dropout_layer(fc2)

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
                input_reshape = vgg16_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                vgg16_builder.control_params(Dropout_control=dropout_prob_placeholder)

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
                drop1 = vgg16_builder.Dropout_layer(fc1)

                fc2 = vgg16_builder.FC_layer(drop1, filters=4096)
                drop2 = vgg16_builder.Dropout_layer(fc2)

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
                train_state_placeholder = tf.placeholder(tf.bool, name="Train_State")

                input_reshape = alexnet_builder.Reshape_input(input_placeholder, width=self.kwargs['Image_width'], height=self.kwargs['Image_height'], colorspace= self.kwargs['Image_cspace'])

                #Setting control params
                alexnet_builder.control_params(Dropout_control=dropout_prob_placeholder, Train_state=train_state_placeholder)

                #FEATURE EXTRACTION
                conv1 = alexnet_builder.Conv2d_layer(input_reshape, stride=[1, 4, 4, 1], k_size=[11, 11], filters=96, padding='VALID', Batch_norm=True)
                
                pool1 = alexnet_builder.Pool_layer(conv1, k_size=[1, 3, 3, 1], padding='VALID')

                pad1 = alexnet_builder.Pad_layer(pool1, p_type='SYMMETRIC')
                conv2 = alexnet_builder.Conv2d_layer(pad1, k_size=[5, 5], filters=256, padding='VALID', Batch_norm=True)

                pool2 = alexnet_builder.Pool_layer(conv2, k_size=[1, 3, 3, 1], padding='VALID')

                conv3 = alexnet_builder.Conv2d_layer(pool2, filters=384, Batch_norm=True)
                conv4 = alexnet_builder.Conv2d_layer(conv3, filters=384, Batch_norm=True)
                conv5 = alexnet_builder.Conv2d_layer(conv4, filters=256, Batch_norm=True)

                pool5 = alexnet_builder.Pool_layer(conv5, k_size=[1, 3, 3, 1])

                #DENSELY CONNECTED
                fc1 = alexnet_builder.FC_layer(pool5, filters=4096)
                drop1 = alexnet_builder.Dropout_layer(fc1)

                fc2 = alexnet_builder.FC_layer(drop1, filters=4096)
                drop2 = alexnet_builder.Dropout_layer(fc2)

                output = alexnet_builder.FC_layer(drop2, filters=self.kwargs['Classes'], readout=True)

                Alexnet_dict = {'Input_ph': input_placeholder, 'Output_ph': output_placeholder, 'Output': output, 'Dropout_prob_ph': dropout_prob_placeholder, 'Train_state' : train_state_placeholder}
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