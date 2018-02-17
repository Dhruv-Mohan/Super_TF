from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Segnet import Base_Segnet

class Pnet(Base_Segnet):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        
    def build_net(self):
        '''Small network to generate prior maps for final segmentation network'''
        with tf.name_scope('Pnet'):
            with Builder(**kwargs) as Pnet_builder:
                #Setting control params
                Pnet_builder.control_params(Dropout_control=dropout_prob_placeholder, State=state_placeholder)


                #Stem
                conv1 = Pnet_builder.Conv2d_layer(self.input_placeholder, stride=[1, 2, 2, 1], filters=32,Batch_norm=True) #512
                conv3 = Pnet_builder.DConv_layer(conv1, filters=32, Batch_norm=True, D_rate=2)


                #ASPP 
                conv1_rate4 = Pnet_builder.DConv_layer(conv3, filters=32, Batch_norm=True, D_rate=4)
                R_conn = Pnet_builder.Concat([conv1_rate4,  conv3])
                conv1_rate4a = Pnet_builder.DConv_layer(R_conn, filters=32, Batch_norm=True, D_rate=4)
                R_conn1 = Pnet_builder.Concat([conv1_rate4, conv1_rate4a, conv3])
                conv1_rate4b = Pnet_builder.DConv_layer(R_conn1, filters=32, Batch_norm=True, D_rate=4) #Embedd layers into final net #248
                R_conn2 = Pnet_builder.Concat([conv1_rate4, conv1_rate4a, conv1_rate4b, conv3])
                conv2_rate4 = Pnet_builder.DConv_layer(R_conn2, filters=32, Batch_norm=True, D_rate=4, name='Embedd4') #Embedd layers into final net #248
                conv2_rate4 = Pnet_builder.Concat([conv1_rate4, conv1_rate4a, conv1_rate4b, conv2_rate4, conv3])

                conv1_rate8 = Pnet_builder.DConv_layer(conv3, filters=32, Batch_norm=True, D_rate=8)
                R_conn18 = Pnet_builder.Concat([conv1_rate8,  conv3])
                conv1_rate8a = Pnet_builder.DConv_layer(R_conn18, filters=32, Batch_norm=True, D_rate=8)
                R_conn28 = Pnet_builder.Concat([conv1_rate8, conv1_rate8a, conv3])
                conv1_rate8b = Pnet_builder.DConv_layer(R_conn28, filters=32, Batch_norm=True, D_rate=8)
                R_conn38 = Pnet_builder.Concat([conv1_rate8, conv1_rate8a, conv1_rate8b, conv3])
                conv2_rate8 = Pnet_builder.DConv_layer(R_conn38, filters=32, Batch_norm=True, D_rate=8, name='Embedd8') #Embedd layers into final net #64
                conv2_rate8 = Pnet_builder.Concat([ conv1_rate8, conv1_rate8a, conv1_rate8b, conv2_rate8, conv3])

                conv1_rate16 = Pnet_builder.DConv_layer(conv3, filters=64, Batch_norm=True, D_rate=16)
                R_conn116 = Pnet_builder.Concat([ conv1_rate16,  conv3])
                conv1_rate16a = Pnet_builder.DConv_layer(R_conn116, filters=64, Batch_norm=True, D_rate=16) #Embedd layers into final net #32
                R_conn216 = Pnet_builder.Concat([ conv1_rate16, conv1_rate16a, conv3])
                conv1_rate16b = Pnet_builder.DConv_layer(R_conn216, filters=64, Batch_norm=True, D_rate=16)
                R_conn316 = Pnet_builder.Concat([ conv1_rate16, conv1_rate16a, conv1_rate16b, conv3])
                conv2_rate16 = Pnet_builder.DConv_layer(R_conn316, filters=64, Batch_norm=True, D_rate=16) #Embedd layers into final net #32
                conv2_rate16 = Pnet_builder.Concat([ conv1_rate16, conv1_rate16a, conv1_rate16b, conv2_rate16, conv3])


                conv1_rate32 = Pnet_builder.DConv_layer(conv3, filters=128, Batch_norm=True, D_rate=32)
                R_conn132 = Pnet_builder.Concat([ conv1_rate32, conv3])
                conv1_rate32a = Pnet_builder.DConv_layer(R_conn132, filters=128, Batch_norm=True, D_rate=32) #Embedd layers into final net #32
                R_conn232 = Pnet_builder.Concat([ conv1_rate32, conv1_rate32a, conv3])
                conv1_rate32b = Pnet_builder.DConv_layer(R_conn232, filters=128, Batch_norm=True, D_rate=32)
                R_conn332 = Pnet_builder.Concat([ conv1_rate32, conv1_rate32a, conv1_rate32b, conv3])
                conv2_rate32  = Pnet_builder.DConv_layer(R_conn332, filters=128, Batch_norm=True, D_rate=32) #Embedd layers into final net #32
                conv2_rate32 = Pnet_builder.Concat([conv1_rate32, conv1_rate32a, conv1_rate32b, conv2_rate32 , conv3])


                concat = Pnet_builder.Concat([conv2_rate16, conv2_rate4, conv2_rate8, conv2_rate32])
                conv_l = Pnet_builder.Conv2d_layer(concat, filters=256, k_size=[1,1])
                unpool = Pnet_builder.Conv_Resize_layer(conv_l)
                conv5 = Pnet_builder.Conv2d_layer(unpool,  filters=256,Batch_norm=True) #512
                output = Pnet_builder.Conv2d_layer(conv5, filters =1, Activation=False, name='Output', Batch_norm=False)
                #logits = tf.reshape(output, shape= [-1, kwargs['Image_width']*kwargs['Image_height']])
                return output
            '''
                #Add loss and debug
                with tf.name_scope('BCE_Loss'):
                    offset = 1e-5
                    Threshold = 0.1
                    Probs = tf.nn.sigmoid(logits)
                    Probs_processed = tf.clip_by_value(Probs, offset, 1.0)
                    Con_Probs_processed = tf.clip_by_value(1-Probs, offset, 1.0)
                    W_I = (-output_placeholder * tf.log(Probs_processed) - (1-output_placeholder)*tf.log(Con_Probs_processed))
                    Weighted_BCE_loss = tf.reduce_sum(W_I) / tf.cast(tf.maximum(tf.count_nonzero(W_I -Threshold),0), tf.float32)*5
                    EU_loss =tf.losses.huber_loss(output_placeholder, Probs)
                #Dice Loss
                
                with tf.name_scope('Dice_Loss'):
                    eps = tf.constant(value=1e-5, name='eps')
                    sigmoid = tf.nn.sigmoid(logits,name='sigmoid') + eps
                    intersection =tf.reduce_sum(sigmoid * output_placeholder,axis=1,name='intersection')
                    union = eps + tf.reduce_sum(sigmoid,1,name='reduce_sigmoid') + (tf.reduce_sum(output_placeholder,1,name='reduce_mask'))
                    Dice_loss = 2 * intersection / (union)
                    Dice_loss = 1 - tf.reduce_mean(Dice_loss, name='diceloss')

                #Graph Exports
                tf.add_to_collection(kwargs['Model_name'] + '_Input_ph', input_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Input_reshape', input_reshape)
                tf.add_to_collection(kwargs['Model_name'] + '_Weight_ph', weight_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Output_ph', output_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Output', output)
                tf.add_to_collection(kwargs['Model_name'] + '_Dropout_prob_ph', dropout_prob_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_State', state_placeholder)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', Weighted_BCE_loss)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', Dice_loss)
                tf.add_to_collection(kwargs['Model_name'] + '_Embedd_layers', conv2_rate4)
                tf.add_to_collection(kwargs['Model_name'] + '_Embedd_layers', conv2_rate8)
                tf.add_to_collection(kwargs['Model_name'] + '_Embedd_layers', conv2_rate16)
                tf.add_to_collection(kwargs['Model_name'] + '_Loss', EU_loss)
                #tf.add_to_collection(kwargs['Model_name'] + '_Loss', Jacard_loss)

                if kwargs['Summary']:
                    tf.summary.scalar('WBCE loss', Weighted_BCE_loss)
                    tf.summary.image('WCBE', tf.reshape(W_I, [-1,kwargs['Image_width'], kwargs['Image_height'], 1]))
                    tf.summary.scalar('Dice loss', Dice_loss)
                return 'Segmentation'
                '''







                
                



