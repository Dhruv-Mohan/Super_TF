from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Segnet import Base_Segnet
import  numpy as np

class Pnet(Base_Segnet):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.X_seg_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
        self.X_seg_oh = tf.squeeze(tf.one_hot(self.X_seg_placeholder, 8), 1)
        self.Y_seg_placeholder =tf.placeholder(tf.int32, shape=[None, 1])
        self.Y_seg_oh =  tf.squeeze(tf.one_hot(self.Y_seg_placeholder, 8), 1)
        self.X_reg_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y_reg_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

    def construct_IO_dict(self, batch):
        return {self.input_placeholder: batch[0], self.output_placeholder: batch[1], self.X_seg_placeholder: batch[2], 
                self.Y_seg_placeholder: batch[3], self.X_reg_placeholder: batch[4], self.Y_reg_placeholder: batch[5]} #Input image and output image

    def build_net(self):
        '''Small network to generate prior maps for final segmentation network'''
        with tf.name_scope('Pnet'):
            with Builder(**self.build_params) as Pnet_builder:

                def compress(input_layer):
                    #Feature Extraction
                    conv1a = Pnet_builder.Conv2d_layer(input_layer, filters=64, Batch_norm=True)
                    conv1b = Pnet_builder.Conv2d_layer(conv1a, filters=64, Batch_norm=True)
                    #conv1d = Pnet_builder.Concat([input_layer, conv1a, conv1b, conv1c, conv1d])
                    pool1 = Pnet_builder.Pool_layer(conv1b)

                    conv2a = Pnet_builder.Conv2d_layer(pool1, filters=128, Batch_norm=True)
                    conv2b = Pnet_builder.Conv2d_layer(conv2a, filters=128, Batch_norm=True)
                    pool2 = Pnet_builder.Pool_layer(conv2b)

                    conv3a = Pnet_builder.Conv2d_layer(pool2, filters=256, Batch_norm=True)
                    conv3ad = Pnet_builder.Concat([pool2, conv3a])
                    conv3b = Pnet_builder.Conv2d_layer(conv3ad, filters=256, Batch_norm=True)
                    conv3bd = Pnet_builder.Concat([pool2, conv3a, conv3b])
                    conv3c = Pnet_builder.Conv2d_layer(conv3bd, filters=256, Batch_norm=True)
                    conv3cd = Pnet_builder.Concat([pool2, conv3a, conv3b, conv3c])
                    conv3d = Pnet_builder.Conv2d_layer(conv3cd, filters=256, Batch_norm=True)
                    #conv3f = Pnet_builder.Concat([pool2, conv3a, conv3b, conv3c, conv3d, conv3e, conv3f])
                    pool3 = Pnet_builder.Pool_layer(conv3d)

                    conv4a = Pnet_builder.Conv2d_layer(pool3, filters=512, Batch_norm=True)
                    conv4ad = Pnet_builder.Concat([pool3, conv4a])
                    conv4b = Pnet_builder.Conv2d_layer(conv4ad, filters=512, Batch_norm=True)
                    conv4bd = Pnet_builder.Concat([pool3, conv4a, conv4b])
                    conv4c = Pnet_builder.Conv2d_layer(conv4bd, filters=512, Batch_norm=True)
                    conv4cd = Pnet_builder.Concat([pool3, conv4a, conv4b, conv4c])
                    conv4d = Pnet_builder.Conv2d_layer(conv4cd, filters=512, Batch_norm=True)
                    conv4dd = Pnet_builder.Concat([pool3, conv4a, conv4b, conv4c, conv4d])
                    conv4e = Pnet_builder.Conv2d_layer(conv4dd, filters=512, Batch_norm=True)
                    conv4ed = Pnet_builder.Concat([pool3, conv4a, conv4b, conv4c, conv4d, conv4e])
                    conv4f = Pnet_builder.Conv2d_layer(conv4ed, filters=512, Batch_norm=True)
                    #conv4f = Pnet_builder.Concat([pool3, conv4a, conv4b, conv4c, conv4d, conv4e, conv4f])

                    pool4 = Pnet_builder.Pool_layer(conv4f)

                    conv5a = Pnet_builder.Conv2d_layer(pool4, filters=512, Batch_norm=True)
                    conv5ad = Pnet_builder.Concat([pool4, conv5a])
                    conv5b = Pnet_builder.Conv2d_layer(conv5ad, filters=512, Batch_norm=True)
                    conv5bd = Pnet_builder.Concat([pool4, conv5a, conv5b])
                    conv5c = Pnet_builder.Conv2d_layer(conv5bd, filters=512, Batch_norm=True)
                    conv5cd = Pnet_builder.Concat([pool4, conv5a, conv5b, conv5c])
                    conv5d = Pnet_builder.Conv2d_layer(conv5cd, filters=512, Batch_norm=True)
                    conv5dd = Pnet_builder.Concat([pool4, conv5a, conv5b, conv5c, conv5d])
                    conv5e = Pnet_builder.Conv2d_layer(conv5dd, filters=512, Batch_norm=True)
                    conv5ed = Pnet_builder.Concat([pool4, conv5a, conv5b, conv5c, conv5d, conv5e])
                    conv5f = Pnet_builder.Conv2d_layer(conv5ed, filters=512, Batch_norm=True)
                    #conv5f = Pnet_builder.Concat([pool4, conv5a, conv5b, conv5c, conv5d, conv5e, conv5f])

                    pool5 = Pnet_builder.Pool_layer(conv5f)

                    #Densely Connected
                    fc1 = Pnet_builder.FC_layer(pool5, filters=4096)
                    #fc2 = Pnet_builder.FC_layer(fc1, filters=4096)


                    #output = Pnet_builder.FC_layer(drop2, filters=self.build_params['Classes'], readout=True)
                    return fc1

                #Setting control params
                Pnet_builder.control_params(Dropout_control=self.dropout_placeholder, State=self.state_placeholder)


                #Stem
                conv1 = Pnet_builder.Conv2d_layer(self.input_placeholder, stride=[1, 2, 2, 1], filters=32, Batch_norm=True) #512
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

                x_seg = compress(conv_l)
                self.x_seg = Pnet_builder.FC_layer(x_seg, filters=8, readout=True)
                self.x_seg_out = tf.argmax(self.x_seg, 1)
                #y_seg = compress(conv_l)
                self.y_seg = Pnet_builder.FC_layer(x_seg, filters=8, readout=True)
                self.y_seg_out = tf.argmax(self.y_seg, 1)
                #x_reg  = compress(conv_l)
                self.x_reg = tf.tanh(Pnet_builder.FC_layer(x_seg, filters=1, readout = True) / 50) /2
                #y_reg = compress(conv_l)
                self.y_reg = tf.tanh(Pnet_builder.FC_layer(x_seg, filters=1, readout = True) / 50) /2


                '''
                conv_c = Pnet_builder.Conv2d_layer(pool_c, filters=256, k_size=[7,1], padding='VALID', Batch_norm=True)
                conv_c = Pnet_builder.Conv2d_layer(conv_c, filters=256, k_size=[1,7], padding='VALID', Batch_norm=True)
                
                pool_c = Pnet_builder.Pool_layer(conv_c)
                conv_c = Pnet_builder.Conv2d_layer(pool_c, filters=256, k_size=[7,1], padding='VALID', Batch_norm=True)
                conv_c = Pnet_builder.Conv2d_layer(conv_c, filters=256, k_size=[1,7], padding='VALID', Batch_norm=True)
                pool_c = Pnet_builder.Pool_layer(conv_c)
                conv_c = Pnet_builder.Conv2d_layer(pool_c, filters=256, k_size=[3,3], padding='VALID', Batch_norm=True)
                conv_c = Pnet_builder.Conv2d_layer(conv_c, filters=256, k_size=[3,3], padding='VALID', Batch_norm=True)
                pool_c = Pnet_builder.Pool_layer(conv_c)
                conv_c = Pnet_builder.Conv2d_layer(pool_c, filters=256, k_size=[3,3], Batch_norm=True)
                conv_c = Pnet_builder.Conv2d_layer(conv_c, filters=256, k_size=[3,3], Batch_norm=True)
                pool_c = Pnet_builder.Pool_layer(conv_c, k_size=[1, 8, 8, 1], pooling_type='AVG')
                '''

                unpool = Pnet_builder.Conv_Resize_layer(conv_l)
                conv5 = Pnet_builder.Conv2d_layer(unpool,  filters=256,Batch_norm=True) #512
                output = Pnet_builder.Conv2d_layer(conv5, filters =1, Activation=False, name='Output', Batch_norm=False)
                #logits = tf.reshape(output, shape= [-1, kwargs['Image_width']*kwargs['Image_height']])
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Pnet')
                print(self.update_ops)
                return output
    
    def construct_loss(self):
        super().construct_loss()
        output = tf.reshape(self.output, shape=(-1, self.build_params['Image_width'] * self.build_params['Image_height']))
        output_placeholder = tf.reshape(self.output_placeholder, shape=(-1, self.build_params['Image_width'] * self.build_params['Image_height']))
        Probs = tf.nn.sigmoid(output)
        offset = 1e-5
        Threshold = 0.1
        Probs_processed = tf.clip_by_value(Probs, offset, 1.0)
        Con_Probs_processed = tf.clip_by_value(1-Probs, offset, 1.0)
        W_I = (-output_placeholder * tf.log(Probs_processed) - (1 - output_placeholder)*tf.log(Con_Probs_processed))
        Weighted_BCE_loss = tf.reduce_sum(W_I) / tf.cast(tf.maximum(tf.count_nonzero(W_I -Threshold),0), tf.float32)
        tf.summary.scalar('WBCE loss', Weighted_BCE_loss)
        tf.summary.image('WCBE', tf.reshape(W_I, [-1, self.build_params['Image_width'], self.build_params['Image_height'], 1]))
        self.loss.append(Weighted_BCE_loss*50)
        x_seg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.X_seg_oh, logits=self.x_seg))
        y_seg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y_seg_oh, logits=self.y_seg))
        self.loss.append(x_seg_loss)
        self.loss.append(y_seg_loss)
        #x_reg_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.X_reg_placeholder +0.5,self.x_reg +0.5)))
        #y_reg_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.Y_reg_placeholder +0.5,self.y_reg +0.5)))
        x_reg_loss = tf.losses.mean_squared_error(labels=self.X_reg_placeholder +0.5 , predictions=self.x_reg +0.5)
        y_reg_loss = tf.losses.mean_squared_error(labels=self.Y_reg_placeholder +0.5, predictions=self.y_reg +0.5)
        self.loss.append(x_reg_loss*10)
        self.loss.append(y_reg_loss*10)

        tf.summary.scalar('X_Seg_Loss', x_seg_loss)
        tf.summary.scalar('Y_Seg_Loss', y_seg_loss)
        tf.summary.scalar('X_Reg_Loss', x_reg_loss)
        tf.summary.scalar('Y_Reg_Loss', y_reg_loss)


    def set_accuracy_op(self):
        super().set_accuracy_op()
        correct_x = tf.equal(tf.cast(self.X_seg_placeholder, tf.int64), tf.argmax(self.x_seg, 1))
        correct_y = tf.equal(tf.cast(self.Y_seg_placeholder, tf.int64), tf.argmax(self.y_seg, 1))
        accuracy_x = tf.reduce_mean(tf.cast(correct_x, tf.float32))
        accuracy_y = tf.reduce_mean(tf.cast(correct_y, tf.float32))
        tf.summary.scalar('X_Seg_Acc', accuracy_x)
        tf.summary.scalar('Y_Seg_Acc', accuracy_y)
        
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

    def test(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']
        #batch = kwargs['data']
        batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        IO_feed_dict = self.construct_IO_dict(batch)
        test_dict = self.construct_control_dict(Type='Test')
        print(test_dict)
        test_feed_dict = {**IO_feed_dict, **test_dict}

        if self.accuracy is not None:
            summary, _, xs,ys,xr,yr= session.run([kwargs['merged'], self.accuracy, self.x_seg_out, self.y_seg_out, self.x_reg, self.y_reg], feed_dict=test_feed_dict)
            print('Predicted XY\t', xs,ys)
            print('Actual XY\t', batch[2], batch[3])
            print('Predicted XY\t', xr,yr)
            print('Actual XY\t', batch[4], batch[5])
        else:
            summary, xs,ys, xp,yp,xss,yss= session.run([kwargs['merged'], self.x_seg_out, self.y_seg_out, self.X_seg_oh, self.Y_seg_oh,self.x_seg, self.y_seg], feed_dict=test_feed_dict)
            print('Predicted XY\t', xs,ys)
            print('Actual XY\t', batch[2], batch[3])
            print('Predicted XY\t', xss,yss)
            print('Actual XY\t', xp,yp)

        return summary

    def predict(self, **kwargs):
        session = kwargs['session']
        test_dict = self.construct_control_dict(Type='Test')
        predict_io_dict = {self.input_placeholder: kwargs['Input_Im']}
        predict_feed_dict = {**predict_io_dict, **test_dict}
        print (test_dict)
        image, xs,ys,xr,yr = session.run([self.output, self.x_seg_out, self.y_seg_out, self.x_reg, self.y_reg], feed_dict=predict_feed_dict)
        return image, xs,ys,xr,yr

                
                



