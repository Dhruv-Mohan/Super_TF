from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Classifier import Base_Classifier


class Inception_resnet_v2py(Base_Classifier):
    """Inception-resnet-v2 as described in the https://arxiv.org/abs/1602.07261"""
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.Endpoints = {}
        self.ordered_gt_logits = self.split_logits(self.output_placeholder)
        self.keys = ['eyebrow_shape', 'face_shape', 'is_acc', 'is_eye_open', 'is_hair_vis', 
        'is_mouth_open', 'is_smiling', 'lip_shape', 'nose_shape','skin_tone', 'which_eye']
        
    def build_net(self):
        with tf.name_scope('Inception_Resnet_v2_model'):
            with Builder(**self.build_params) as inceprv2_builder:
                #Setting control params
                inceprv2_builder.control_params(Dropout_control=self.dropout_placeholder, State=self.state_placeholder)
                
                #Construct functional building blocks
                def stem(input):
                    with tf.name_scope('Stem') as scope:
                        conv1 = inceprv2_builder.Conv2d_layer(input, stride=[1,2,2,1], filters=32, padding='VALID', Batch_norm=True)
                        conv2 = inceprv2_builder.Conv2d_layer(conv1, stride=[1,1,1,1], filters=32, padding='VALID', Batch_norm=True)
                        conv3 = inceprv2_builder.Conv2d_layer(conv2, stride=[1,1,1,1], filters=64, Batch_norm=True)
                        
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(conv3, stride=[1,2,2,1], filters=96, padding='VALID', Batch_norm=True)
                        pool1b_split1 = inceprv2_builder.Pool_layer(conv3, padding='VALID')

                        concat1 = inceprv2_builder.Concat([conv1a_split1, pool1b_split1])

                        conv1a_split2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=64, Batch_norm=True)
                        conv2a_split2 = inceprv2_builder.Conv2d_layer(conv1a_split2, stride=[1, 1, 1, 1], k_size=[7, 1], filters=64, Batch_norm=True)
                        conv3a_split2 = inceprv2_builder.Conv2d_layer(conv2a_split2, stride=[1, 1, 1, 1], k_size=[1, 7], filters=64, Batch_norm=True)
                        conv4a_split2 = inceprv2_builder.Conv2d_layer(conv3a_split2, stride=[1, 1, 1, 1], filters=96, padding='VALID', Batch_norm=True)

                        conv1b_split2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=64, Batch_norm=True)
                        conv2b_split2 = inceprv2_builder.Conv2d_layer(conv1b_split2, stride=[1, 1, 1, 1], filters=96, padding='VALID', Batch_norm=True)

                        concat2 = inceprv2_builder.Concat([conv4a_split2, conv2b_split2])

                        pool1a_split3 = inceprv2_builder.Pool_layer(concat2, padding="VALID")
                        conv1b_split3 = inceprv2_builder.Conv2d_layer(concat2, stride=[1, 2, 2, 1], filters=192, padding='VALID', Batch_norm=True)

                        concat3 = inceprv2_builder.Concat([pool1a_split3, conv1b_split3])
                        return concat3

                def inception_resnet_A(input):
                    with tf.name_scope('Inception_Resnet_A') as scope:
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], filters=48, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 1, 1, 1], filters=64, Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        conv2b_split1 = inceprv2_builder.Conv2d_layer(conv1b_split1, stride=[1, 1, 1, 1], filters=32, Batch_norm=True)

                        conv1c_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=32, Batch_norm=True)
                        
                        concat1 = inceprv2_builder.Concat([conv3a_split1, conv2b_split1, conv1c_split1])

                        conv2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=384, Batch_norm=True, Activation=False)

                        conv2_scale = inceprv2_builder.Scale_activations(conv2,scaling_factor = 1)
                        residual_out = inceprv2_builder.Residual_connect([input, conv2_scale])

                        return residual_out

                def reduction_A(input):
                    with tf.name_scope('Reduction_A') as scope:
                        '''
                        k=256, l=256, m=384, n=384
                        '''
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], filters=256, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 2, 2, 1], filters=384, padding='VALID', Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 2, 2, 1], filters=384, padding='VALID', Batch_norm=True)

                        pool1c_split1 = inceprv2_builder.Pool_layer(input, padding='VALID')

                        concat = inceprv2_builder.Concat([conv3a_split1, conv1b_split1, pool1c_split1])
                        
                        return concat

                def inception_resnet_B(input):
                    with tf.name_scope('Inception_Resnet_B') as scope:
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=128, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], k_size=[1, 7], filters=160, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 1, 1, 1], k_size=[7, 1], filters=192, Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)

                        concat1 = inceprv2_builder.Concat([conv3a_split1, conv1b_split1])

                        conv2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=1152, Batch_norm=True, Activation=False) #paper discrepancy filter = 1154
                        conv2_scale = inceprv2_builder.Scale_activations(conv2, scaling_factor=0.4)

                        residual_out = inceprv2_builder.Residual_connect([input, conv2_scale])

                        return residual_out

                def reduction_B(input):
                    with tf.name_scope('Reduction_B') as scope: 
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], filters=288, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 2, 2, 1], filters=384, padding='VALID', Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2b_split1 = inceprv2_builder.Conv2d_layer(conv1b_split1, stride=[1, 2, 2, 1], filters=256, padding='VALID', Batch_norm=True)

                        conv1c_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=256, Batch_norm=True)
                        conv2c_split1 = inceprv2_builder.Conv2d_layer(conv1c_split1, stride=[1, 2, 2, 1], filters=256, padding='VALID', Batch_norm=True)

                        pool1d_split1 = inceprv2_builder.Pool_layer(input, padding='VALID')

                        concat = inceprv2_builder.Concat([conv3a_split1, conv2b_split1, conv2c_split1, pool1d_split1])
                        return concat

                def inception_resnet_C(input):
                    with tf.name_scope('Inception_Resnet_C') as scope:
                        conv1a_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)
                        conv2a_split1 = inceprv2_builder.Conv2d_layer(conv1a_split1, stride=[1, 1, 1, 1], k_size=[1, 3], filters=224, Batch_norm=True)
                        conv3a_split1 = inceprv2_builder.Conv2d_layer(conv2a_split1, stride=[1, 1, 1, 1], k_size=[3, 1], filters=256, Batch_norm=True)

                        conv1b_split1 = inceprv2_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=192, Batch_norm=True)

                        concat1 = inceprv2_builder.Concat([conv3a_split1, conv1b_split1])

                        conv2 = inceprv2_builder.Conv2d_layer(concat1, stride=[1, 1, 1, 1], k_size=[1, 1], filters=2048, Batch_norm=True, Activation=False)
                        conv2_scale = inceprv2_builder.Scale_activations(conv2)

                        residual_out = inceprv2_builder.Residual_connect([input, conv2_scale])
                        
                        return residual_out
                # Model Construction

                # Stem
                model_stem = stem(self.input_placeholder)
                # 5x Inception Resnet A
                inception_A1 = inception_resnet_A(model_stem)
                inception_A2 = inception_resnet_A(inception_A1)
                inception_A3 = inception_resnet_A(inception_A2)
                inception_A4 = inception_resnet_A(inception_A3)
                inception_A5 = inception_resnet_A(inception_A4)
                self.Endpoints['Block_A'] = inception_A5
                # Reduction A
                model_reduction_A = reduction_A(inception_A5)
                # 10X Inception Resnet B
                inception_B1 = inception_resnet_B(model_reduction_A) # Don't know if i'm missing something or now, but reduction A's output for inception resnetv2 is a tensor of depth 1152
                inception_B2 = inception_resnet_B(inception_B1)
                inception_B3 = inception_resnet_B(inception_B2)
                inception_B4 = inception_resnet_B(inception_B3)
                inception_B5 = inception_resnet_B(inception_B4)
                inception_B6 = inception_resnet_B(inception_B5)
                inception_B7 = inception_resnet_B(inception_B6)
                inception_B8 = inception_resnet_B(inception_B7)
                inception_B9 = inception_resnet_B(inception_B8)
                inception_B10 = inception_resnet_B(inception_B9)
                self.Endpoints['Block_B'] = inception_B10
                # Reduction B
                model_reduction_B = reduction_B(inception_B10)
                # 5X Inception Resnet C
                inception_C1 = inception_resnet_C(model_reduction_B)
                inception_C2 = inception_resnet_C(inception_C1)
                inception_C3 = inception_resnet_C(inception_C2)
                inception_C4 = inception_resnet_C(inception_C3)
                inception_C5 = inception_resnet_C(inception_C4)
                self.Endpoints['Block_C'] = inception_C5
                # Average Pooling
                average_pooling = inceprv2_builder.Pool_layer(inception_C5, k_size=[1, 8, 8, 1], stride=[1, 8, 8, 1], padding='SAME', pooling_type='AVG')
                # Dropout
                drop1 = inceprv2_builder.Dropout_layer(average_pooling)
                # Output
                output = inceprv2_builder.FC_layer(drop1, filters=self.build_params['Classes'], readout=True)
                self.ordered_logits = self.split_logits(output)
                return output

    def split_logits(self, logits):
    
        eyebrow_shape = logits[...,0:7]
        face_shape = logits[...,7:14]
        acc = logits[...,14:16]
        eye_open = logits[...,16:18]
        hair_vis = logits[...,18:20]
        mouth_open = logits[..., 20:22]
        smiling = logits[..., 22:24]
        lip_shape = logits[...,24:33]
        nose_shape = logits[...,33:41]
        skin_tone = logits[...,41:46]
        which_eye = logits[...,46:49]

        return eyebrow_shape, face_shape, acc, eye_open, hair_vis, mouth_open, smiling, lip_shape, nose_shape, skin_tone, which_eye



    def construct_loss(self):
        if self.output is None:
            self.set_output()
        logit_loss = 0 
        aux_logit_loss = 0
        for index, key in enumerate(self.keys):
            logits = self.ordered_logits[index]
            gt_logits= self.ordered_gt_logits[index]

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_logits, logits=logits))
            #loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=gt_logits, logits=logits))
            tf.summary.scalar('loss/'+key, loss)
            #aux_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_logits, logits=aux_logits))
            #aux_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=gt_logits, logits=aux_logits))
            #tf.summary.scalar('aux_loss/'+key, aux_loss)
            #aux_logit_loss += aux_loss
            logit_loss += loss
        self.loss.append(logit_loss)
        #self.loss.append(aux_logit_loss)

        #total_loss = tf.reduce_mean(tf.losses.get_regularization_losses())
        #tf.summary.scalar('loss/total', total_loss)
        #self.loss.append(total_loss)

    def set_accuracy_op(self):
        for index, key in enumerate(self.keys):
            correct_prediction = tf.equal(tf.argmax(self.ordered_gt_logits[index], 1), tf.argmax(self.ordered_logits[index], 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy/' +key, accuracy)