from utils.builder import Builder
import tensorflow as tf
from tensorflow import nn
from tensorflow.python.layers.core import Dense
from utils.Base_Archs.Base_RNN import Base_RNN
import cv2
import copy
import numpy as np
import sys
_slim_path = '/media/nvme/tfslim/models/research/slim'
sys.path.append(_slim_path)
slim = tf.contrib.slim
#from nets.nasnet import pnasnet
from nets import inception_resnet_v2
from tensorflow.python.platform import tf_logging as logging

class MDM(Base_RNN):
    """MDM for facial landmark tracking"""

    def __init__(self, kwargs):
        self.global_step=tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
        self.build_params = kwargs
        tf.add_to_collection('Global_Step', self.global_step)
        iter = kwargs['iter']

        #self.input_placeholder = tf.placeholder(tf.float32, [None, None, 3])
        #ass  = tf.assign(self.input_placeholder, iter['input'])
        self.class_tags, self.input_placeholder, self.target_seq_placeholder, self.input_seq_placeholder, \
            self.incep_mean = iter.get_next()

        self.class_image = self.input_placeholder
        self.ordered_gt_logits, self.ordered_gt_lms = self.split_logits(self.class_tags, False)
        #self.input_seq_placeholder = iter['mean_pts']
        #self.target_seq_placeholder = iter['target_pts']
        #tf.summary.image('crap', self.input_placeholder)
        self.input_seq_placeholder.set_shape([self.build_params['Batch_size'], kwargs['Patches'], 2])
        self.target_seq_placeholder.set_shape([self.build_params['Batch_size'], kwargs['Patches'], 2])



        print(self.target_seq_placeholder.get_shape().as_list())

        #self.iter_train_op = kwargs['train_op']
        #self.iter_test_op =kwargs['val_op']
        self.Init_train = False
        #self.input_placeholder = tf.placeholder(tf.float32, shape=[self.build_params['Batch_size'], kwargs['Image_width'], kwargs['Image_height'],kwargs['Image_cspace']], name='Input')
        #self.input_seq_placeholder = tf.placeholder(tf.float32, shape=[self.build_params['Batch_size'], kwargs['Patches'], 2], name='Input_Seq') 
        
        #self.target_seq_placeholder = tf.placeholder(tf.float32, shape=[self.build_params['Batch_size'], kwargs['Patches'], 2], name='Target_Seq') 
        
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.state_placeholder = tf.placeholder(tf.string, name='State')
        #self.tiled_inputs = tf.tile(self.input_placeholder, [kwargs['Patches'], 1, 1, 1])
        self.output = None
        self.loss = []
        self.deltas = []
        self.accuracy = None
        self.Init_predict = True
        self.celeba = True
        self.training = self.build_params['Training']

    def fine_tune(self):
        session = tf.get_default_session()
        print('loading finetune weights')
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'biases']
        tvs = slim.get_variables_to_restore(exclude = exclude)
        #tvs = tf.trainable_variables()
        #init_fn = slim.assign_from_checkpoint_fn('/media/nvme/TFmodels/Pnas/i/i.ckpt', tvs, ignore_missing_vars=True)
        init_fn = slim.assign_from_checkpoint_fn('/media/nvme/TFmodels/celeba/logs/model.ckpt-22083', tvs, ignore_missing_vars=True)
        init_fn(session)
        #tvs  = [v for v in tf.trainable_variables() if 'biases'  not in v.name and 'InceptionResnetV2/Auxlogits' not in v.name and 'final_layer' not in v.name]
        #saver = tf.train.Saver(tvs)
        #latest_ckpt = tf.train.latest_checkpoint('/media/nvme/TFmodels/Pnas/mdl-ft/')
        #saver.restore(session, latest_ckpt)

    def get_central_crop(self, input, crop_size):
        _, w, h, _ = input.get_shape().as_list()
        half_box = (crop_size[0]/2., crop_size[1]/2.)
        a = slice(int((w // 2) - half_box[0]), int((w // 2) + half_box[0]))
        b = slice(int((h // 2) - half_box[1]), int((h // 2) + half_box[1]))
        return input[:, a, b, :]


    def build_conv_model2(self, inputs):
        conv_settings = dict(padding='SAME', num_outputs=32, kernel_size=3,
            weights_initializer=slim.initializers.xavier_initializer_conv2d(False),
            weights_regularizer=slim.l2_regularizer(5e-5)
        )

        with tf.variable_scope('convnet'):
            with slim.arg_scope([slim.conv2d], **conv_settings):
                with slim.arg_scope([slim.max_pool2d], kernel_size=2):
                    net = slim.conv2d(inputs, scope='conv_1')
                    net = slim.max_pool2d(net)
                    net = slim.conv2d(net, scope='conv_2')
                    net = slim.max_pool2d(net)
                    net = slim.conv2d(net, scope='conv_3')
        return net

    def build_conv_model(self, inputs):
        with tf.variable_scope('Patches', reuse=tf.AUTO_REUSE):
            with Builder(**self.build_params) as conv_builder:
                print('inp shape', inputs.get_shape().as_list())
                #inputs = tf.stop_gradient(inputs)
                inputs = tf.reshape(inputs, (self.build_params['Batch_size'] , self.build_params['Patches'] * 32, 32, 3))
                print('inp shape', inputs.get_shape().as_list())
                #input('wait')
                self.patches = inputs
                conv1 = conv_builder.Conv2d_layer(inputs, filters=32, Batch_norm=False)
                pool1 = conv_builder.Pool_layer(conv1)
                conv2 = conv_builder.Conv2d_layer(pool1, filters=32, Batch_norm=False)
                pool2 = conv_builder.Pool_layer(conv2)

                crop_size = pool2.get_shape().as_list()[1:3]
                cropped_conv2 = self.get_central_crop(conv2, crop_size)

                output = conv_builder.Concat([cropped_conv2, pool2])
                return output

    def get_preds(self, inputs, hidden_state=None):
        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
            with Builder(**self.build_params) as rnn_builder:
                image_embeddings = tf.reshape(inputs, (self.build_params['Batch_size'], -1))
                features = rnn_builder.Concat([image_embeddings, hidden_state], axis=1)
                hidden_state = tf.tanh(rnn_builder.FC_layer(features, filters=256, flatten=False, readout=True))
                processed_preds = rnn_builder.FC_layer(hidden_state, filters=2*self.build_params['Patches'], readout=True, name='Preds')

                return processed_preds, hidden_state

    def build_sampling_grid(self, patch_shape):
        patch_shape = np.array(patch_shape)
        patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
        start = -patch_half_shape
        end = patch_half_shape
        sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
        return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)

    def extract_patches_image(self, image, centres):
        """ Extracts patches from an image.
        Args:
            pixels: a `Tensor` of dimensions [batch_size, height, width, channels]
            centres: a `Tensor` of dimensions [batch_size, num_patches, 2]
            sampling_grid: `ndarray` (patch_width, patch_height, 2)
        Returns:
            a `Tensor` [num_patches, height, width, channels]
        """
        sampling_grid=self.build_sampling_grid((32, 32))
        max_y = tf.shape(image)[0]
        max_x = tf.shape(image)[1]

        patch_grid = tf.to_int32(sampling_grid[None, :, :, :] + centres[:,  None, None, :])
        Y = tf.clip_by_value(patch_grid[:, :, :, 0], 0, max_y - 1)
        X = tf.clip_by_value(patch_grid[:, :, :, 1], 0, max_x - 1)

        return tf.gather_nd(image, tf.transpose(tf.stack([Y, X]), (2, 3, 1, 0)))

    def extract_patches(self, centres):
        #centres = centres * 512
        sampling_grid=self.build_sampling_grid((32, 32))
        #print('center shape', centres.get_shape().as_list())
        batch_size = self.input_placeholder.get_shape().as_list()[0]
        patches = tf.stack([self.extract_patches_image(self.input_placeholder[i], centres[i])
                           for i in range(self.build_params['Batch_size'])])
        return tf.transpose(patches, [0, 3, 1, 2, 4])

    def get_image_patches(self, patch_shape, inds):
         patches = self.extract_patches(inds)
         print('patches shape', patches.get_shape().as_list())
         return patches

    def split_logits(self, logits, output=True):
            celeba_attrs = logits[..., 0:self.build_params['Classes']]
            celeba_lms = logits[..., self.build_params['Classes']: self.build_params['Classes'] + self.build_params['core_pts']]
            if output:
                celeba_lms *= 20
                celeba_lms += self.incep_mean
            celeba_lms = tf.reshape(celeba_lms, [-1, 5, 2])
            return celeba_attrs, celeba_lms

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

    def transform_pts(self, M, pts):
        out_pts = []
        appe = np.ones([pts.shape[0], 1], np.float32)

        pts = np.concatenate((pts, appe), -1)
        pts = np.transpose(pts)
        pts = np.matmul(M, pts)
        pts = np.transpose(pts)
        for pt in pts:
            x = pt[0] / pt[2]
            y = pt[1] / pt[2]
            out_pts.append([x, y])

        return np.asarray(out_pts)

    def align_input_shape(self, incep_pts, gt_pts, init_mdm_pts):

        out_init_mdm_pts = []
        for index, inceppts in enumerate(incep_pts):
            gtpts = gt_pts[index]
            initmdmpts = init_mdm_pts[index]

            inceppts = np.reshape(inceppts, [-1, 2])
            gtpts = np.reshape(gtpts, [-1, 2])
            M, mask = cv2.findHomography(inceppts, gtpts)
            M = np.asarray(M)
            initmdmpts = self.transform_pts(M, initmdmpts)
            out_init_mdm_pts.append(initmdmpts)
        out_init_mdm_pts = np.asarray(out_init_mdm_pts, np.float32)
        #out_init_mdm_pts = np.expand_dims(out_init_mdm_pts, axis=0)
        return out_init_mdm_pts

    def build_net(self):
        #sampling_grid = self.build_sampling_grid(patch_shape)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            net, endpoints = inception_resnet_v2.inception_resnet_v2(self.class_image, num_classes=self.build_params['Classes'], is_training=self.training, create_aux_logits=False)
            self.fine_tune()
            vectors = endpoints['Conv2d_7b_1x1']
            vectors_flat = slim.flatten(vectors)
            hidden_state = slim.fully_connected(vectors_flat, 512, activation_fn=None, scope='Hidden_state')
            self.output = net
            self.ordered_logits, self.ordered_lms = self.split_logits(self.output)
            init_mdm_pts = tf.py_func(self.align_input_shape,
                                      [self.incep_mean, self.ordered_lms, self.input_seq_placeholder], tf.float32)

        #hidden_state = tf.zeros((self.build_params['Batch_size'], 512))
        deltas = tf.zeros((self.build_params['Batch_size'], self.build_params['Patches'], 2))
        self.predictions = []
        for step in range(3):
            with tf.device('/cpu:0'):
                    patches = self.get_image_patches((32,32), self.input_seq_placeholder + deltas)
                    patches = tf.stop_gradient(patches)     
                    patches = tf.reshape(patches, (self.build_params['Batch_size'] , self.build_params['Patches'] * 32, 32, 3))       

            with tf.variable_scope('convnet', reuse=tf.AUTO_REUSE):
                features = self.build_conv_model2(patches)   

            features = tf.reshape(features, (self.build_params['Batch_size'], -1))

            with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE) as scope:
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(5e-5)):
                    hidden_state = slim.fully_connected(tf.concat([features, hidden_state], axis=1), 512, activation_fn=tf.tanh)
                prediction = slim.linear(hidden_state, self.build_params['Patches'] * 2, scope='pred')
                if step is 2: #0
                    prediction *=1.5
            prediction = tf.reshape(prediction, (self.build_params['Batch_size'] , self.build_params['Patches'], 2))
            deltas += prediction
            self.predictions.append(init_mdm_pts + deltas)

    '''
    def build_net(self):
        with tf.variable_scope('MDM', reuse=tf.AUTO_REUSE):
            with Builder(**self.build_params) as MDM_builder:
                #BUILD CONV MODEL HERE get image embeddings
                #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(512, tf.tanh, tf.AUTO_REUSE)
                #rnn_state = rnn_cell.zero_state(tf.shape(image_embeddings)[0], dtype=tf.float32)
                #self.rnn_state = rnn_state

                delta = tf.zeros((tf.shape(self.target_seq_placeholder)[0], self.build_params['Patches'], 2))
                hidden_state = tf.zeros((self.build_params['Batch_size'], 256))
                for i in range(4):
                    patches = self.get_image_patches((32,32), self.input_seq_placeholder + delta)
                    patches = tf.stop_gradient(patches)
                    image_embeddings = self.build_conv_model(patches)
                    print(image_embeddings.get_shape().as_list())
                    image_embeddings = tf.reshape(image_embeddings, (self.build_params['Batch_size'], -1))
                    #features = MDM_builder.Concat([image_embeddings, hidden_state], axis=1)
                    #hidden_state = tf.tanh(MDM_builder.FC_layer(features, filters=256, flatten=False))
                    #processed_preds = MDM_builder.FC_layer(hidden_state, filters=2*106, readout=True, name='Preds')

                    #input('wait')
                    #predictions,  rnn_state = rnn_cell(image_embeddings,  rnn_state)
                    #self.fs = rnn_state
                    #processed_preds = MDM_builder.FC_layer(predictions, filters=2*106, readout=True, name='Preds')
                    
                    processed_preds, hidden_state = self.get_preds(image_embeddings, hidden_state)
                    output_preds = tf.reshape(processed_preds, [tf.shape(self.target_seq_placeholder)[0], self.build_params['Patches'], 2])
                    print(output_preds.get_shape().as_list())
                    #input('wait')
                    delta += output_preds 
                    self.deltas.append(delta)
                    #patches = self.get_image_patches((32,32), self.input_seq_placeholder + self.deltas[-1])
                    #image_embeddings = self.build_conv_model(patches)
                    inds= self.input_seq_placeholder + self.deltas[-1]
                    inds = tf.reshape(self.target_seq_placeholder, (self.build_params['Batch_size']*106, 2))
                    self.inds = inds
                    #inds = tf.constant((0.5258215962441315, 0.5421245421245421))
                    
    '''  

    def construct_IO_dict(self, batch):
        return {self.input_placeholder: batch[0], self.input_seq_placeholder: batch[2], self.target_seq_placeholder: batch[1]}

    def draw_core_landmarks(self, image, pts):
        for pt in pts:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0, 1, 0))

        return image

    def draw_landmarks(self, image, pts1, pts2, pts, trg):
            image1 = np.copy(image)
            image2 = np.copy(image)
            pts = pts
            for ind, pt in enumerate(pts):
                    tpt = trg[ind]
                    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0,0,1), 5, -1)
                    #cv2.circle(image, (int(tpt[0]), int(tpt[1])), 3, (1,0,0), 5)
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 4, (1, 0, 0))
                    cv2.line(image, (int(pt[0]), int(pt[1])), (int(tpt[0]), int(tpt[1])), (0, 0, 1))

            pts = pts1
            for ind, pt in enumerate(pts):
                    tpt = trg[ind]
                    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0,0,1), 5, -1)
                    #cv2.circle(image, (int(tpt[0]), int(tpt[1])), 3, (1,0,0), 5)
                    cv2.circle(image1, (int(pt[0]), int(pt[1])), 4, (1, 0, 0))
                    cv2.line(image1, (int(pt[0]), int(pt[1])), (int(tpt[0]), int(tpt[1])), (0, 0, 1))

            pts = pts2
            for ind, pt in enumerate(pts):
                    tpt = trg[ind]
                    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0,0,1), 5, -1)
                    #cv2.circle(image, (int(tpt[0]), int(tpt[1])), 3, (1,0,0), 5)
                    cv2.circle(image2, (int(pt[0]), int(pt[1])), 4, (1, 0, 0))
                    cv2.line(image2, (int(pt[0]), int(pt[1])), (int(tpt[0]), int(tpt[1])), (0, 0, 1))

            return image1, image2, image


    def set_output(self):


        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=self.training):
            self.build_net()
        '''
        self.Predict_op =   self.input_seq_placeholder + self.deltas[0]
        self.Predict_op2 =  self.input_seq_placeholder + self.deltas[1]
        self.Predict_op3 =  self.input_seq_placeholder + self.deltas[2]
        self.Predict_op4 =  self.input_seq_placeholder +  self.deltas[3]
        '''
        self.Predict_op =   self.predictions[0]
        self.Predict_op2 =  self.predictions[1]
        self.Predict_op3 =  self.predictions[2]


        #self.Predict_op5 =  self.input_seq_placeholder +  self.deltas[4]
        images = self.input_placeholder[0]
        pts = self.Predict_op3[0]

        im, im2, im3 = tf.py_func(self.draw_landmarks, [images, self.Predict_op[0], self.Predict_op2[0], pts, self.target_seq_placeholder[0]], (tf.float32, tf.float32, tf.float32))
        core_im = tf.py_func(self.draw_core_landmarks, [self.input_placeholder[0], self.ordered_lms[0]], (tf.float32))
        core_im = tf.expand_dims(core_im, axis=0)
        #im1 = tf.py_func(self.draw_landmarks, [images, self.Predict_op[0], self.target_seq_placeholder[0]], tf.float32)
        #im2 = tf.py_func(self.draw_landmarks, [images, self.Predict_op2[0], self.target_seq_placeholder[0]], tf.float32)
        im = tf.expand_dims(im, axis=0)
        im2 = tf.expand_dims(im2, axis=0)
        im3 = tf.expand_dims(im3, axis=0)
        im_final = tf.concat(axis=2, values=[im, im2, im3])
        tf.summary.image('core', core_im)
        tf.summary.image('predictions', im_final)
        tf.summary.image('outim', im3)
        #self.Predict_op5 =  self.input_seq_placeholder +  self.deltas[4]

    def predict(self, **kwargs):
        session = kwargs['session']
        if self.Init_predict:
            self.set_output()
            variables_to_restore = slim.get_model_variables()
            latest_ckpt = tf.train.latest_checkpoint(self.build_params['Save_dir'] + '/logs/')
            init_fn = slim.assign_from_checkpoint_fn(latest_ckpt, variables_to_restore,ignore_missing_vars=True)
            self.eput = tf.sigmoid(self.ordered_logits)
            init_fn(kwargs['session'])
            #session.run(self.iter_train_op)
            #self.Init_train = True
        #predict_io_dict = {self.input_placeholder: kwargs['Input_Im'], self.input_seq_placeholder:kwargs['Input_seq']}
        predict_feed_dict = self.construct_control_dict(Type='Test')
        #predict_feed_dict = {**predict_io_dict, **test_dict}
        index = 0
        l1_loss = 0.0
        counter = 1
        smiling = 0.0
        accesory = 0.0
        while(1):
            pts, GT, image, tags, gt_tags =  session.run([self.Predict_op3, self.target_seq_placeholder, self.input_placeholder, self.eput[0], self.class_tags[0]], feed_dict=predict_feed_dict)
            pts = pts[0]
            GT = GT[0]
            image = image[0]
            image /= 2
            image += 0.5
            image *= 255
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512), 0)
            distance = 0.0
            pts *= 512/ self.build_params['Image_width']
            GT *= 512/ self.build_params['Image_width']
            for i , pt in enumerate(pts):

                gpt = GT[i]
                distance += np.sqrt(np.square(pt[0] - gpt[0]) + np.square(pt[1] - gpt[1]))
                pred_pt = (int(pt[0]), int(pt[1]))
                grnd_pt = (int(gpt[0]), int(gpt[1]))
                cv2.circle(image, pred_pt, 2, (255,0,0))
                cv2.line(image, pred_pt, grnd_pt, (0,0,255))

            distance /= self.build_params['Patches']
            l1_loss += distance
            print('L1_Loss: ', l1_loss/ counter)


            cv2.imwrite('/media/Disk3/Projects/PersonalGit/Tensorflow_Playground/Super_TF/out/' + str(index) + '.jpg', image)
            index +=1
            thre, rtr = cv2.threshold(tags, 0.5, 1, cv2.THRESH_BINARY)
            rtr = np.squeeze(rtr)
            rtr2 = rtr.astype(np.bool)
            gt = gt_tags.astype(np.bool)
            if rtr2[0] == gt[0]:
                smiling +=1
            if rtr2[1] == gt[1]:
                accesory +=1

            print('Smiling: ', smiling / counter)
            print('Accesory: ', accesory / counter)
            counter += 1
            print(gt)

            print(rtr2)
            print(tags)
            #input('cra')
            #cv2.imshow('image', image)
            #cv2.waitKey(0)


    def construct_loss(self):
        if self.output is None:
            self.set_output()

        key = [1, 1.5, 2]
        '''
        norm = tf.sqrt(tf.reduce_sum(((self.target_seq_placeholder[:, 19, :] - self.target_seq_placeholder[:, 93, :])**2), 1))
        for indx, delta in enumerate(self.deltas):
            norm_rms_loss = tf.reduce_sum(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(self.input_seq_placeholder + delta - self.target_seq_placeholder), 2)), 1) / norm*106)
            tf.summary.scalar('norm_loss_level_' + str(indx), norm_rms_loss)
            self.loss.append(norm_rms_loss)
        '''
        def normalized_rmse(pred, gt_truth, n_landmarks=self.build_params['Patches'], left_eye_index=75, right_eye_index=85):
            norm = tf.sqrt(1e-12 + tf.reduce_sum(((gt_truth[:, left_eye_index, :] - gt_truth[:, right_eye_index, :]) ** 2), 1))

            return tf.reduce_sum(tf.sqrt(1e-12 + tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * n_landmarks)

        if self.celeba:
            multi_class_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.ordered_gt_logits, logits=self.ordered_logits)*0.2
            tf.summary.scalar('losses/multi_class_loss', multi_class_loss)
            lm_loss = normalized_rmse(self.ordered_lms, self.ordered_gt_lms, n_landmarks = self.build_params['core_pts'], left_eye_index = 0, right_eye_index = 1)
            lm_loss = tf.losses.compute_weighted_loss(lm_loss)
            tf.summary.scalar('loss/incep_landmarks', tf.reduce_mean(lm_loss))
        gt_truth = self.target_seq_placeholder


        for i, prediction in enumerate(self.predictions):

            norm_error = normalized_rmse(prediction, gt_truth) * key[i]
            #mse_error =  tf.reduce_mean(tf.reduce_mean(tf.reduce_sum((tf.abs(prediction - self.target_seq_placeholder)), -1), -1) , -1) /10
            #mse_error = tf.losses.compute_weighted_loss(mse_error)
            #tf.summary.scalar('losses/mes_step_{}'.format(i), mse_error)
            norm_error = tf.reduce_mean(norm_error)
            norm_error = tf.losses.compute_weighted_loss(norm_error)
            tf.summary.scalar('losses/step_{}'.format(i), norm_error)
            #abs_val = tf.abs(self.target_seq_placeholder - prediction)

            #l1_loss = tf.reduce_mean(tf.losses.absolute_difference(labels=self.target_seq_placeholder, predictions=prediction)/)
            #tf.summary.scalar('losses/l1_loss_step_{}'.format(i), l1_loss)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total_loss', total_loss)
        self.loss.append(total_loss)
        '''
        for indx, delta in enumerate(self.deltas):
            pred = self.input_seq_placeholder + delta
            print(gt_truth.get_shape().as_list())
            assert pred.get_shape()[1] == gt_truth.get_shape()[1] == n_landmarks
            self.loss.append(tf.reduce_mean(tf.sqrt(1e-12 + tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (106))
        '''

    def train(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']
        #batch = kwargs['data']
        if not self.Init_train:
            self.construct_loss()
            print('Train')
            #session.run(self.iter_train_op)
            self.Init_train = True

        self.set_accuracy_op()
        self.loss = tf.add_n(self.loss, 'Loss_accu')
        global_step_tensor = slim.create_global_step()
        learning_rate = tf.train.exponential_decay(0.01, global_step_tensor, decay_steps=3000, decay_rate=0.94, staircase=False)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1, beta1=0.9, beta2=0.99)
        #batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        #IO_feed_dict = self.construct_IO_dict(batch)
        train_op = slim.learning.create_train_op(self.loss, optimizer, summarize_gradients=False)
        logging.set_verbosity(10)
        slim.learning.train(
            train_op= train_op,
            logdir = '/media/nvme/TFmodels/Pnas/logs/',
            number_of_steps=100000,
            save_summaries_secs=30,
            save_interval_secs=3600,
            global_step = global_step_tensor)
        #batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        #IO_feed_dict = self.construct_IO_dict(batch)
        train_feed_dict = self.construct_control_dict(Type='Train')
        #train_feed_dict = {**train_dict}
        if kwargs['merged'] is None:
            _, GT, out, out1, out2, out3 = session.run([self.train_step, self.target_seq_placeholder, self.Predict_op, self.Predict_op2 , self.Predict_op3 , self.Predict_op4], feed_dict=train_feed_dict)
            summary = None
        else:
            _, GT, summary, out, out1, out2, out3  = session.run([self.train_step, self.target_seq_placeholder, kwargs['merged'], self.Predict_op, self.Predict_op2 , self.Predict_op3 , self.Predict_op4], feed_dict=train_feed_dict)
        
        #disp_image = disp_image[0]
        print('GT')
        print(GT[0][19])
        #print('meanpt')
        #print(batch[2][0][19])
        print('Output3')
        print(out2[0][19])
        print('Output4')
        print(out3[0][19])
 
        #disp_image = cv2.cvtColor(batch[-1][0], cv2.COLOR_RGB2BGR) 

        '''
        points = copy.deepcopy(out2[0])
        for point in points:
            #point += 0.5
            point = [int(point[0]), int(point[1])]
            cv2.circle(disp_image, (point[0], point[1]), 4, (255,0,0), 2)
            #cv2.rectangle(disp_image, pt1 = (point[0]-16, point[1]-16), pt2 = (point[0]+16, point[1]+16), color=(0,0,255))
       '''


        '''
        points2 = copy.deepcopy(GT[0])  
        for point in points2:
                point = [int(point[0]), int(point[1])]
                cv2.circle(disp_image, (point[0], point[1]), 4, (255,0,255), 2)
         '''   

        
        #cv2.imshow('crap', cv2.cvtColor(pat, cv2.COLOR_BGR2RGB))
        #disp_image = cv2.resize(disp_image, (512, 512),0)
        #cv2.imshow('image', disp_image)

        
        '''
        print('Output1')
        print(out[0][19])
        print('Output2')
        print(out1[0][19])

        print('Output4')
        '''
        #cv2.waitKey(1)
        return summary
    def test(self, **kwargs):
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']
        #batch = kwargs['data']
        #session.run(self.iter_test_op)
        #batch = kwargs['data'].next_batch(self.build_params['Batch_size'])
        #IO_feed_dict = self.construct_IO_dict(batch)
        test_feed_dict = self.construct_control_dict(Type='Test')
        #test_feed_dict = {**IO_feed_dict, **test_dict}
        '''
        if self.accuracy is not None:
            summary, _ = session.run([kwargs['merged'], self.accuracy], feed_dict=test_feed_dict)

        else:
            summary = session.run([kwargs['merged']], feed_dict=test_feed_dict)[0]

        return summary
        '''

    def set_accuracy_op(self):
        if not self.celeba:
            for index, key in enumerate(self.keys):
                correct_prediction = tf.equal(tf.argmax(self.ordered_gt_logits, 1), tf.argmax(self.ordered_logits, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy/' +key, accuracy)
        else:
            sigmoid = tf.nn.sigmoid(self.ordered_logits)
            sigmoid = tf.ceil(sigmoid - 0.5 + 1e-10)
            intersection = tf.reduce_sum(sigmoid * self.ordered_gt_logits, axis=1) + 1e-10  # per instance
            union = tf.reduce_sum(sigmoid, axis=1) + tf.reduce_sum(self.ordered_gt_logits, axis=1) + 1e-10
            accuracy = tf.reduce_mean((2 * intersection) / union)
            tf.summary.scalar('accuracy/', accuracy)