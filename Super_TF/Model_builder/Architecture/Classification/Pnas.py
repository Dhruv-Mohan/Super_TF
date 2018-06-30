from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Classifier import Base_Classifier
import sys
_slim_path = '/media/nvme/tfslim/models/research/slim'
sys.path.append(_slim_path)
slim = tf.contrib.slim
#from nets.nasnet import pnasnet
from nets import inception_resnet_v2
from nets import inception_v3
import cv2
import pickle
pic_in = open("/media/nvme/Datasets/Celeba/mean_face.pickle", "rb")
mean_celeba = pickle.load(pic_in)




class Pnas(Base_Classifier):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.ordered_gt_logits, self.ordered_gt_lms_logits = self.split_logits(self.output_placeholder, output=False)
        #self.ordered_gt_logits = self.output_placeholder
        self.keys = ['is_smiling']
        #, 'face_shape', 'is_acc', 'is_eye_open', 'is_hair_vis', 'is_mouth_open', 'is_smiling', 'lip_shape', 'nose_shape',
        # 'skin_tone', 'which_eye']
        self.Init_predict = True
        self.celeba = True
    def build_net(self):
        #net, endpoints = pnasnet.build_pnasnet_large(self.input_placeholder, num_classes=49, is_training=True, final_endpoint='Logits')
        #scope = inception_resnet_v2.inception_resnet_v2_arg_scope(batch_norm_decay=0.99)
        #net, endpoints = inception_v3.inception_v3(self.input_placeholder, num_classes=self.build_params['Classes'], is_training=True, create_aux_logits=False)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            net, endpoints = inception_resnet_v2.inception_resnet_v2(self.input_placeholder, num_classes=self.build_params['Classes'], is_training=self.training, create_aux_logits=False)
        self.output = net
        self.ordered_logits , self.ordered_lms_logits = self.split_logits(self.output)

        #self.ordered_logits = self.output
        #aux_logits = endpoints['AuxLogits']
        #self.ordered_aux_logits = self.split_logits(aux_logits)
        self.endpoints = endpoints

    def fine_tune(self):
        session = tf.get_default_session()
        print('loading finetune weights')
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'biases']
        tvs = slim.get_variables_to_restore(exclude = exclude)
        #tvs = tf.trainable_variables()
        init_fn = slim.assign_from_checkpoint_fn('/media/nvme/TFmodels/Pnas/i/i.ckpt', tvs, ignore_missing_vars=True)
        #init_fn = slim.assign_from_checkpoint_fn('/media/nvme/TFmodels/Pnas/celeba/model.ckpt-100000', tvs, ignore_missing_vars=True)
        init_fn(session)
        #tvs  = [v for v in tf.trainable_variables() if 'biases'  not in v.name and 'InceptionResnetV2/Auxlogits' not in v.name and 'final_layer' not in v.name]
        #saver = tf.train.Saver(tvs)
        #latest_ckpt = tf.train.latest_checkpoint('/media/nvme/TFmodels/Pnas/mdl-ft/')
        #saver.restore(session, latest_ckpt)

    def split_logits(self, logits, output=True):
            celeba_attrs = logits[...,0:40]
            celeba_lms = logits[...,40:50]
            if output:
                celeba_lms = logits[..., 40:50] *20
                celeba_lms += mean_celeba
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


    def construct_loss(self):

        def normalized_rmse(pred, gt_truth):
            n_landmarks = 5
            left_eye_index = 0
            right_eye_index = 1
            norm = tf.sqrt(
                1e-12 + tf.reduce_sum(((gt_truth[:, left_eye_index, :] - gt_truth[:, right_eye_index, :]) ** 2), 1))

            return tf.reduce_sum(tf.sqrt(1e-12 + tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (
                        norm * n_landmarks)


        if self.output is None:
            self.set_output()
        if self.celeba:
            tf.losses.sigmoid_cross_entropy(multi_class_labels=self.ordered_gt_logits, logits=self.ordered_logits)
            lm_loss = normalized_rmse(self.ordered_lms_logits, self.ordered_gt_lms_logits) *5
            lm_loss = tf.losses.compute_weighted_loss(lm_loss)
            tf.summary.scalar('loss/landmarks', tf.reduce_mean(lm_loss))
            #tf.losses.sigmoid_cross_entropy(multi_class_labels=self.ordered_aux_logits, logits=self.ordered_logits)
        else:
            logit_loss = 0
            aux_logit_loss = 0
            for index, key in enumerate(self.keys):
                logits = self.ordered_logits
                gt_logits= self.ordered_gt_logits
                #aux_logits = self.ordered_aux_logits
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_logits, logits=logits))
                #loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=gt_logits, logits=logits))
                tf.summary.scalar('loss/'+key, loss)
                #aux_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_logits, logits=aux_logits))
                #aux_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=gt_logits, logits=aux_logits))
                #tf.summary.scalar('aux_loss/'+key, aux_loss)
                #aux_logit_loss += aux_loss
                logit_loss += loss
            self.loss.append(logit_loss)
        self.loss.append(slim.losses.get_total_loss())
        #self.loss.append(aux_logit_loss)

        #total_loss = tf.reduce_mean(tf.losses.get_regularization_losses())
        #tf.summary.scalar('loss/total', total_loss)
        #self.loss.append(total_loss)
    def draw_landmarks(self, image, pts, trg):
            pts = pts
            for ind, pt in enumerate(pts):
                    tpt = trg[ind]
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 4, (0.5, 0, 0))
                    cv2.line(image, (int(pt[0]), int(pt[1])), (int(tpt[0]), int(tpt[1])), (0, 0, 0.5))
            return image

    def set_accuracy_op(self):
        if not self.celeba:
            for index, key in enumerate(self.keys):
                correct_prediction = tf.equal(tf.argmax(self.ordered_gt_logits, 1), tf.argmax(self.ordered_logits, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy/' +key, accuracy)

        else:
            images = self.input_placeholder[0]
            im = tf.py_func(self.draw_landmarks,
                       [images, self.ordered_lms_logits[0], self.ordered_gt_lms_logits[0]],
                       (tf.float32))
            im = tf.expand_dims(im, axis=0)
            tf.summary.image('landmark', im)
            sigmoid = tf.nn.sigmoid(self.ordered_logits)
            sigmoid = tf.ceil(sigmoid - 0.5 + 1e-12)
            intersection = tf.reduce_sum(sigmoid * self.ordered_gt_logits, axis=1) + 1e-12  # per instance
            union = tf.reduce_sum(sigmoid, axis=1) + tf.reduce_sum(self.ordered_gt_logits, axis=1) + 1e-12
            accuracy = tf.reduce_mean((2 * intersection) / union)
            tf.summary.scalar('accuracy/', accuracy)


    def predict(self, **kwargs):
        if self.Init_predict:
            self.set_output()
            variables_to_restore = slim.get_model_variables()
            latest_ckpt = tf.train.latest_checkpoint(self.build_params['Save_dir'] + '/logs/')
            init_fn = slim.assign_from_checkpoint_fn(latest_ckpt, variables_to_restore,ignore_missing_vars=True)
            self.eput = tf.sigmoid(self.ordered_logits)
            init_fn(kwargs['session'])
            self.Init_predict = False
        if kwargs['session'] is None:
            session = tf.get_default_session()
        else:
            session = kwargs['session']
        return session.run([self.eput[0], self.ordered_gt_logits[0], self.input_placeholder, self.ordered_lms_logits[0], self.ordered_gt_lms_logits[0]])