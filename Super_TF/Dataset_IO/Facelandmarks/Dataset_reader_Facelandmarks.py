from utils.Dataset_reader import Dataset_reader
from Dataset_IO.Facelandmarks.Dataset_config_Facelandmarks import Dataset_config_Facelandmarks
import Dataset_IO.Facelandmarks.Dataset_Facelandmarks_pb2 as proto
import tensorflow as tf
import os
import random
import numpy as np
import copy
import cv2
from imgaug import augmenters as iaa

class Dataset_reader_Facelandmarks(Dataset_reader, Dataset_config_Facelandmarks):
    """description of class"""

    def __init__(self, filename=None, epochs=100, vocab=None):
        super().__init__()
        with tf.name_scope('Dataset_Facelandmarks_Reader') as scope:
            self.batch_size = tf.placeholder(tf.int32, name='Dataset_batch_size')
            self.open_dataset(filename=filename, epochs=epochs)
            self.mean_header_proto = proto.Image_set()
            dataset_path, dataset_name = os.path.split(filename)
            common_name, _ = os.path.splitext(dataset_name)
            mean_file_path = os.path.join(dataset_path,common_name +'_mean.proto')
            with open(mean_file_path,"rb") as mean_header_file:
                self.mean_header_proto.ParseFromString(mean_header_file.read())
            self.image_shape = [self.mean_header_proto.Image_headers.image_height, self.mean_header_proto.Image_headers.image_width, self.mean_header_proto.Image_headers.image_depth]
            mean_image_data = self.mean_header_proto.mean_data

            self.mean_image = tf.image.convert_image_dtype(tf.image.decode_image(mean_image_data), tf.float32)
            self.mean_image.set_shape(self.image_shape)
            self.images , self.landmarkgt, self.landmarkinit, self.true_images= self.batch_inputs()


    def pre_process_image(self,pre_process_op):
        with tf.name_scope('Pre_Processing_op') as scope:
            self.images = pre_process_op(self.images)
            self.true_images = pre_process_op(self.true_images)

    def single_read(self):
        features = tf.parse_single_example(self.serialized_example, features=self._Feature_dict)
        image = tf.image.decode_image(features[self._Image_handle])
        image.set_shape(self.image_shape)
        image = tf.image.convert_image_dtype(image, tf.float32)
        true_image = image
        mean_pix = [tf.reduce_mean(self.mean_image[:,:,0]), tf.reduce_mean(self.mean_image[:,:,1]), tf.reduce_mean(self.mean_image[:,:,2] )]
        image = image -mean_pix 
        #tf.reduce_mean(self.mean_image)
        image = tf.image.random_brightness(image, max_delta = 0.05)
        image = tf.image.random_contrast(image, lower = 0.95, upper = 1.05)
        image = tf.image.random_hue(image, max_delta = 0.01)
        image += tf.random_normal(tf.shape(image),stddev=0.02,dtype=tf.float32,seed=42,name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.image.per_image_standardization(image)
        landmark_gt = features[self._Landmarks_GT]
        landmark_init = features[self._Landmarks_init]
        
        return image, landmark_gt, landmark_init, true_image

    def batch_inputs(self):
        image , landmark_gt, landmark_init , true_image= self.single_read()
        images , landmark_gts, landmark_inits,  true_images = tf.train.shuffle_batch([image, landmark_gt, landmark_init, true_image], batch_size=self.batch_size, num_threads=8, capacity=100, min_after_dequeue=90)
        return images , landmark_gts, landmark_inits, true_images


    def apply_tansform_mat(self, landmarks, M):
        rot_pts = []
        for point in landmarks:
            x = point[0]
            y = point[1]

            qx = M[0,0]*x + M[0,1]*y + M[0,2]
            qy = M[1,0]*x + M[1,1]*y + M[1,2]
            rot_pts.append([qx, qy])
        return rot_pts

    def augment(self, image, landmark):
        #Affine transformation
        key = random.randint(0,3)
        w, h, _ = image.shape
        scale = random.uniform(-0.04, 0.04) + 1

        if key == 0:
            dis = image
            rot_pts = landmark
        elif key == 1:
            #rotation transformation
           
            angle = random.randint(-10,10)
            M = cv2.getRotationMatrix2D((w/2,h/2), 10, scale)
            dis = cv2.warpAffine(image, M, (w, h))
            rot_pts = self.apply_tansform_mat(landmark, M)

        elif key == 2:
            #translation transformation
            
            X_offset = random.randint(-40,40)
            Y_offset = random.randint(-40,40)
            M = np.float32([[scale, 0, X_offset],[0, scale, Y_offset]])
            dis = cv2.warpAffine(image, M, (w, h))
            rot_pts = self.apply_tansform_mat(landmark, M)

        elif key == 3:
            #rotation and translation transformation
            angle = random.randint(-10,10)
            M_rot = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
            X_offset = random.randint(-40,40)
            Y_offset = random.randint(-40,40)
            M_tr = np.float32([[scale, 0, X_offset],[0, scale, Y_offset]])
            dis = cv2.warpAffine(image, M_rot, (w, h))
            dis = cv2.warpAffine(dis, M_tr, (w, h))
            rot_pts = self.apply_tansform_mat(landmark, M_rot)
            rot_pts = self.apply_tansform_mat(rot_pts, M_tr)



        aug = iaa.SomeOf((0, None), [
        #iaa.AdditiveGaussianNoise(scale=(0, 0.002)),
        iaa.Noop(),
        iaa.GaussianBlur(sigma=(0.0, 1.5)),
        iaa.Dropout(p=(0, 0.2)),
        iaa.CoarseDropout(0.2, size_percent=(0.001, 0.2))
        ], random_order=True)
        dis = aug.augment_image(dis)

        return dis, rot_pts

    def train_time_augmentation(self, images, landmarks):
        aug_im = []
        aug_landmark = []
        for index, image, in enumerate(images):
            landmark = landmarks[index]
            a_im, a_lm = self.augment(image, landmark)
            aug_im.append(a_im)
            aug_landmark.append(a_lm)

        return aug_im, aug_landmark

    def next_batch(self, batch_size=1, sess=None):
        with tf.name_scope('Batch_getter') as scope:
            if sess is None :
                self.sess = tf.get_default_session()
            else:
                self.sess = sess


        images, landmark_gts, landmark_inits, true_images = self.sess.run([self.images, self.landmarkgt, self.landmarkinit, self.true_images], feed_dict={self.batch_size:batch_size})

        landmark_gts_decoded = []
        landmark_inits_decoded = []
        for index, gtpts in enumerate(landmark_gts):
            landmark_gts_decoded.append(gtpts.decode().split())
            landmark_inits_decoded.append(landmark_inits[index].decode().split())
        landmark_gts_decoded = np.reshape(landmark_gts_decoded, (-1, 106, 2)).astype(np.float) * 512
        landmark_inits_decoded = np.reshape(landmark_inits_decoded, (-1, 106, 2)).astype(np.float) * 512

        images, landmark_gts_decoded = self.train_time_augmentation(images, landmark_gts_decoded)
        return images, landmark_gts_decoded, landmark_inits_decoded, images