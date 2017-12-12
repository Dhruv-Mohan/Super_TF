from utils.Dataset_reader import Dataset_reader
from Dataset_IO.Segmentation.Dataset_config_segmentation import Dataset_config_segmentation
import Dataset_IO.Segmentation.Dataset_segmentation_pb2 as proto
import tensorflow as tf
import os


#TODO: ADD TFRECORDS AND MEANPROTO READING CHECKS
class Dataset_reader_segmentation(Dataset_reader, Dataset_config_segmentation):
    """Implementation of Dataset reader for segmentation"""


    def __init__(self, filename=None, epochs=100, num_classes=1):

        super().__init__()
        with tf.name_scope('Dataset_Segmentation_Reader') as scope:
            self.batch_size = tf.placeholder(tf.int32, name='Dataset_batch_size')
            self.num_classes = num_classes
            self.open_dataset(filename=filename, epochs=epochs)
            self.mean_header_proto = proto.Image_set()
            dataset_path, dataset_name = os.path.split(filename)
            common_name, _ = os.path.splitext(dataset_name)
            mean_file_path = os.path.join(dataset_path,common_name +'_header.proto')
            
            with open(mean_file_path,"rb") as mean_header_file:
                self.mean_header_proto.ParseFromString(mean_header_file.read())
            self.flip_prob = tf.Variable(tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32),trainable=False)
            self.crop_prob = tf.Variable(tf.random_uniform(shape=[1],  minval=0, maxval=1, dtype=tf.float32),trainable=False)
            self.crop_val = tf.Variable(tf.random_uniform(shape=[1], minval=1.1, maxval=1.25, dtype=tf.float32),trainable=False)
            self.init_randoms = tf.group(self.flip_prob.initializer, self.crop_prob.initializer, self.crop_val.initializer)
            self.sess = None

            self.image_shape = [self.mean_header_proto.Image_headers.image_width, self.mean_header_proto.Image_headers.image_height, self.mean_header_proto.Image_headers.image_depth]
            self.mask_shape = [self.mean_header_proto.Image_headers.image_width, self.mean_header_proto.Image_headers.image_height, 1]
            self.images , self.masks , self.mask_weights, self.names = self.batch_inputs()

    def single_read(self):
        features = tf.parse_single_example(self.serialized_example, features=self._Feature_dict)
        image = tf.image.decode_image(features[self._Image_handle])
        mask = tf.image.decode_image(features[self._Image_mask])
        mask_weight = tf.image.decode_image(features[self._Mask_weights])
        name = features[self._Image_name]
        mask_weight.set_shape(self.image_shape)
        image.set_shape(self.image_shape)
        mask.set_shape(self.image_shape)
        

        image = tf.image.convert_image_dtype(image, tf.float32)
        mask = tf.squeeze(tf.image.convert_image_dtype(mask, tf.float32))
        #mask_weight = tf.squeeze(tf.image.convert_image_dtype(mask_weight, tf.float32))
        #mask_weight = tf.expand_dims(mask_weight,2)

        

        '''
        flip_op =tf.group(tf.assign(image,tf.image.flip_left_right(image)),\
            tf.assign(mask,tf.image.flip_left_right(mask)),\
            tf.assign(mask_weight,tf.image.flip_left_right(mask_weight)))

        zoom_crop_op =tf.group(tf.assign(image,tf.image.resize_bilinear(tf.image.central_crop(image,self.crop_val),size=[self.image_shape[0], self.image_shape[1]])),\
            tf.assign(mask,tf.image.resize_bilinear(tf.image.central_crop(mask,self.crop_val),size=[self.image_shape[0], self.image_shape[1]])),\
            tf.assign(mask_weight,tf.image.resize_bilinear(tf.image.central_crop(mask_weight,self.crop_val),size=[self.image_shape[0], self.image_shape[1]])))
        '''
        if self.sess is not None:
            #update randoms
            self.sess.run([self.init_randoms])
            print(self.flip_prob.eval())

        with tf.control_dependencies([self.flip_prob.initializer]):
            '''
            image = tf.squeeze(tf.where(tf.greater(self.flip_prob,0.5), tf.expand_dims(tf.image.flip_left_right(image),0), tf.expand_dims(image,0)))
            mask = tf.squeeze(tf.where(tf.greater(self.flip_prob,0.5), tf.expand_dims(tf.image.flip_left_right(mask),0), tf.expand_dims(mask,0)))
            mask_weight = tf.squeeze(tf.where(tf.greater(self.flip_prob,0.5), tf.expand_dims(tf.image.flip_left_right(mask_weight),0), tf.expand_dims(mask_weight,0)))
            '''
            '''
            image = tf.squeeze(tf.where(tf.greater(self.crop_prob,0.5), tf.expand_dims(tf.image.resize_bilinear(tf.image.central_crop(image,self.crop_val),size=[self.image_shape[0], self.image_shape[1]]),0), tf.expand_dims(image,0)))
            mask = tf.squeeze(tf.where(tf.greater(self.crop_prob,0.5), tf.expand_dims(tf.image.resize_bilinear(tf.image.central_crop(mask,self.crop_val),size=[self.image_shape[0], self.image_shape[1]]),0), tf.expand_dims(mask,0)))
            mask_weight = tf.squeeze(tf.where(tf.greater(self.crop_prob,0.5),tf.expand_dims(tf.image.resize_bilinear(tf.image.central_crop(mask_weight,self.crop_val),size=[self.image_shape[0], self.image_shape[1]]),0), tf.expand_dims(mask_weight,0)))
            '''
            #mask_weight = tf.expand_dims(mask_weight,2)
            cshape = tf.expand_dims(tf.squeeze([self.crop_val,self.crop_val,1-self.crop_val,1-self.crop_val]),0)
            #image = tf.squeeze(tf.where(tf.greater(self.flip_prob,-1), tf.image.crop_and_resize(tf.expand_dims(image,0),cshape,[1],[self.image_shape[0], self.image_shape[1]]), tf.expand_dims(image,0)))
            #mask = tf.squeeze(tf.where(tf.greater(self.flip_prob,-1), tf.image.crop_and_resize(tf.expand_dims(mask,0),cshape,[1],[self.image_shape[0], self.image_shape[1]]), tf.expand_dims(mask,0)))
            #mask_weight = tf.squeeze(tf.where(tf.greater(self.flip_prob,-1), tf.image.crop_and_resize(tf.expand_dims(mask_weight,0),cshape,[1],[self.image_shape[0], self.image_shape[1]]), tf.expand_dims(mask_weight,0)))
            #tf.image.crop_and_resize(tf.expand_dims(image,0),[[self.crop_val,self.crop_val,1-self.crop_val,1-self.crop_val]],[1],[self.image_shape[0], self.image_shape[1]])

        #image = tf.image.random_brightness(image, max_delta = 0.01)
        #image = tf.image.random_contrast(image, lower = 0.99, upper = 1.01)
        #image = tf.image.random_hue(image, max_delta = 0.01)
        image =tf.image.per_image_standardization(image)

        return image, mask, mask_weight, name


    def pre_process_image(self,pre_process_op):
        with tf.name_scope('Pre_Processing_op') as scope:
            self.images = pre_process_op(self.images)

    def pre_process_weights(self,pre_process_op):
        with tf.name_scope('Pre_Processing_op') as scope:
            self.mask_weights = pre_process_op(self.mask_weights)

    def pre_process_mask(self,pre_process_op):
        with tf.name_scope('Pre_Processing_op') as scope:
            self.masks = pre_process_op(self.masks)

    def batch_inputs(self):
        image, mask, mask_weight, name = self.single_read()
        images,masks ,mask_weights, names= tf.train.shuffle_batch([image, mask, mask_weight, name], batch_size=self.batch_size, num_threads=8, capacity=100, min_after_dequeue=90)
        return images, masks, mask_weights, names
        #TODO: CONFIGURABLE PARAMS


    def next_batch(self, batch_size=1, sess=None):
        with tf.name_scope('Batch_geter') as scope:
            if sess is None :
                self.sess = tf.get_default_session()
            else:
                self.sess = sess
            images, masks, mask_weights, names= self.sess.run([self.images, self.masks, self.mask_weights, self.names], feed_dict={self.batch_size : batch_size})
            return images, masks, mask_weights, names[0].decode()

