from utils.Dataset_reader import Dataset_reader
from Dataset_IO.ImageSeqGen.Dataset_config_ImageSeqGen import Dataset_conifg_ImageSeqGen
import Dataset_IO.ImageSeqGen.Dataset_ImageSeqGen_pb2 as proto
import tensorflow as tf
import os
import random


class Dataset_reader_ImageSeqGen(Dataset_reader, Dataset_conifg_ImageSeqGen):

    def __init__(self, filename=None, epochs=100, vocab=40):
        super().__init__()
        with tf.name_scope('Dataset_ImageSeqGen_Reader') as scope:
            self.batch_size = tf.placeholder(tf.int32, name='Dataset_batch_size')
            self.num_vocab = vocab
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
            self.images , self.input_seq, self.output_seq, self.mask = self.batch_inputs()

    def pre_process_image(self,pre_process_op):
        with tf.name_scope('Pre_Processing_op') as scope:
            self.images = pre_process_op(self.images)

        def single_read(self):
            features = tf.parse_single_example(self.serialized_example, features=self._Feature_dict)
            image = tf.image.decode_image(features[self._Image_handle])
            image.set_shape(self.image_shape)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = image - self.mean_image
            #Alright we've got images, now to get seqs and masks

            complete_seq = features[self._Seq_handle]
            complete_mask = features[self._Seq_mask]
            sequence_lenght = len(complete_seq)
            input_seq = complete_seq[0:sequence_lenght-1]
            target_seq = complete_seq[1:sequence_lenght]
            final_mask = complete_mask[0:sequence_lenght-1]
            return images, input_seq, target_seq, final_mask


    def batch_inputs(self):
        image , input_seq, target_seq, final_mask = self.single_read()
        images , out_input_seq, out_target_seq, out_final_mask = tf.train.shuffle_batch([image , input_seq, target_seq, final_mask], batch_size=self.batch_size, num_threads=8, capacity=5000+128, min_after_dequeue=5000)
        #TODO: seq encoding via vocab dict
        return images, out_input_seq, out_target_seq, out_final_mask
