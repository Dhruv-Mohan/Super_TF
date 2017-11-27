from utils.Dataset_reader import Dataset_reader
from Dataset_IO.ImageSeqGen.Dataset_config_ImageSeqGen import Dataset_conifg_ImageSeqGen
import Dataset_IO.ImageSeqGen.Dataset_ImageSeqGen_pb2 as proto
import tensorflow as tf
import os
import random


class Dataset_reader_ImageSeqGen(Dataset_reader, Dataset_conifg_ImageSeqGen):

    def __init__(self, filename=None, epochs=100, vocab=None):
        super().__init__()
        with tf.name_scope('Dataset_ImageSeqGen_Reader') as scope:
            self.batch_size = tf.placeholder(tf.int32, name='Dataset_batch_size')

            self.fwd_dict = {}
            self.rev_dict = {}
            if vocab is None:
                self.vocab = [['0', 0], ['1', 1], ['2', 2], ['3', 3], ['4', 4], ['5', 5], ['6', 6], ['7', 7], ['8', 8], ['9', 9],\
                    ['A', 10], ['B', 11], ['C', 12], ['D', 13], ['E', 14], ['F', 15], ['G', 16], ['H', 17], ['I', 18], ['J', 19], ['K', 20], ['L', 21],\
                    ['M', 22], ['N', 23], ['O', 24], ['P', 25], ['Q', 26], ['R', 27], ['S', 28], ['T', 29], ['U', 30], ['V', 31], ['W', 32], ['X', 33],\
                    ['Y', 34], ['Z', 35], ['$', 36], ['#', 37], ['-', 38]]
            else:
                self.vocab = vocab

            self.const_vocab_dict()

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

            decoded_seq = get_seq(complete_seq.upper())
            decoded_mask = get_seq(complete_mask)

            sequence_lenght = len(complete_seq)
            input_seq = decoded_seq[0:sequence_lenght-1]
            target_seq = decoded_seq[1:sequence_lenght]
            final_mask = decoded_mask[0:sequence_lenght-1]
            return images, input_seq, target_seq, final_mask


    def batch_inputs(self):
        image , input_seq, target_seq, final_mask = self.single_read()
        images , out_input_seq, out_target_seq, out_final_mask = tf.train.shuffle_batch([image , input_seq, target_seq, final_mask], batch_size=self.batch_size, num_threads=8, capacity=5000+128, min_after_dequeue=5000)
        #TODO: seq encoding via vocab dict
        return images, out_input_seq, out_target_seq, out_final_mask

    def const_vocab_dict(self):
        for element in self.vocab:
            self.fwd_dict[element[0]] = element[1]
            self.rev_dict[element[1]] = element[0]

    def get_seq(sequence):
        output_seq = []
        for char in sequence:
            output_seq.append(self.fwd_dict[char])
        return output_seq
