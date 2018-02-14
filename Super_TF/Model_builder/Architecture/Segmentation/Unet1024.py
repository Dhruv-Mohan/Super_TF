from utils.builder import Builder
import tensorflow as tf
from utils.Base_Archs.Base_Segnet import Base_Segnet

class Unet1024(Base_Segnet):
    """Unet based on Ronneberger et al. https://arxiv.org/pdf/1505.04597.pdf with default input size 1024x1024"""
    def __init__(self, kwargs):
        super().__init__(kwargs)

    def build_net(self):
        with tf.name_scope('Unet1024'):
            with Builder(**self.build_params) as unet_res_builder:

                #Setting control params
                unet_res_builder.control_params(Dropout_control=self.dropout_placeholder_placeholder, State=self.state_placeholder)

                def stack_encoder(input, out_filters):
                    with tf.name_scope('Encoder'):
                        input = unet_res_builder.Relu(input)

                        #conv1a_split1 = unet_res_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[1, 1], filters=out_filters, Activation=False, Batch_norm=True)

                        conv1 = unet_res_builder.Conv2d_layer(input, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        conv2 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)

                        #res_connect = unet_res_builder.Residual_connect([conv1a_split1, conv2b_split1])

                        return conv2
                def stack_decoder(input, encoder_connect, out_filters, output_shape, infilter=None):
                    with tf.name_scope('Decoder'):
                        encoder_connect_shape = encoder_connect.get_shape().as_list()
                        del encoder_connect_shape[0]
                        res_filters = encoder_connect_shape.pop(2)

                        if infilter is not None:
                            res_filters=infilter
                        #upscale_input = unet_res_builder.Upconv_layer(input, stride=[1, 2, 2, 1], filters=res_filters, Batch_norm=True, output_shape=output_shape) #change_filters to match encoder_connect filters
                        upscale_input = unet_res_builder.Conv_Resize_layer(input, stride=[1,2,2,1], filters=res_filters, Batch_norm=True)
                        uconnect = unet_res_builder.Concat([encoder_connect, upscale_input])
                        conv1 = unet_res_builder.Conv2d_layer(uconnect, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        conv2 = unet_res_builder.Conv2d_layer(conv1, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        conv3 = unet_res_builder.Conv2d_layer(conv2, stride=[1, 1, 1, 1], k_size=[3, 3], filters=out_filters, Batch_norm=True)
                        return conv3

                def Center_pool(input, filters=768):
                    ''' Dense dialations '''
                    with tf.name_scope('Dense_Dialated_Center'):
                        Dconv1 = unet_res_builder.DConv_layer(input, filters=filters, Batch_norm=True, D_rate=1, Activation=False)
                        Dense_connect1 = unet_res_builder.Concat([input, Dconv1])

                        Dconv2 = unet_res_builder.DConv_layer(Dense_connect1, filters=filters, Batch_norm=True, D_rate=2, Activation=False)
                        Dense_connect2 = unet_res_builder.Concat([input, Dconv1, Dconv2])

                        Dconv4 = unet_res_builder.DConv_layer(Dense_connect2, filters=filters, Batch_norm=True, D_rate=4, Activation=False)
                        Dense_connect3 = unet_res_builder.Concat([input, Dconv1, Dconv2, Dconv4 ])

                        Dconv8 = unet_res_builder.DConv_layer(Dense_connect3, filters=filters, Batch_norm=True, D_rate=8, Activation=False)
                        Dense_connect4 = unet_res_builder.Concat([input, Dconv1, Dconv2, Dconv4, Dconv8])

                        Dconv16 = unet_res_builder.DConv_layer(Dense_connect4, filters=filters, Batch_norm=True, D_rate=16, Activation=False)
                        Dense_connect5 = unet_res_builder.Concat([input, Dconv1, Dconv2, Dconv4, Dconv8, Dconv16])

                        Dconv32 = unet_res_builder.DConv_layer(Dense_connect5, filters=filters, Batch_norm=True, D_rate=32, Activation=False)
                        Dense_connect6 = unet_res_builder.Concat([input, Dconv1, Dconv2, Dconv4, Dconv8, Dconv16, Dconv32])

                        Scale_output = unet_res_builder.Scale_activations(Dense_connect6,scaling_factor=0.9)

                        return Scale_output
                #Build Encoder
                
                Encoder1 = stack_encoder(self.input_placeholder, 24)
                Pool1 = unet_res_builder.Pool_layer(Encoder1) #512

                Encoder2 = stack_encoder(Pool1, 64)
                Pool2 = unet_res_builder.Pool_layer(Encoder2) #256

                Encoder3 = stack_encoder(Pool2, 128)
                Pool3 = unet_res_builder.Pool_layer(Encoder3) #128

                Encoder4 = stack_encoder(Pool3, 256)
                Pool4 = unet_res_builder.Pool_layer(Encoder4) #64

                Encoder5 = stack_encoder(Pool4, 512)
                Pool5 = unet_res_builder.Pool_layer(Encoder5) #32

                Encoder6 = stack_encoder(Pool5, 768)
                Pool6 = unet_res_builder.Pool_layer(Encoder6) #16

                Encoder7 = stack_encoder(Pool6, 768)
                Pool7 = unet_res_builder.Pool_layer(Encoder7) #8

                #Center
                #Conv_center = unet_res_builder.Conv2d_layer(Pool7, stride=[1, 1, 1, 1], filters=768, Batch_norm=True, padding='SAME')
                Conv_center = Center_pool(Pool7)
                #Pool_center = unet_res_builder.Pool_layer(Conv_center) #8
                #Build Decoder
                Decode1 = stack_decoder(Conv_center, Encoder7, out_filters=768, output_shape=[16, 16])
                Decode2 = stack_decoder(Decode1, Encoder6, out_filters=768, output_shape=[32, 32])
                Decode3 = stack_decoder(Decode2, Encoder5, out_filters=512, output_shape=[64, 64], infilter=768)
                Decode4 = stack_decoder(Decode3, Encoder4, out_filters=256, output_shape=[128, 128], infilter=512)
                Decode5 = stack_decoder(Decode4, Encoder3, out_filters=128, output_shape=[256, 256], infilter=256)
                Decode6 = stack_decoder(Decode5, Encoder2, out_filters=64, output_shape=[512,512],  infilter=128)
                Decode7 = stack_decoder(Decode6, Encoder1, out_filters=24, output_shape=[1024,1024], infilter=64)
                output = unet_res_builder.Conv2d_layer(Decode7, stride=[1, 1, 1, 1], filters=1, Batch_norm=True, k_size=[1, 1], Activation=False) #output
                return output
