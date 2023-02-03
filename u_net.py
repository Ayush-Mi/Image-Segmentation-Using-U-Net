'''
A python class for U-Net Architecture for image segmentation.
It uses a down sampler or 2 encoder blocks of each - 64,128,256 and 512 filters
It uses a bottleneck of 2 1024 filters
It uses upsampler or 2 decoder blocks of each - 512,256,128 and 64 filters
'''

import tensorflow as tf

class u_net:
    def __init__(self,input_shape=(128,128,3),OUTPUT_CHANNELS=3):
        self.input_shape = input_shape
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS

    def conv2d_block(self,input_tensor, n_filters, kernel_size = 3):
        x = input_tensor
        for i in range(2):
            x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                    kernel_initializer = 'he_normal', padding = 'same')(x)
            x = tf.keras.layers.Activation('relu')(x)
        
        return x

    def encoder_block(self,inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
        f = self.conv2d_block(inputs, n_filters=n_filters)
        p = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(f)
        p = tf.keras.layers.Dropout(0.3)(p)
        
        return f, p

    def encoder(self,inputs):
        f1, p1 = self.encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3)
        f2, p2 = self.encoder_block(p1, n_filters=128, pool_size=(2,2), dropout=0.3)
        f3, p3 = self.encoder_block(p2, n_filters=256, pool_size=(2,2), dropout=0.3)
        f4, p4 = self.encoder_block(p3, n_filters=512, pool_size=(2,2), dropout=0.3)

        return p4, (f1, f2, f3, f4)

    def bottleneck(self,inputs):
        bottle_neck = self.conv2d_block(inputs, n_filters=1024)
        return bottle_neck


    def decoder_block(self,inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
        u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides = strides, padding = 'same')(inputs)
        c = tf.keras.layers.concatenate([u, conv_output])
        c = tf.keras.layers.Dropout(dropout)(c)
        c = self.conv2d_block(c, n_filters, kernel_size=3)

        return c

    def decoder(self,inputs, convs, output_channels):
        f1, f2, f3, f4 = convs

        c6 = self.decoder_block(inputs, f4, n_filters=512, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c7 = self.decoder_block(c6, f3, n_filters=256, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c8 = self.decoder_block(c7, f2, n_filters=128, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c9 = self.decoder_block(c8, f1, n_filters=64, kernel_size=(3,3), strides=(2,2), dropout=0.3)

        outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='softmax')(c9)

        return outputs

    def unet(self,):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        encoder_output, convs = self.encoder(inputs)
        bottle_neck = self.bottleneck(encoder_output)
        outputs = self.decoder(bottle_neck, convs, output_channels=self.OUTPUT_CHANNELS)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model
