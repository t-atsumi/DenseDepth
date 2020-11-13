import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, LSTM, ConvLSTM2D, BatchNormalization, MaxPooling2D, Conv3D, TimeDistributed
from layers import BilinearUpSampling2D
from loss import depth_loss_function

def create_model(existing='', is_twohundred=False, is_halffeatures=True, n_sumples=10, image_size=320):
    if len(existing) == 0:
        # Encoder Layers
        input_shape = (n_sumples, image_size, image_size, 3)
        inputs = Input(shape=input_shape)
        
        # Define downsampling layer
        def downproject(tensor, filters, name):
            dp_i = TimeDistributed(Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', activation="relu"))(tensor)
            if filters != 512:
                dp_i = ConvLSTM2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', return_sequences=True, go_backwards=True)(dp_i)
                dp_i = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))(dp_i)
            else:
                dp_i = ConvLSTM2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', return_sequences=False, go_backwards=True)(dp_i)
                dp_i = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(dp_i)
            return dp_i

        encoder = downproject(inputs, 16, 'dp1')
        encoder = downproject(encoder, 32, 'dp2')
        encoder = downproject(encoder, 64, 'dp3')
        encoder = downproject(encoder, 128, 'dp4')
        encoder = downproject(encoder, 256, 'dp5')
        encoder = downproject(encoder, 512, 'dp6')

        # Decoder Layers
        decode_filters = int(encoder.shape[-1])
        
        # Define upsampling layer
        def upproject(tensor, filters, name):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=encoder.shape, name='conv2')(encoder)
        decoder = upproject(decoder, int(decode_filters/2), 'up1')
        decoder = upproject(decoder, int(decode_filters/4), 'up2')
        decoder = upproject(decoder, int(decode_filters/8), 'up3')
        decoder = upproject(decoder, int(decode_filters/16), 'up4')
        decoder = upproject(decoder, int(decode_filters/32), 'up5')

        # Extract depths(final Layer)
        conv3 = Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding='same', name='conv3')(decoder)

        # Create the Model
        model = Model(inputs=inputs, outputs=conv3)

        model.summary()
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    return model

model = create_model()