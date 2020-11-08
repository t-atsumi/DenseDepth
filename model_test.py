import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, LSTM, ConvLSTM2D, BatchNormalization, MaxPooling2D
from layers import BilinearUpSampling2D
from loss import depth_loss_function

def create_model(existing='', is_twohundred=False, is_halffeatures=True):
    if len(existing) == 0:
        inputs = Input(shape=(None, 10, 320, 320, 3))
        conv1 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same' , activation="relu")(inputs)
        conv_lstm1 = ConvLSTM2D(16, kernel_size=3, strides=1, padding='same')(conv1)
        outputs = conv_lstm1

        model = Model(inputs=inputs, outputs=conv3, name="test_model")
        model.summary()
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    return model
