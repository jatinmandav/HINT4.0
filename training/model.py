from keras.models import Model
from keras.layers import Input, Dense, Dropout, MaxPool2D, Conv2D, Flatten
from keras.layers.normalization import BatchNormalization

import keras.backend as K

class Model:
    def __init__(self, input_shape=None, classes=None):
        assert input_shape != None or classes != None, 'No Input shape/Classes provided'
        self.model = None
        self.input_shape = input_shape
        self.classes = classes

    def build_model(self):
        # Input
        inp = Input(shape=input_shape)

        x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(inp)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)

        x = Conv2D(16, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)

        x = Conv2D(32, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(64, kernel_size=7, padding='same', activation='relu')(x)
        #x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        #x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(128, kernel_size=11, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        #x = MaxPool2D(pool_size=3, strides=2)(x)

        #x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        #x = Conv2D(128, kernel_size=11, padding='same', activation='relu')(x)

        x = Flatten()(x)

        #x = Dense(512, activation='relu')(x)
        #x = Dropout(0.1)(x)
        x = Dense(512, activation='relu')(x)

        out = Dense(output_shape, activation='softmax')(x)
        self.model = Model(inputs=inp, outputs=out)

        return self.model
