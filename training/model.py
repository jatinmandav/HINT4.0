from keras.models import Model
from keras.layers import Input, Dense, Dropout, MaxPool2D, Conv2D, Flatten, Multiply, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from keras.applications.vgg16 import VGG16

class CNNModel:
    def __init__(self, input_shape=None, classes=None):
        assert input_shape != None or classes != None, 'No Input shape/Classes provided'
        self.model = None
        self.input_shape = input_shape
        self.classes = classes

    def build_model(self):
        # Input
        inp = Input(shape=self.input_shape)

        x1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(inp)
        x2 = Conv2D(32, kernel_size=5, padding='same', activation='relu')(x1)

        x = Multiply()([x1, x2])

        x3 = Conv2D(32, kernel_size=5, padding='same', activation='relu')(x)
        x4 = Conv2D(32, kernel_size=5, padding='same', activation='relu')(x3)

        x = Multiply()([x, x4])

        x5 = Conv2D(32, kernel_size=7, padding='same', activation='relu')(x)
        x6 = Conv2D(32, kernel_size=7, padding='same', activation='relu')(x5)
        x7 = Conv2D(32, kernel_size=7, padding='same', activation='relu')(x6)

        x = Multiply()([x, x7])

        x8 = Conv2D(32, kernel_size=11, padding='same', activation='relu')(x)

        x = Multiply()([x1, x8])

        x = GlobalAveragePooling2D()(x)

        #x = MaxPool2D(pool_size=3, strides=2)(x)

        #x = Conv2D(128, kernel_size=11, padding='same', activation='relu')(x)

        #x = Flatten()(x)

        #x = Dense(512, activation='relu')(x)
        #x = Dropout(0.1)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)

        out = Dense(self.classes, activation='softmax')(x)
        self.model = Model(inputs=inp, outputs=out)

        return self.model

if __name__ == '__main__':
    model = CNNModel((64, 64, 3), 7)
    model = model.build_model()
    model.summary()
