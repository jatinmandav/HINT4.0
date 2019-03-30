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
        inp = Input(shape=self.input_shape, name='input_1')

        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(inp)
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)

        out = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=inp, outputs=out)
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

if __name__ == '__main__':
    model = CNNModel((48, 48, 3), 7)
    model = model.build_model()
    model.summary()
