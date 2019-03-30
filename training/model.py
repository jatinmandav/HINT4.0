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

        #x = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x)
        #x = Conv2D(8, kernel_size=3, padding='same', activation='relu', name='conv2d_1')(inp)
        #x = BatchNormalization(name='batch_normalization_1')(x)
        #x = MaxPool2D(pool_size=3, strides=2, name='max_pooling2d_1')(x)

        #x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
        #x = Conv2D(16, kernel_size=3, padding='same', activation='relu', name='conv2d_2')(x)
        x = Conv2D(16, kernel_size=3, padding='same', activation='relu', name='conv2d_3')(inp)
        x = BatchNormalization(name='batch_normalization_2')(x)
        x = MaxPool2D(pool_size=3, strides=2, name='max_pooling2d_2')(x)

        #x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu', name='conv2d_4')(x)
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu', name='conv2d_5')(x)
        x = BatchNormalization(name='batch_normalization_3')(x)
        x = MaxPool2D(pool_size=3, strides=2, name='max_pooling2d_3')(x)

        #x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu', name='conv2d_6')(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu', name='conv2d_7')(x)
        x = BatchNormalization(name='batch_normalization_4')(x)
        x = MaxPool2D(pool_size=3, strides=2, name='max_pooling2d_4')(x)

        #x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='conv2d_8')(x)
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='conv2d_9')(x)
        x = BatchNormalization(name='batch_normalization_5')(x)
        x = MaxPool2D(pool_size=3, strides=2, name='max_pooling2d_5')(x)

        #x = Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
        #x = Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(192, kernel_size=3, padding='same', activation='relu', name='conv2d_10')(x)
        x = Conv2D(192, kernel_size=3, padding='same', activation='relu', name='conv2d_11')(x)
        x = BatchNormalization(name='batch_normalization_6')(x)
        x = Flatten(name='flatten_1')(x)

        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dense(512, activation='relu', name='dense_2')(x)

        out = Dense(self.classes, activation='softmax', name='output_1')(x)

        model = Model(inputs=inp, outputs=out)
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

if __name__ == '__main__':
    model = CNNModel((48, 48, 3), 7)
    model = model.build_model()
    model.summary()
