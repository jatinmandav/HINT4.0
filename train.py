from training.model import CNNModel
from dataset.preprocess import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

train_faces, train_emotions = read_data('dataset/fer2013/fer2013.csv', 'Training')
test_faces, test_emotions = read_data('dataset/fer2013/fer2013.csv', 'PublicTest')
val_faces, val_emotions = read_data('dataset/fer2013/fer2013.csv', 'PrivateTest')

print('Training Size: {}, {}'.format(train_faces.shape, train_emotions.shape))
print('Testing Size: {}, {}'.format(test_faces.shape, test_emotions.shape))
print('Validation Size: {}, {}'.format(val_faces.shape, val_emotions.shape))

#model = CNNModel(input_shape=(image_size[0], image_size[1], 3), classes=len(EMOTIONS))
#model = model.build_model()

#inp = Input(shape=(48, 48, 1))
#base_model = VGG16(input_shape=(48, 48, 3), weights='imagenet', include_top=False)
#base_model = DenseNet121(input_shape=(48, 48, 3), weights='imagenet', include_top=False)
base_model = VGG19(input_shape=(48, 48, 3), weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='softmax')(x)
x = Dense(len(EMOTIONS), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

for layer in base_model.layers:
    layer.trainable = False

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

log_dir = 'logs'

logging = TrainValTensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(5), verbose=1)

earlystopper = EarlyStopping('val_loss', patience=8)

model.fit(train_faces, train_emotions, batch_size=150, epochs=25, verbose=1,
          callbacks=[checkpoint, logging], validation_data=(val_faces, val_emotions))


for layer in base_model.layers:
    layer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_faces, train_emotions, batch_size=100, epochs=100, verbose=1,
          callbacks=[checkpoint, logging, reduce_lr, earlystopper], validation_data=(val_faces, val_emotions))


print('Accuracy on Training Data: {}'.format(model.evaluate(train_faces, train_emotions, batch_size=100)))
print('Accuracy on Validation Data: {}'.format(model.evaluate(val_faces, val_emotions, batch_size=100)))
print('Accuracy on Testing Data: {}'.format(model.evaluate(test_faces, test_emotions, batch_size=100)))

import time
start = time.time()
model.predict(np.reshape(test_faces[0], (1, image_size[0], image_size[1], 3)))
print('Inference Time: {}'.format(time.time() - start))
