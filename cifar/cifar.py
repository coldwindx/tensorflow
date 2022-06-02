import imp
import os
import matplotlib
import tensorflow as tf

matplotlib.use('Agg')
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255 , x_test / 255

class Baseline(tf.keras.Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(6, (5, 5), padding = 'same')
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')
        self.p1 = tf.keras.layers.MaxPool2D((2, 2), strides = 2, padding = 'same')
        self.d1 = tf.keras.layers.Dropout(0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.l1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.l2 = tf.keras.layers.Dense(10, activation = 'softmax')
    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.d2(x)
        x = self.l2(x)
        return x

model = Baseline()
############################ 断点续训 ##################################
model_save_path = "./cifar/checkpoints/cifar.ckpt"
if os.path.exists(model_save_path + ".index"):
    print("------------ Load Model ------------")
    model.load_weights(model_save_path)
save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = model_save_path,
    save_weights_only = True,
    save_best_only = True
)

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['sparse_categorical_accuracy']
)
history = model.fit(
    x_train, y_train, batch_size = 32,
    epochs = 5,
    validation_data = (x_test, y_test),
    validation_freq = 1,
    callbacks = [save_callback]
)
model.summary()

############################ 可视化 ##################################
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.title('Training and Validation Accuracy')
plt.plot(acc, label = 'Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Training and Validation Loss')
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.legend()

plt.savefig('./cifar/training.png')
plt.show()