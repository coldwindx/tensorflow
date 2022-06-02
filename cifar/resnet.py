import os
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

class ResNetBlk(tf.keras.Model):
    def __init__(self, filters, strides = 1, residual_path = False):
        super(ResNetBlk, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = tf.keras.layers.Conv2D(filters, (3, 3), strides = strides, padding = 'same', use_bias = False)
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')

        self.c2 = tf.keras.layers.Conv2D(filters, (3, 3), strides = 1, padding = 'same', use_bias = False)
        self.b2 = tf.keras.layers.BatchNormalization()

        if residual_path:
            self.down_c1 = tf.keras.layers.Conv2D(filters, (1, 1), strides = strides, padding = 'same', use_bias = False)
            self.down_b1 =  tf.keras.layers.BatchNormalization()
        self.a2 = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)
        
        return self.a2(y + residual)

class ResNet(tf.keras.Model):
    def __init__(self, blocks, filters = 64):
        super(ResNet, self).__init__()
        self.blocks = blocks
        self.filters = filters

        self.c1 = tf.keras.layers.Conv2D(
            filters, (3, 3), strides = 1, padding = 'same', use_bias = False, kernel_initializer = 'he_normal'
        )
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')

        self.r1 = tf.keras.models.Sequential()
        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if i != 0 and j == 0:
                    block = ResNetBlk(self.filters, strides = 2, residual_path=True)
                else:
                    block = ResNetBlk(self.filters, residual_path=False)
                self.r1.add(block)
            self.filters *= 2
        
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10)
    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.r1(x)
        x = self.p1(x)
        x = self.f1(x)
        return x

model = ResNet([2, 2, 2, 2])

model_save_path = './cifar/checkpoints/resnet.ckpt'
if os.path.exists(model_save_path + '.index'):
    print('------------- Load Model ---------------')
    model.load_weights(model_save_path)
model_save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = model_save_path,
    save_weights_only = True,
    save_best_only = True
)
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['sparse_categorical_accuracy']
)
model.fit(
    x_train, y_train, batch_size = 128,
    epochs = 5,
    validation_data = (x_test, y_test),
    validation_freq = 1,
    callbacks = [model_save_callback]
)


model.summary()