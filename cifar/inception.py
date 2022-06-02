import os
import tensorflow as tf

class ConvModel(tf.keras.Model):
    def __init__(self, ch, kernelsz = 3, strides = 1, padding = 'same'):
        super(ConvModel, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(ch, kernelsz, strides = strides, padding = padding),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')    
        ])
    def call(self, x):
        return self.model(x)

class InceptionBlk(tf.keras.Model):
    def __init__(self, ch, strides = 1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        
        self.c1 = ConvModel(ch, kernelsz=1, strides=strides)
        self.c2 = tf.keras.models.Sequential([
            ConvModel(ch, kernelsz=1, strides=strides),
            ConvModel(ch, kernelsz=3, strides=1)
        ])

        self.c3 = tf.keras.models.Sequential([
            ConvModel(ch, kernelsz=1, strides=strides),
            ConvModel(ch, kernelsz=5, strides=1)
        ])
        self.c4 = tf.keras.models.Sequential([
            tf.keras.layers.MaxPool2D(3, strides=1, padding = 'same'),
            ConvModel(  ch, kernelsz=1, strides=strides)
        ])
    def call(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = tf.concat([x1, x2, x3, x4], axis = 3)
        return x

class InceptionNet(tf.keras.Model):
    def __init__(self, blocks, classes, filters = 16, **kwargs):
        super(InceptionNet, self).__init__(**kwargs)
        self.out_channels = filters
        self.input_channels = filters
        self.filters = filters
        self.blocks = blocks

        self.c1 = ConvModel(filters)
        self.b1 = tf.keras.models.Sequential()
        for block_id in range(blocks):
            for layer_id in range(2):
                strides = 2 if layer_id == 0 else 1
                self.b1.add(InceptionBlk(filters, strides=strides))
            self.out_channels *= 2
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(classes, activation = 'softmax')
    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.p1(x)
        x = self.f1(x)
        return x


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

model = InceptionNet(blocks=2, classes=10)
################################# 断点续训 ##########################################
model_save_path = './cifar/checkpoints/inception.ckpt'
if os.path.exists(model_save_path + ".index"):
    print("-------------- Load Model ---------------")  
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
model.fit(x_train, y_train, batch_size = 512, 
    epochs = 5,
    validation_data = (x_test, y_test),
    validation_freq = 1,
    callbacks = [model_save_callback]
)
model.summary()
