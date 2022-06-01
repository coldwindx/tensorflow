import os
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
########################### 数据增强 #################################
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1 / 1,                # 
    # rotation_range = 45,            # 随机45度旋转
    width_shift_range = 0.15,       # 宽度偏移
    height_shift_range = 0.15,      # 高度偏移
    # horizontal_flip = True,         # 水平翻转
    zoom_range = 0.5                # 随机缩放
)
image_gen_train.fit(x_train)

########################### 构建模型 #################################
class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation = "relu")
        self.d2 = tf.keras.layers.Dense(10, activation = "softmax")
    
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MnistModel()

########################### 断点续训 #################################
check_point_path = "./mnist/checkpoints/mnist.ckpt"
if os.path.exists(check_point_path + ".index"):
    print("--------------- Load Model ------------------")
    model.load_weights(check_point_path)
check_point_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = check_point_path,
    save_weights_only = True,
    save_best_only = True
)

model.compile(
    optimizer = tf.keras.optimizers.SGD(lr = 0.2),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['sparse_categorical_accuracy']    
)
history = model.fit(
    image_gen_train.flow(x_train, y_train, batch_size = 32),
    epochs = 5,
    validation_data = (x_test, y_test),
    validation_freq = 1,
    callbacks = [check_point_callback]
)
model.summary()

########################### 参数提取 #################################
file = open("./mnist/weights.txt", "w")
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

########################### 可视化 #################################
acc = history.history["sparse_categorical_accuracy"]
val_acc = history.history["val_sparse_categorical_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.subplot(1, 2, 1)
plt.title("Training and Validation Accuracy")
plt.plot(acc, label = "Training Accuracy")
plt.plot(val_acc, label = "Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Training and Validation Loss")
plt.plot(loss, label = "Training Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.legend()

plt.savefig("./mnist/training.png")
plt.show()

########################### 模型应用 #################################
# 实际场景下应当利用断点数据重建模型
for i in range(10):
    image_path = "./mnist/src/{}.png".format(i)
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    for i in range(28):
        for j in range(28):
            img_arr[i][j] = 255 if img_arr[i][j] < 200 else 0

    img_arr = img_arr / 255
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis = 1)
    tf.print(pred)