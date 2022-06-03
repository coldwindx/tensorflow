import os
import numpy as np
import tensorflow as tf

input_words = 'adcdefghijklmnopqrstuvwxyz'
w_to_id = {
    'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6,
    'h' : 7, 'i' : 8, 'g' : 9, 'k' : 10, 'l' : 11, 'm' : 12, 'n' : 13,
    'o' : 14, 'p' : 15, 'q' : 16,
    'r' : 17, 's' : 18, 't' : 19,
    'u' : 20, 'v' : 21, 'w' : 22, 'x' : 23, 'y' : 24, 'z' : 25
}
training_set_scaled = [ x for x in range(26) ]


x_train = []
y_train = []

for i in range(4, 26):
    x_train.append(training_set_scaled[i - 4 : i])
    y_train.append(training_set_scaled[i])


np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(26, 2),
    tf.keras.layers.SimpleRNN(10),
    tf.keras.layers.Dense(26, activation = 'softmax')
])
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['sparse_categorical_accuracy']
)
model_save_path = './word/checkpoints/word.ckpt'
if os.path.exists(model_save_path + '.index'):
    model.load_weights(model_save_path)
model_save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = model_save_path,
    save_weights_only = True,
    save_best_only = True,
    monitor = 'loss'
)
model.fit(
    x_train, y_train,
    batch_size = 32, 
    epochs = 100,
    callbacks = [model_save_callback]
)
model.summary()

########################## predict #############################
preNum = int(input('input the number of test alphabet:'))
for i in range(preNum):
    alphabet1 = input('input test alphabet:')
    alphabet = [w_to_id[a] for a in alphabet1]
    alphabet = np.reshape(alphabet, (1, 4))
    result = model.predict([alphabet])
    pred = np.argmax(result, axis = 1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_words[pred] + '\n')