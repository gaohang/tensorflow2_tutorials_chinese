import tensorflow as tf
from tensorflow.keras import layers
# print(tf.__version__)
# print(tf.keras.__version__)

import numpy as np
train_x = np.random.random((4096, 72))
train_y = np.random.random((4096, 10))
val_x = np.random.random((256, 72))
val_y = np.random.random((256, 10))
ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size=64).repeat(count=2)
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size=64).repeat(count=1)
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()

mdl = tf.keras.Sequential()
mdl.add(layers.Dense(32, activation='relu'))
mdl.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
mdl.add(layers.Dense(10, activation='softmax'))
mdl.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=[tf.keras.metrics.categorical_accuracy])
mdl.fit(ds, epochs=16, validation_freq=1, validation_steps=None, validation_data=val_ds)
mdl.evaluate(test_data, steps=30)
result = mdl.predict(test_x, batch_size=32)
print(result)



tf.keras.layers.BatchNormalization()
tf.keras.optimizers.RMSprop
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model = tf.keras.Sequential()

model = model.add(layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]),filters=32,kernel_size=(3,3), strides=(1,1), padding='valid',activation='relu'))
model.summary()
model.add(layers.MaxPool2D(pool_size=(2,2)))


model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

tf.keras.Input