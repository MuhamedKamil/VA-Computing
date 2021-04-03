import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras

BATCHSIZE    = 128
EPOCHS       = 6
LearningRate = 0.001

#--------------------------------------------------------------------------------------------
"""
Loading Dataset
"""
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label

num_validation_samples = 0.1 * ds_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
#--------------------------------------------------------------------------------------------
"""
Training and validation Data pipeline
"""
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

ds_validation = ds_train.take(num_validation_samples)
ds_validation = ds_validation.batch(num_validation_samples)
validation_inputs, validation_targets = next(iter(ds_validation))


ds_train = ds_train.skip(num_validation_samples)
ds_train = ds_train.batch(BATCHSIZE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
training_inputs, training_targets = next(iter(ds_train))
#--------------------------------------------------------------------------------------------
"""
Testing Data pipeline
"""
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(BATCHSIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
#--------------------------------------------------------------------------------------------
"""
Building Model
"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])
#--------------------------------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(LearningRate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],)

history =model.fit(ds_train,epochs=EPOCHS,validation_data=(validation_inputs, validation_targets), verbose = 2)
#--------------------------------------------------------------------------------------------
"""
training/validation loss, training/validation accuracy Graphs
"""

epochs_range = range(EPOCHS)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


