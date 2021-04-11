import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.keras.backend as K


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.preprocessing import label_binarize

BATCHSIZE    = 128
EPOCHS       = 10
LearningRate = 0.001
img_height   = 28
img_width    = 28

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
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

#--------------------------------------------------------------------------------------------
"""
Training and validation Data pipeline
"""
num_validation_samples = 0.2 * ds_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

ds_validation = ds_train.take(num_validation_samples)
ds_validation = ds_validation.batch(num_validation_samples)
validation_inputs, validation_targets = next(iter(ds_validation))

ds_train = ds_train.skip(num_validation_samples)
ds_train = ds_train.cache()
ds_train = ds_train.batch(BATCHSIZE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
#--------------------------------------------------------------------------------------------
"""
Testing Data pipeline
"""
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(10000)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
#--------------------------------------------------------------------------------------------
"""
Building Model
"""
data_augmentation = keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,img_width,1)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
  ]
)
model = tf.keras.models.Sequential([
  data_augmentation,
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])
#--------------------------------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(LearningRate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #loss=tfa.losses.ContrastiveLoss(0.5),
    #loss=contrastive_loss,
    metrics=['accuracy'],
    #metrics=['binary_accuracy'],
    )


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
#--------------------------------------------------------------------------------------------
"""
Precision , Recall , F1 score
"""
image_batch, label_batch = ds_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
predictions_Graph = predictions
predictions = np.argmax(predictions, axis=1)
precision = precision_score(label_batch, predictions,average='macro')
print('Precision: %f' % precision)
recall = recall_score(label_batch, predictions,average='macro')
print('Recall: %f' % recall)
f1 = f1_score(label_batch, predictions,average='macro')
print('F1 score: %f' % f1)
#--------------------------------------------------------------------------------------------
"""
precision/recall Graph
"""
label_batch_binary = label_binarize(label_batch, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # one hot encode train data

precision = dict()
recall = dict()
for i in range(10):
    precision[i], recall[i], _ = precision_recall_curve(label_batch_binary[:, i],predictions_Graph[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()
#--------------------------------------------------------------------------------------------
