import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_circles
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.preprocessing import label_binarize

EPOCHS       = 3
LearningRate = 0.001
BATCHSIZE    = 128
#--------------------------------------------------------------------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
#--------------------------------------------------------------------------------------------
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
#--------------------------------------------------------------------------------------------
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=LearningRate, decay=1e-6),
              metrics=['accuracy'],)

history = model.fit(x_train,y_train,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test))

#--------------------------------------------------------------------------------------------
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
predictions = model.predict_on_batch(x_test)
predictions_Graph = predictions
predictions = np.argmax(predictions, axis=1)
precision = precision_score(y_test, predictions,average='macro')
print('Precision: %f' % precision)
recall = recall_score(y_test, predictions,average='macro')
print('Recall: %f' % recall)
f1 = f1_score(y_test, predictions,average='macro')
print('F1 score: %f' % f1)
#--------------------------------------------------------------------------------------------
"""
precision/recall Graph
"""
label_batch_binary = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # one hot encode train data

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
