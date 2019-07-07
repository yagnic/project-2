#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
(train_images,train_labels),(test_images,test_labels)=keras.datasets.fashion_mnist.load_data()
train_images=train_images/255.0
test_images=test_images/255.0

model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(32,activation=tf.nn.relu),keras.layers.Dense(64,activation=tf.nn.relu),keras.layers.Dense(32,activation=tf.nn.softmax)])
model.compile(optimizer=keras.optimizers.Adadelta(),loss='sparse_categorical_crossentropy',metrics=["accuracy"])

model_log=model.fit(train_images,train_labels,epochs=5,validation_data=(test_images,test_labels))
model_log.history
fig=plt.figure(figsize=(8,8))
fig.add_subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train','test'],loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(['train','test'], loc='lower right')
plt.tight_layout()

