import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 28 x 28 images of hand written digits 0-9
# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Normalization of images
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# Creation of the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
# fit the model
model.fit(x_train, y_train , epochs=3)
# evaluation of the model
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)

# save the model
model.save('epic_num_reader.model')

new_model = tf.keras.models.load_model('epic_num_reader.model')
# prediction from some data
predictions = model.predict([x_test])
print(predictions)

print(np.argmax(predictions[50]))
plt.imshow(x_test[50])
plt.show()