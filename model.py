import keras
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential

# Ucitavanje FashionMNIST skupa podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Koristimo samo deo trening skupa (prvi od 10 fold-ova) radi efikasnosti treninga
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)
for train_index, test_index in skf.split(x_train, y_train):
	x_train, y_train = x_train[test_index], y_train[test_index]
	break
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#################################################################################
# U ovoj sekciji implementirati Keras neuralnu mrezu koja postize tacnost barem
# 85% na test skupu. Ne menjati fajl van ove sekcije.

# Data normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape input data from (28, 28) to (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

# Model definition
model = Sequential()

# 1 layer
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
#
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# # 2 layers
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# 3 layers
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(rate=0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(10, activation='softmax'))

# Print model summary
print(model.summary())

# Model compilation
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Model training
model.fit(x_train, y_train, batch_size=64, epochs=100)

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=1)

# Print test accuracy
print('\n', 'Test loss:', score[0])
print('Test accuracy : ' + str(score[1] * 100) + '%')

#################################################################################

# Cuvanje istreniranog modela u fajl
model.save('fashion2.h5')
