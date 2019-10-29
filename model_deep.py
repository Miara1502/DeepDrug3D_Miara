# -*- coding: utf-8  -*-

"""Module qui contient tous les fonctions pour la création et l'évaluation du
modèle
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Convolution3D, Flatten, MaxPooling3D, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

def my_model():
    model = Sequential()
            # 1ere Convolution
    model.add(Convolution3D(input_shape = (14,32,32,32), filters=64, kernel_size=5, padding='valid', data_format='channels_first'))
    model.add(LeakyReLU(alpha = 0.1))
    #model.add(MaxPooling3D(pool_size=(2,2,2), padding='valid', data_format='channels_first'))
            # Dropout 1
    model.add(Dropout(0.2))

            # 2ème convolution
    model.add(Convolution3D(filters=64, kernel_size=3, padding='valid', data_format='channels_first',))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_first'))
            # Dropout 2
    model.add(Dropout(0.4))


            # FC 1
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha = 0.1))
            # Dropout 3
    model.add(Dropout(0.4))
    # Fully connected layer 2 to shape (2) for 2 classes
    model.add(Dense(3))
    model.add(Activation('softmax'))


    return model

def training(model , x_train , y_train , x_test , y_test):
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    EarlyStopping(monitor='val_loss',min_delta=0.1,patience=4,verbose=1,mode='auto',
    baseline=None, restore_best_weights=False)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size = 10)
    val_loss , val_acc = model.evaluate(x_test, y_test)
    print(model.metrics_names)
    print(val_loss , val_acc)



def prediction(model, x) :
    pred = model.predict(x)
    return pred

def main():
    features = np.load('features.npy')
    targets = np.load('targets.npy')

    print(features.shape)
    print(targets)
    y = np_utils.to_categorical(targets , num_classes = 3)
    print('hello ici normalement')
    print(y)


    x_train,x_test,y_train,y_test=train_test_split(
    features, y,test_size=0.2,random_state=42)


    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = my_model()
    print(model.input_shape)
    print(model.output_shape)

    model.summary()
    model_training = training(model,x_train , y_train , x_test , y_test)




if __name__ == "__main__":
    main()
