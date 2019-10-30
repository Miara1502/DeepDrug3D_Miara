# -*- coding: utf-8  -*-

"""Module qui contient tous les fonctions pour la création et l'évaluation du
modèle
"""
import argparse
import sys
import os
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from itertools import cycle

from keras.models import Sequential
from keras.layers import Dense, Convolution3D, Flatten, MaxPooling3D, Dropout
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

def my_model():
    '''Creation of the model Sequential and return a model
    for the training
    '''
    model = Sequential()
    # Conv layer 1
    model.add(Convolution3D(
        input_shape = (14,32,32,32),
        filters=64,
        kernel_size=6,
        data_format='channels_first',
    ))

    model.add(LeakyReLU(alpha = 0.1))
    # Dropout 1
    model.add(Dropout(0.2))
    # Conv layer 2
    model.add(Convolution3D(
        filters=64,
        kernel_size=3,
        padding='valid',
        data_format='channels_first',
    ))
    model.add(LeakyReLU(alpha = 0.1))
    # Maxpooling 1
    model.add(MaxPooling3D(
        pool_size=(2,2,2),
        strides=None,
        padding='valid',
        data_format='channels_first'
    ))
    # conv Layer 2
    model.add(Convolution3D(
        filters=64,
        kernel_size=1,
        padding='valid',
        data_format='channels_first',
    ))
    model.add(LeakyReLU(alpha = 0.1))
    # Maxpooling 1
    model.add(MaxPooling3D(
        pool_size=(2,2,2),
        strides=None,
        padding='valid',    # Padding method
        data_format='channels_first'
    ))
    # Dropout 2
    model.add(Dropout(0.4))
    # FC 1
    model.add(Flatten())
    model.add(Dense(128)) #
    model.add(LeakyReLU(alpha = 0.1))
    # Dropout 3
    # Fully connected layer 2 to shape (3) for 3 classes
    model.add(Dense(3))
    model.add(Activation('sigmoid'))
    model.summary()
    return model

def training(model,nb_epoch,batch_size,x_train , y_train , x_test , y_test):
    ''' Training of the model
    '''
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    EarlyStopping(monitor='val_loss',min_delta=0.1,patience=6,verbose=0,mode='auto',
    baseline=None, restore_best_weights=False)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epoch, batch_size = batch_size)
    val_loss , val_acc = model.evaluate(x_test, y_test)
    print(model.metrics_names)
    print(val_loss , val_acc)



def prediction(model, x):
    '''take a list of value and return the prediction
    by using the model
    '''
    pred = model.predict(x)
    return pred

def Courbe_roc(y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    #Compute micro-averafe ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #PLOT for every class
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2  # Largeur de ligne
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('ROC_AUC.png')
    plt.show()


def main():
    '''Main of the programm 
    '''
    parser=argparse.ArgumentParser(description="Deepdrug traitement fichiers.")
    parser.add_argument("-e", "--nb-epoch",
                        help="Définir un nombre d'epoch",
                        type=int, default=200, dest="nEpoch")
    parser.add_argument("-b", "--batch-size",
                        help="Définir un nombre pour le batch size",
                        type=int, default=200, dest="bSize")
    args = parser.parse_args()
    features = np.load('features.npy')
    targets = np.load('targets.npy')
    x_train,x_test,y_train,y_test=train_test_split(
    features, targets,test_size=0.4,random_state=42)
    #Création du model
    model = my_model()
    #training du model
    training(model,args.nEpoch, args.bSize,x_train,y_train,x_test,y_test)
    #Prédiction
    y_pred = prediction(model, x_test)
    #Courbe_roc
    Courbe_roc(y_test, y_pred)

if __name__ == "__main__":
    main()
