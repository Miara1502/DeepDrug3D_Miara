# -*- coding: utf-8  -*-

"""Module qui contient tous les fonctions pour la création et l'évaluation du
modèle
"""

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
    model = Sequential()

    model.add(Convolution3D(input_shape = (14,32,32,32), filters=64, kernel_size=5, padding='valid', data_format='channels_first'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling3D(pool_size=(2,2,2), padding='valid', data_format='channels_first'))
    # 2ème convolution
    model.add(Convolution3D(filters=64, kernel_size=3, padding='valid', data_format='channels_first',))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_first'))
    model.add(Dropout(0.4))
    # 3ème convolution
    model.add(Convolution3D(filters=128, kernel_size=2, padding='valid', data_format='channels_first',))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_first'))
    model.add(Dropout(0.4))
    # Fully connected layer pour 3 classes
    model.add(Flatten())
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    return model

def training(model , x_train , y_train , x_test , y_test):
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    EarlyStopping(monitor='val_loss',min_delta=0.1,patience=6,verbose=0,mode='auto',
    baseline=None, restore_best_weights=False)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    val_loss , val_acc = model.evaluate(x_test, y_test)
    print(model.metrics_names)
    print(val_loss , val_acc)



def prediction(model, x) :
    pred = model.predict(x)
    return pred

def Courbe_roc(y_test, y_pred):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        print('valeur prises par la courbe')
        print('\n')
        print(y_test[:, i])
        print( y_pred[:, i])

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
    features = np.load('features.npy')
    targets = np.load('targets.npy')

    print(features.shape)
    print(targets.shape)

    #y = np_utils.to_categorical(targets , num_classes = 3)
    x_train,x_test,y_train,y_test=train_test_split(
    features, targets,test_size=0.4,random_state=42)

    #Création du model
    model = my_model()
    print(model.input_shape)
    print(model.output_shape)
    #model.summary()

    #training du model
    training(model,x_train , y_train , x_test , y_test)
    #Prédiction
    y_pred = prediction(model, x_test)

    '''
    print('ce qui a été prédit :')
    print(y_pred)
    print('alors que X)')
    print(y_test)
    '''
    #print(accuracy_score(y_test, y_pred))
    Courbe_roc(y_test, y_pred)

if __name__ == "__main__":
    main()
