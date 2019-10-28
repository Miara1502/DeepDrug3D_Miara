# Lecture des données
import numpy as np
from sklearn.model_selection import train_test_split


features = np.load('features.npy')
targets = np.load('targets.npy')
X_control = np.load('controls.npy') #ajouter y control pour faire des courbes ROC

X_train, X_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.05, random_state=42)
#Split arrays or matrices into random train and test subsets

# Création du model
from keras.models import Sequential
from keras.layers import Dense, Convolution3D, Flatten, MaxPooling3D, Dropout , Activation
from keras.layers.advanced_activations import LeakyReLU
model = Sequential()

        # 1ère convolution

model.add(Convolution3D(input_shape = (14,32,32,32), filters=64, kernel_size=5, padding='valid', data_format='channels_first'))
model.add(LeakyReLU(alpha = 0.1))
        # Dropout 1
model.add(Dropout(0.2))

        # 2ème convolution

model.add(Convolution3D(filters=32, kernel_size=3, padding='valid', data_format='channels_first',))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_first'))
        # Dropout 2
model.add(Dropout(0.4))

        # FC 1
model.add(Flatten())
model.add(Dense(128)) # TODO changed to 64 for the CAM
model.add(LeakyReLU(alpha = 0.1))

        # Dropout 3
model.add(Dropout(0.4))
# Fully connected layer 2 to shape (2) for 2 classes
model.add(Dense(2))
model.add(Activation('softmax'))

## Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## Training
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=3)

model.summary()
