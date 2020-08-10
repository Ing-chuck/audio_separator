import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU

def create_model(input_shape):
    model = Sequential()
    
    # first combolution unuit
    model.add(Conv2D(16, (3,3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # second combolution unuit
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # fully connected unit
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(513, activation='sigmoid'))
    
    sgd = optimizers.SGD(lr=0.0x1, decay=1e-6, momentum=0.9, nesterov=True)
    #loss = losses.MeanSquaredError()
    loss = losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    return model
