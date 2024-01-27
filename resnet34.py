from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import tensorflow as tf
def identity_block(X,filter):
    # copy tensor to variable called x_skip
    x_skip = X
    # layer 1 
    X = tf.keras.layers.Conv2D(filter,(3,3),padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Layer 2
    X = tf.keras.layers.Conv2D(filter,(3,3),padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    
    # Add Resiude
    X = tf.keras.layers.Add()([X,x_skip])
    X = tf.keras.layers.Activation('relu')(X)
    return X

# Convolutional block -> add the bootleck to the layer to solve the unbalance shape from skip layer
def Convolutional_block(X,filter):
    X_skip = X
    # layer 1 
    X = tf.keras.layers.Conv2D(filter,(3,3),padding='same',strides=(2,2))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Layer 2
    X = tf.keras.layers.Conv2D(filter,(3,3),padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    
    # processing residue with conv(1,1)
    X_skip = tf.keras.layers.Conv2D(filter,(1,1),strides=(2,2),padding='same')(X_skip)
    
    # add residue
    X = tf.keras.layers.Add()([X,X_skip])
    X = tf.keras.layers.Activation('relu')(X)
    return X

def ResNet34(shape=(28,28,1),classes=11):
    # step 1 -> setup(input layer)
    X_input = tf.keras.layers.Input(shape)
    X = tf.keras.layers.ZeroPadding2D((3,3))(X_input)
    
    # step 2 -> (initial conv layer along with maxpool)
    X = tf.keras.layers.Conv2D(64,kernel_size=(7,7),strides=2,padding='same')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')(X)
    
    # step 3 ->define the size of sub blocks and initila filter size
    block_layers=[3,4,6,3]
    filter_size = 64
    # step 4 add the resnet block
    for i in range(4):
        if i==0:
            # for sub-block 1 residual/convolutional block not needed increment value
            for j in range(block_layers[i]):
                X = identity_block(X,filter_size)
        else:
            # for sub-block 1++ residula/convolutional block need increment value
            filter_size = filter_size*2
            X = Convolutional_block(X,filter_size)
            for j in range(block_layers[i]-1):
                X = identity_block(X,filter_size)
    
    # step 4 end dense network with fcp (full conencted layer)
    X = tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same')(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(512,activation='relu')(X)
    X = tf.keras.layers.Dense(classes,activation='softmax')(X)
    model = tf.keras.models.Model(inputs=X_input,outputs=X,name='ResNet34')
    return model