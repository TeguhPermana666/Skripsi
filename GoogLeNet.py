from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,AveragePooling2D,Flatten,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization
from tensorflow.keras.layers import concatenate
from keras.layers import  Lambda
import tensorflow as tf
def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
  # Input: 
  # - f1: number of filters of the 1x1 convolutional layer in the first path
  # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
  # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
  # - f4: number of filters of the 1x1 convolutional layer in the fourth path

  # 1st path:
  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  # 2nd path
  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  # 3rd path
  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  # 4th path
  path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer

def GoogLeNet():
    input_layer = Input(shape=(28, 28, 1))
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = BatchNormalization()(X)
    X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(X)
    X = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Inception_block(X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Inception_block(X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)
    X = GlobalAveragePooling2D(name='GAPL')(X)
    X = Dropout(0.4)(X)
    X = Dense(11, activation='softmax', name='output')(X)
    model = Model(input_layer, X, name='GoogLeNetSingleOutput')
    return model