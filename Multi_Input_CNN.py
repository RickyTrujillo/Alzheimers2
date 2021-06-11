import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, concatenate,AveragePooling2D, Input, Flatten, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from keras.optimizers import SGD

k_init = keras.initializers.glorot_uniform()
b_init = keras.initializers.constant(value=0.2)

input_shape1 = Input(shape=(256, 256, 1))
input_shape2 = Input(shape=(256, 256, 1))
input_shape3 = Input(shape=(256, 256, 1))

def inception_module(input_x, filter_1x1, filter_3x3_reduce, filter_3x3, filter_5x5_reduce, filter_5x5, filter_pool,
                     name=None):
    path1 = Conv2D(filter_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(input_x)

    path2 = Conv2D(filter_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(input_x)
    path2 = Conv2D(filter_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(path2)

    path3 = Conv2D(filter_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(input_x)
    path3 = Conv2D(filter_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(path3)

    path4 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(input_x)
    path4 = Conv2D(filter_pool, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(path4)

    output = concatenate([path1, path2, path3, path4], axis=3, name=None)
    return output

def model_architecture(test_images, test_labels, valid_images, valid_labels, train_images, train_labels):
    #first image
    x1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer=k_init,
               bias_initializer=b_init, name='conv_1_7x7/2')(input_shape1)
    x1 = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool1_3x3/2')(x1)

    #second image
    x2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer=k_init,
                bias_initializer=b_init, name='conv_2_7x7/2')(input_shape2)
    x2 = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool2_3x3/2')(x2)

    #third image
    x3 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer=k_init,
                bias_initializer=b_init, name='conv_3_7x7/2')(input_shape3)
    x3 = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool3_3x3/2')(x3)

    x = Concatenate(axis=-1)([x1,x2,x3])

    x = inception_module(x, filter_1x1=64, filter_3x3_reduce=96, filter_3x3=126, filter_5x5_reduce=16, filter_5x5=32,
                         filter_pool=32, name='inception_4')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool4_3x3/2')(x)

    x = inception_module(x, filter_1x1=192, filter_3x3_reduce=96, filter_3x3=208, filter_5x5_reduce=16, filter_5x5=48,
                         filter_pool=64,
                         name='inception_5')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool5_3x3/2')(x)

    x = inception_module(x, filter_1x1=192, filter_3x3_reduce=96, filter_3x3=208, filter_5x5_reduce=16, filter_5x5=48,
                         filter_pool=64,
                         name='inception_5')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool6_3x3/2')(x)

    x = inception_module(x, filter_1x1=192, filter_3x3_reduce=96, filter_3x3=208, filter_5x5_reduce=16, filter_5x5=48,
                         filter_pool=64,
                         name='inception_6')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool7_3x3/2')(x)

    x = inception_module(x, filter_1x1=192, filter_3x3_reduce=96, filter_3x3=208, filter_5x5_reduce=16, filter_5x5=48,
                         filter_pool=64,
                         name='inception_7')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool8_3x3/2')(x)

    x = inception_module(x, filter_1x1=192, filter_3x3_reduce=96, filter_3x3=208, filter_5x5_reduce=16, filter_5x5=48,
                         filter_pool=64,
                         name='inception_8')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool9_3x3/2')(x)

    x = inception_module(x, filter_1x1=192, filter_3x3_reduce=96, filter_3x3=208, filter_5x5_reduce=16, filter_5x5=48,
                         filter_pool=64,
                         name='inception_9')

    x = GlobalAveragePooling2D(name='gblavgpool5_3x3/1')(x)
    x = Dropout(0.4)(x)
    x = Dense(4, activation='softmax', name='output')(x)
    model = Model([input_shape1, input_shape2, input_shape3], x)
    model.summary()

    sgd = SGD(learning_rate=1e-4, momentum=0.4, decay=0.01, nesterov=False)
    model.compile(loss='categorical_crossentropy', loss_weights=0.3, optimizer=sgd, metrics=['accuracy'])

    #How can we incorporate the idea such that training images are fed in but still need to be unpacked?
    history = model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=10,batch_size=32)
