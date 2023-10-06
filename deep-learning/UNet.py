import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

# UNet model from 256x256x3 sized images
def UNet256():
    inputs = tf.keras.Input(shape=(256, 256, 3))

    # Encoder
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)    
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    # Decoder
    up5 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv3])
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv2])
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv1])
    conv7 = Conv2D(64, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3,3), activation='relu', padding='same')(conv7)

    # Output
    outputs = Conv2D(3, (1,1), activation='sigmoid')(conv7)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
     