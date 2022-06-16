from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D
from keras.models import Model
from keras.models import Sequential
from tensorflow.keras.layers import Input,concatenate,add,Activation,Conv2DTranspose,Conv2D, Convolution2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D,UpSampling2D, Dropout, BatchNormalization, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
'''def actual_unet(img_rows, img_cols, N = 2):
    """This model is based on:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    """

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(
        2**(N + 3), (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(
        2**(N + 3), (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(
        2**(N + 4), (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(
        2**(N + 4), (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate(
        [
            Conv2DTranspose(
                2**(N + 3),
                (2, 2), strides=(2, 2), padding='same')(conv5), conv4
        ],
        axis=3)
    conv6 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(
        2**(N + 3), (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate(
        [
            Conv2DTranspose(
                2**(N + 2),
                (2, 2), strides=(2, 2), padding='same')(conv6), conv3
        ],
        axis=3)
    conv7 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate(
        [
            Conv2DTranspose(
                2**(N + 1),
                (2, 2), strides=(2, 2), padding='same')(conv7), conv2
        ],
        axis=3)
    conv8 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate(
        [
            Conv2DTranspose(2**N, (2, 2), strides=(2, 2),
                            padding='same')(conv8), conv1
        ],
        axis=3)
    conv9 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model'''
def resblock(inputs, num, depth, scale=0.1): 
    residual = Conv2D(depth, (num, num), padding='same')(inputs)
    residual = BatchNormalization(axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res)
  
def actual_unet(img_rows, img_cols, N = 2):
    """This model is based on:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    """

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same', dilation_rate=7)(inputs)
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same', dilation_rate=7)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    

    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same', dilation_rate=7)(pool1)
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same', dilation_rate=7)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    

    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=3)(pool2)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=3)(pool3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=3)(pool4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=5)(conv5)

    up6 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same', dilation_rate=3)(conv5), conv4],axis=3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=3)(up6)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=5)(conv6)

    up7 = concatenate([Conv2DTranspose(16,(2, 2), strides=(2, 2), padding='same', dilation_rate=3)(conv6), conv3],axis=3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=3)(up7)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=5)(conv7)
    
    up8 = concatenate([Conv2DTranspose(8,(2, 2), strides=(2, 2), padding='same', dilation_rate=3)(conv7), conv2],axis=3)
    conv8 = Conv2D(8, (3, 3), activation='relu', padding='same',  dilation_rate=3)(up8)
    conv8 = Conv2D(8, (3, 3), activation='relu', padding='same', dilation_rate=5)(conv8)

    up9 = concatenate([Conv2DTranspose(4, (2, 2), strides=(2, 2),padding='same', dilation_rate=3)(conv8), conv1],axis=3)
    conv9 = Conv2D(4, (3, 3), activation='relu', padding='same', dilation_rate=3)(up9)
    conv9 = Conv2D(4, (3, 3), activation='relu', padding='same', dilation_rate=5)(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def simple_unet( img_rows, img_cols, N = 3):

    print(img_rows, img_cols)

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(pool1)
    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(pool2)
    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(conv3)

    up1 = concatenate(
        [
            Conv2D(2**(N+1), 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv3)),
            conv2
        ],

        axis=3)
    conv4 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(up1)
    conv4 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(conv4)


    up2 = concatenate(
        [
         Conv2D(2**(N), 2, activation = 'relu', padding = 'same', dilation_rate=(2,2))(UpSampling2D(size = (2,2))(conv4)),
         conv1
        ],
        axis=3)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(up2)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same', dilation_rate=(2,2))(conv5)

    conv6 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[conv6])

    return model

def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(residual._keras_shape[1], (1,1),strides=(stride_width, stride_height),padding="valid")(_input)
                                 
                                  

    return add([shortcut, residual])


    
def poolConvolution2D(nb_filter, filters, strides=(1, 1), name=None):
    def f(_input):
        conv = Conv2D(nb_filter, filters, strides=strides, padding='same', name=name)(_input)
                              
        norm = BatchNormalization(axis=1)(conv)
        return ELU()(norm)
    return f 


    
def inception_block(inputs, depth, splitted=True, activation='relu', name=None):
    actv = LeakyReLU
    
    c1_1 = Conv2D(int(depth/4), (1, 1), padding='same', name=name)(inputs)
    c2_1 = Conv2D(int(depth/8*3), (1, 1),padding='same')(inputs)
    c2_1 = actv()(c2_1)
    
    if splitted:
        c2_2 = Conv2D(int(depth/2), (1, 3), padding='same')(c2_1)
        c2_2 = BatchNormalization(axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(int(depth/2), (3, 1),padding='same')(c2_2)
    else:
        c2_3 = Conv2D(int(depth/2), (3, 3),  padding='same')(c2_1)
    
    c3_1 = Conv2D(int(depth/16), (1, 1), padding='same', activation='relu')(inputs)
    c3_1 = actv()(c3_1)
    
    if splitted:
        c3_2 = Conv2D(int(depth/8), (1, 5), padding='same')(c3_1)
        c3_2 = BatchNormalization(axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(int(depth/8), (5, 1), padding='same')(c3_2)
    else:
        c3_3 = Conv2D(int(depth/8), (3, 3), padding='same')(c3_1)
    
    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1),padding='same')(inputs)
    c4_2 = Conv2D(int(depth/8), (1, 1), padding='same')(p4_1)
    
    res = concatenate([c1_1, c2_3, c3_3, c4_2],axis=3)
    res = BatchNormalization(axis=1)(res)
    res=actv()(res)
    return res     

 
'''    
def PoolConvolution2D(nb_filter):
    def f(_input):
        conv = Conv2D(nb_filter,(2,2), padding='same', strides=(2,2))(_input)
        norm = BatchNormalization(axis=1)(conv)
        return ELU()(norm)
    return f  
'''    

    
def resblock(inputs, num, depth, scale=0.1): 
    residual = Conv2D(depth, (num, num), padding='same')(inputs)
    residual = BatchNormalization(axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res)  
    
#keras.layers.convolutional.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid')
#keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation=None)



def N2(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 1))
    splitted=True
    act='relu'
    strides=(2,2)
    padding='valid'
    
    conv1 = (inception_block(inputs, 32, splitted= splitted, activation= act, name = 'INCEP_1'))
    pool1 = poolConvolution2D(32, filters=(3, 3), strides=(2,2), name = 'POOL_1')(conv1)

    conv2 = inception_block(pool1, 64, splitted= splitted, activation=act, name = 'INCEP_2')
    pool2 = poolConvolution2D(64, filters=(3, 3), strides=(2,2), name = 'POOL_2')(conv2)


    conv3 = inception_block(pool2, 128, splitted=splitted, activation=act, name = 'INCEP_3')
    pool3 = poolConvolution2D(128, filters=(3, 3), strides=(2,2), name = 'POOL_3')(conv3)


    conv4 = inception_block(pool3, 256, splitted=splitted, activation=act, name = 'INCEP_4')
    pool4 = poolConvolution2D(256, filters=(3, 3), strides=(2,2), name = 'POOL_4')(conv4)
    #pool4=Dropout(pool4)

    conv5 = inception_block(pool4, 512, splitted=splitted, activation=act, name = 'INCEP_5')
    
    after_conv4 = resblock(conv4, 1, 256)   
    
    decon1=Conv2DTranspose(12,(2,2),strides=strides, padding=padding)(conv5)
    decon1=Convolution2D(256, (1, 1), activation='relu',padding='same')(decon1)  
    
    up6 = concatenate([decon1, after_conv4], axis=3)
    
    conv6 = inception_block(up6, 256, splitted=splitted, activation=act, name = 'INCEP_6')
    #conv6=Dropout(conv6)
    
    
    after_conv3 = resblock(conv3, 1, 128)
    decon2=Conv2DTranspose(24,(2,2),strides=strides, padding=padding)(conv6)
    decon2=Convolution2D(128, (1, 1), activation='relu',padding='same')(decon2) 
    up7 = concatenate([decon2, after_conv3], axis=3)
    #conv7=Dropout(up7)
    conv7 = inception_block(up7, 128, splitted=splitted, activation=act)
   
    
    after_conv2 = resblock(conv2, 1, 64)
    decon3=Conv2DTranspose(48,(2,2),strides=strides, padding=padding)(conv7)
    decon3=Convolution2D(64, (1, 1), activation='relu',padding='same')(decon3) 
    up8 = concatenate([decon3, after_conv2], axis=3)
    #conv8=Dropout(up8)
    conv8 = inception_block(up8, 64, splitted=splitted, activation=act)
   
    after_conv1 = resblock(conv1, 1, 32)
    decon4=Conv2DTranspose(96,(2,2),strides=strides, padding=padding)(conv8)
    decon4=Convolution2D(32, (1, 1), activation='relu',padding='same')(decon4) 
    up9 = concatenate([decon4, after_conv1], axis=3)
    #conv9=Dropout(up9)
    conv9 = inception_block(up9, 32, splitted=splitted, activation=act, name = 'INCEP_7')
    
    conv10 = Convolution2D(1, (1, 1), activation='sigmoid',name='main_output')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    return model
    
