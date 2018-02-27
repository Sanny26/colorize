from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.layers import Input, Lambda, Add
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, concatenate
from keras.preprocessing import image

import pdb
import os
import tensorflow as tf
import numpy as np
#def custom_loss(y_true, y_pred):



input_image = Input(shape= (224, 224, 1) )
img = concatenate([input_image, input_image], axis=-1)
img = concatenate([img, input_image], axis = -1)

vgg_model = VGG16(include_top=False, weights= "imagenet", input_tensor = img) #Y 

for layer in vgg_model.layers:
    layer.trainable = False 

x = BatchNormalization(name = "bn1" )(vgg_model.get_layer(name="block4_conv3").output)
x = Conv2D(256, (1, 1), activation="relu", name= "conv1" )(x)
up1 = UpSampling2D(size = (2,2))(x)
x1 = BatchNormalization(name = "bn2")(vgg_model.get_layer(name="block3_conv3").output)
x = Add()([x1, up1])
x = Conv2D(128, (1, 1), activation="relu", name="conv2" )(x)
up2 = UpSampling2D(size = (2,2))(x)
x2 = BatchNormalization(name="bn3")(vgg_model.get_layer(name="block2_conv2").output)
x = Add()([x2, up2])
x = Conv2D(64, (1, 1), activation="relu", name="conv3" )(x)
up3 = UpSampling2D(size = (2,2))(x)
x3 = BatchNormalization(name="bn4")(vgg_model.get_layer(name="block1_conv2").output)
x = Add()([x3, up3])
x = Conv2D(3, (1, 1), activation="relu", name="conv4" )(x)
uv = Conv2D(2, (1,1), activation="relu", name="conv5")(x) ##UV output
pred_img = concatenate([input_image, uv], axis = -1)


model = Model(inputs = input_image, outputs= pred_img)
model.compile(loss = 'mean_squared_error',
                optimizer = 'sgd',
                metrics=['accuracy'])

input_path = "input/"
output_path = "output/"
files = os.listdir(input_path)[:2]
X = np.empty((2, 224, 224, 1))
y = np.empty((2, 224, 224, 3))
for i, f in enumerate(files):
    img = image.load_img(input_path+f, grayscale = True, target_size=(224, 224))
    x = image.img_to_array(img)
    X[i]=x

    pred_img = image.load_img(output_path + f, target_size=(224, 224))
    x = image.img_to_array(img)
    y[i]= x
    
 
model.fit(X, y) 
model.save("arch.h5")



