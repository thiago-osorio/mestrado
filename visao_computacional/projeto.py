# Bibliotecas
import os
import sys
import boto3
import warnings
import subprocess

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.applications import MobileNet, EfficientNetB0, DenseNet121, ResNet50, vgg16, Xception
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, InputLayer, MaxPooling2D, Dropout, concatenate, Input, GlobalAveragePooling2D

warnings.filterwarnings('ignore')

# Funções
def normalizer(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

# Dados -> https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
train = image_dataset_from_directory(
    "data/train",
    image_size=(128, 128),
    batch_size=32,
    label_mode='int'
)
train = train.map(normalizer)

test = image_dataset_from_directory(
    "data/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode='int'
)
test = test.map(normalizer)

val = image_dataset_from_directory(
    "data/val",
    image_size=(128, 128),
    batch_size=32,
    label_mode='int'
)
val = val.map(normalizer)

# Código
## LeNet-5 Architecture
lenet_5 = Sequential()

lenet_5.add(InputLayer(input_shape=(128, 128, 3)))

lenet_5.add(Conv2D(filters=6, kernel_size=(5, 5), activation='tanh'))
lenet_5.add(AveragePooling2D(pool_size=(2, 2)))

lenet_5.add(Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))
lenet_5.add(AveragePooling2D(pool_size=(2, 2)))

lenet_5.add(Flatten())

lenet_5.add(Dense(120, activation='tanh'))
lenet_5.add(Dense(84, activation='tanh'))
lenet_5.add(Dense(1, activation='sigmoid'))

lenet_5.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

lenet_5.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = lenet_5.evaluate(test)
print(f"""Métricas Teste LeNet-5
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

lenet_5.save('lenet_5.h5')

## AlexNet Architecture
alexnet = Sequential()

alexnet.add(InputLayer(input_shape=(128,128,3)))

alexnet.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'))
alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

alexnet.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
alexnet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

alexnet.add(Flatten())

alexnet.add(Dense(4096, activation='relu'))
alexnet.add(Dropout(0.5))

alexnet.add(Dense(4096, activation='relu'))
alexnet.add(Dropout(0.5))

alexnet.add(Dense(1, activation='sigmoid'))

alexnet.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

alexnet.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = alexnet.evaluate(test)
print(f"""Métricas Teste AlexNet
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

alexnet.save('alexnet.h5')

## Inception-v1 Architecture
input_layer = Input(shape=(128,128,3))

model = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(input_layer)
model = MaxPooling2D((3,3), strides=(2,2), padding='same')(model)
model = Conv2D(64, (1,1), padding='same', activation='relu')(model)
model = Conv2D(192, (3,3), padding='same', activation='relu')(model)
model = MaxPooling2D((3,3), strides=(2,2), padding='same')(model)

conv_1x1_1 = Conv2D(64, (1,1), padding='same', activation='relu')(model)
conv_3x3_reduce_1 = Conv2D(96, (1,1), padding='same', activation='relu')(model)
conv_3x3_1 = Conv2D(128, (3,3), padding='same', activation='relu')(conv_3x3_reduce_1)
conv_5x5_reduce_1 = Conv2D(16, (1,1), padding='same', activation='relu')(model)
conv_5x5_1 = Conv2D(32, (5,5), padding='same', activation='relu')(conv_5x5_reduce_1)
pool_proj_1 = Conv2D(32, (1,1), padding='same', activation='relu')(MaxPooling2D((3,3), strides=(1,1), padding='same')(model))
inception_1 = concatenate([conv_1x1_1, conv_3x3_1, conv_5x5_1, pool_proj_1], axis=-1)

model = AveragePooling2D((7,7), strides=(1,1))(inception_1)
model = Flatten()(model)
model = Dense(1024, activation='relu')(model)
output_layer = Dense(1, activation='softmax')(model)

inception_v1 = Model(inputs=input_layer, outputs=output_layer)

inception_v1.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

inception_v1.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = inception_v1.evaluate(test)
print(f"""Métricas Teste Inception-v1
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

inception_v1.save('inception_v1.h5')

## MobileNet LightWeight
mobilenet = Sequential()

mobilenet.add(InputLayer(input_shape=(128, 128, 3)))

mobilenet_base = MobileNet(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.5,
    pooling=None
)

mobilenet.add(mobilenet_base)

mobilenet.add(GlobalAveragePooling2D())
mobilenet.add(Dense(1, activation='sigmoid'))

mobilenet.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

mobilenet.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = mobilenet.evaluate(test)
print(f"""Métricas Teste MobileNet v1
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

mobilenet.save('mobilenet.h5')

## EfficientNetB0 LightWeight
efficientenetb0 = Sequential()

efficientenetb0 = Sequential()

efficientenetb0.add(InputLayer(input_shape=(128, 128, 3)))

efficientenetb0_base = EfficientNetB0(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)

efficientenetb0.add(efficientenetb0_base)

efficientenetb0.add(GlobalAveragePooling2D())
efficientenetb0.add(Dense(1, activation='sigmoid'))

efficientenetb0.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

efficientenetb0.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = efficientenetb0.evaluate(test)
print(f"""Métricas Teste EfficientNetB0
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

efficientenetb0.save('efficientenetb0.h5')

## DenseNet121 LightWeight
densenet121 = Sequential()

densenet121.add(InputLayer(input_shape=(128, 128, 3)))

densenet121_base = DenseNet121(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)

densenet121.add(densenet121_base)

densenet121.add(GlobalAveragePooling2D())
densenet121.add(Dense(1, activation='sigmoid'))

densenet121.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

densenet121.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = densenet121.evaluate(test)
print(f"""Métricas Teste DenseNet121
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

densenet121.save('densenet121.h5')

## ResNet50 Transfer Learning
resnet50 = Sequential()

resnet50.add(InputLayer(input_shape=(128, 128, 3)))

resnet50_base = ResNet50(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)

resnet50.add(resnet50_base)

resnet50.add(GlobalAveragePooling2D())
resnet50.add(Dense(1, activation='sigmoid'))

resnet50.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

resnet50.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = resnet50.evaluate(test)
print(f"""Métricas Teste ResNet50
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

resnet50.save('resnet50.h5')

## VGG16 Transfer Learning
vgg16 = Sequential()

vgg16.add(InputLayer(input_shape=(128, 128, 3)))

vgg16_base = VGG16(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)

vgg16.add(vgg16_base)

vgg16.add(GlobalAveragePooling2D())
vgg16.add(Dense(1, activation='sigmoid'))

vgg16.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

vgg16.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = vgg16.evaluate(test)
print(f"""Métricas Teste VGG16
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

vgg16.save('vgg16.h5')

## Xception Transfer Learning
xception = Sequential()

xception.add(InputLayer(input_shape=(128, 128, 3)))

xception_base = Xception(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)

xception.add(xception_base)

xception.add(GlobalAveragePooling2D())
xception.add(Dense(1, activation='sigmoid'))

xception.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

xception.fit(train, epochs=10, validation_data=val)

test_loss, test_acc, test_prec, test_rec = xception.evaluate(test)
print(f"""Métricas Teste Xception
      Loss: {test_loss}
      Accuracy: {test_acc}
      Precision: {test_prec}
      Recall: {test_rec}
      """)

xception.save('xception.h5')

# Out-of-Sample - Best Model
## Dados -> https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
oos = image_dataset_from_directory(
    "data/real_data",
    image_size=(128, 128),
    batch_size=32,
    label_mode='int'
)
oos = oos.map(normalizer)

## Modelo
model = load_model('vgg16.h5')
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

## Evaluation
oos_loss, oos_acc, oos_prec, oos_recall = model.evaluate(oos)
print(f"""Métricas Out-Of-Sample VGG16
      Loss: {oos_loss}
      Accuracy: {oos_acc}
      Precision: {oos_prec}
      Recall: {oos_recall}
      """)