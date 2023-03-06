import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Resizing, Input, Flatten, Dense 
from tensorflow.keras.models import Model

# prepare data 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
print(x_train.shape)
print("Dataset loaded.")

# prepare model 
vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

print(vgg16)
vgg16.summary()

input_size = (32, 32, 3)

# input part 
input_layer = Input(shape=(32,32,3))
resize_layer = Resizing(224, 224)(input_layer)
# vgg16 conovolutional layers 
output_vgg16 = vgg16(resize_layer)
# top part
layer = Flatten()(output_vgg16)
layer = Dense(4096, activation="relu")(layer)
layer = Dense(1024, activation="relu")(layer)
output_layer = Dense(10, activation="softmax")(layer)

net = Model(inputs=input_layer, outputs=output_layer)

print(net)
net.summary()

net.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

net.fit(x_train, y_train, epochs=50, batch_size=64) 
