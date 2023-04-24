import sys

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

import tensorflow_datasets as tfds

import torch

from keras.layers.fake_approx_convolutional import FakeApproxConv2D 


from autoattack import AutoAttack
import utils_tf2

USE_APPROX = False

def load_network(network_path):

    vgg16 = VGG16(
        weights=None,
        include_top=False,
        pooling="avg",
        input_shape=(224,224,3)
    )

    if USE_APPROX:
        #layers = vgg16.get_layer
        #print(layers)
        faked_vgg = []
        for layer in vgg16.layers:
            if isinstance(layer, Conv2D):
#                print(dir(layer))
                faked_vgg.append(
                    FakeApproxConv2D(
                        layer.filters,
                        layer.kernel_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        data_format=layer.data_format,
                        dilation_rate=layer.dilation_rate,
                        activation=layer.activation,
                        use_bias=layer.use_bias,
                        kernel_initializer=layer.kernel_initializer,
                        bias_initializer=layer.bias_initializer,
                        kernel_regularizer=layer.kernel_regularizer,
                        bias_regularizer=layer.bias_regularizer,
                        activity_regularizer=layer.activity_regularizer,
                        kernel_constraint=layer.kernel_constraint,
                        bias_constraint=layer.bias_constraint,
                    )
                )
            else:
                faked_vgg.append(layer)
    #        vgg16 = Sequential(faked_vgg)
    
    
    input_layer = Input(shape=(224, 224,3))
    # # #    data_augmentation = RandomRotation(0.2)(input_layer)
    out_vgg16 = vgg16(input_layer)
    layer = Flatten()(out_vgg16)
    layer = Dense(4096, activation="relu")(layer)
    layer = Dense(1024, activation="relu")(layer)
    layer = Dense(10)(layer)
    output_layer = Activation("softmax")(layer)

    net = Model(inputs=input_layer, outputs=output_layer)


    optimizer = tf.keras.optimizers.Adam()
    net.compile(optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    net.load_weights(network_path)

    
    return net

def create_twin(net):

    vgg16 = VGG16(
        weights=None,
        include_top=False,
        pooling="avg",
        input_shape=(224,224,3)
    )

    faked_vgg = [Input(shape=(224,224,3))]
    for layer in vgg16.layers:
        if USE_APPROX and isinstance(layer, Conv2D):
            #                print(dir(layer))
            faked_vgg.append(
                FakeApproxConv2D(
                    layer.filters,
                    layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    data_format=layer.data_format,
                    dilation_rate=layer.dilation_rate,
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                    kernel_initializer=layer.kernel_initializer,
                    bias_initializer=layer.bias_initializer,
                    kernel_regularizer=layer.kernel_regularizer,
                    bias_regularizer=layer.bias_regularizer,
                    activity_regularizer=layer.activity_regularizer,
                    kernel_constraint=layer.kernel_constraint,
                    bias_constraint=layer.bias_constraint,
                )
            )
        else:
            faked_vgg.append(layer)
            #        vgg16 = Sequential(faked_vgg)

    faked_vgg += [
        Flatten(),
        Dense(4096, activation="relu"),
        Dense(1024, activation="relu"),
        Dense(10),
        Activation("softmax")
    ]
            
    new_network = Sequential(faked_vgg)

    optimizer = tf.keras.optimizers.Adam()
    new_network.compile(optimizer=optimizer,
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])
    
    # copy weights per layer
    vgg16_layers = len(vgg16.layers)
    for dest, source in zip(new_network.layers[:vgg16_layers], net.layers[1].layers):
        dest.set_weights(source.get_weights())
    for dest, source in zip(new_network.layers[vgg16_layers:], net.layers[2:]):
        dest.set_weights(source.get_weights())
        
    return new_network
        
def load_data():
    
    def normalize_img(image):
        image = tf.cast(image, tf.float32) 
        image = tf.image.resize(image, (224,224))
        image = preprocess_input(image)
        return image
        
    batch_size = 64

    (trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

    print(testX.shape)
    testX_list = []
    for image in testX:
        testX_list.append(normalize_img(image))

    with tf.device("cpu"):
        testX = tf.stack(testX_list)

    # (ds_test, ds_val, ds_train), ds_info = tfds.load(
    #     "cifar10",
    #     split=["test", "train[0%:10%]", "train[10%:]"],
    #     as_supervised = True,
    #     with_info = True
    # )
    # del ds_val
    # del ds_train
    
    # ds_test = ds_test.map(normalize_img, num_parallel_calls=1)
    # ds_test = ds_test.batch(batch_size)
    # #    ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(batch_size)
    
    return testX, testY
    
    

def main(network_path, epsilon):

        
    net = load_network(network_path)
    net = create_twin(net)
    #    net.summary()

    testX, testY = load_data()
    print("data loaded")

    loss, accuracy = net.evaluate(testX, testY)
    print(accuracy)
    
    # convert to pytorch format
    #    torch_testY = np.argmax(testY, axis=1)
    torch_testX = torch.from_numpy(np.transpose(testX, (0, 3, 1, 2)).copy()).float()
    #print("tf", testX.shape)
    #torch_testX = torch.from_numpy(testX.numpy()).float()
    torch_testY = torch.from_numpy( testY.flatten() ).float()
    print("torch", torch_testX.shape)
    
    # print(torch_testY.shape)
    # return

    print("READY")

    tf_model = net
    
    atk_model = tf.keras.models.Model(inputs=tf_model.input, outputs=tf_model.get_layer(index=-2).output) 
    atk_model.summary()
    model_adapted = utils_tf2.ModelAdapter(atk_model)

    # run attack
    adversary = AutoAttack(model_adapted, norm='Linf', eps=epsilon, version='standard', is_tf_model=True)
    adversary.attacks_to_run = ['square']
    adversary.square.n_queries = 5000

    batch_size = 32

    # sample = sample.reshape(1, 3, 224, 224)
    #     sampleX = sample.to("cuda")
    #     sampleY = torch_testY[i].reshape(1)
    #     print(sampleY.shape)
    #     #break
        
    ret_dict = adversary.run_standard_evaluation(torch_testX[:64], torch_testY[:64], bs=batch_size, return_labels=True)
    # np_x_adv = np.moveaxis(x_adv.cpu().numpy(), 1, 3)
    # np.save("./output/mnist_adv.npy", np_x_adv)

    
    
    print("---- END ---")


if __name__ == "__main__":
    network_path = sys.argv[1]
    epsilon = 8/255
    
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main(network_path, epsilon)
