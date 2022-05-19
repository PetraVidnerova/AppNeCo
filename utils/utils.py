from math import floor
import torch.nn as nn
import numpy as np


def n_parameters(net):

    manual_params = 0

    for layer in net.features:
        if isinstance(layer, nn.ReLU):
            continue
        if isinstance(layer, nn.MaxPool2d):
            continue
        # if isinstance(layer, nn.BatchNorm2d):
        #     manual_params += 2 * layer.num_features
        #     continue
        if isinstance(layer, nn.Conv2d):
            manual_params += (layer.in_channels *
                              layer.kernel_size[0]*layer.kernel_size[1]+1)*layer.out_channels
        else:
            print(layer)
            raise ValueError("unknown layer type")

    for layer in net.classifier:
        if isinstance(layer, nn.Linear):
            manual_params += (layer.in_features + 1) * layer.out_features
            continue
        if isinstance(layer, nn.Dropout):
            continue
        if isinstance(layer, nn.ReLU):
            continue

    return manual_params

def out_features(input_shape, layer):

    if input_shape[1] != input_shape[2]:
        raise NotImplementedError

    if isinstance(layer, nn.ReLU):
        return input_shape

    if isinstance(layer, nn.Linear):
        return (1, layer.out_features)

    if isinstance(layer, nn.Dropout):
        return input_shape


    if isinstance(layer, nn.MaxPool2d):
        return (input_shape[0],
                floor((input_shape[1] - layer.kernel_size + 2*layer.padding)/layer.stride)+1,
                floor((input_shape[1] - layer.kernel_size + 2*layer.padding)/layer.stride)+1
                )

    if isinstance(layer, nn.Conv2d):
        return (layer.out_channels,
                floor((input_shape[1] - layer.kernel_size[0] + 2*layer.padding[0])/layer.stride[0])+1,
                floor((input_shape[1] - layer.kernel_size[0] + 2*layer.padding[0])/layer.stride[0])+1
                )
    


        
