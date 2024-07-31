import tqdm
import torch
import torch.nn as nn

from network import SmallDenseNet, SmallConvNet
from dataset import create_dataset


def eval_one_sample(net, sample):
    """evaluates one sample and returns boolean vector 
    of relu's saturations"""
    saturations = []
    
    outputs = sample
    for child in net.children():
        assert isinstance(child, nn.Sequential)
        for layer in child:
            outputs = layer(outputs)
            if isinstance(layer, nn.ReLU):
                sat1 = outputs != 0
                sat2 = outputs == 0
                sat2 = torch.logical_not(sat2)

                assert torch.all(sat1 == sat2)
                
                saturations.append(outputs != 0)     
    return saturations
            
def prune_network(net, saturations):
    layers = []
    for child in net.children():
        assert isinstance(child, nn.Sequential)
        for layer in child:
            layers.append(layer)
            
    for j, saturation in enumerate(saturations):

        # find the j-th ReLU layer and get previous Linear layer
        jj = j 
        for i, l in enumerate(layers):
            if isinstance(l, nn.ReLU):
                if jj == 0:
                    break
                else:
                    jj -= 1
        i -= 1
        assert isinstance(layers[i], nn.Linear)
    
        W, b = layers[i].weight, layers[i].bias
        saturation = saturation.flatten()

        # filter out previous linear layer
        W2 = W[saturation]
        b2 = b[saturation]

        new_pre_layer = nn.Linear(W2.shape[1], W2.shape[0]).double()
        new_pre_layer.weight.data = W2
        new_pre_layer.bias.data = b2

        layers[i] = new_pre_layer

        # find next Linear layer 
        i += 1
        while not isinstance(layers[i], nn.Linear): 
            i += 1
        assert isinstance(layers[i], nn.Linear)
        W, b = layers[i].weight, layers[i].bias

        W2 = W[:, saturation]

        new_post_layer = nn.Linear(W2.shape[1], W2.shape[0]).double()
        new_post_layer.weight.data = W2
        new_post_layer.bias.data = b
        # bias stays the same

        layers[i] = new_post_layer
    
    # create a fixed network
    net = nn.Sequential(*layers).cuda().eval()
    
    print(net)
    return net

def squeeze_network(net):
    layers = []
    assert isinstance(net, nn.Sequential)
    for layer in net:
        layers.append(layer)

    # get rid of ReLU (network already pruned)
    layers = [l for l in layers if not isinstance(l, nn.ReLU)]
    # we  do not need dropout (only eval mode)
    layers = [l for l in layers if not isinstance(l, nn.Dropout)]

    # check that all layers are linear (first can be flatten)
    assert isinstance(layers[0], nn.Flatten) or isinstance(layers[0], nn.Linear)
    for l in layers[1:]:
        assert isinstance(l, nn.Linear)
    

    # take only linear layers 
    lin_layers = [l for l in layers if isinstance(l, nn.Linear)] 

    W = [l.weight.data for l in lin_layers[::-1]]
    b = [l.bias.data for l in lin_layers[::-1]]

    
    W_new = torch.linalg.multi_dot(W)
    bias_new = b[0]
    for i, bias in enumerate(b[1:]):
        ii = i + 1
        if ii > 1:
            W_inter = torch.linalg.multi_dot(W[:ii])
        else:
            W_inter = W[0]
        bias_new += torch.mm(W_inter, bias.reshape((-1, 1))).flatten()

        
    new_layer = nn.Linear(W[-1].shape[1], W[0].shape[0]).double()
    new_layer.weight.data = W_new
    new_layer.bias.data = bias_new

    new_layers = [new_layer]
    if isinstance(layers[0], nn.Flatten):
        new_layers = [layers[0]] + new_layers

    return nn.Sequential(*new_layers).cuda().eval()
    
    
def load_network(network_class, network_path):
    net = network_class()
    net.load_state_dict(torch.load(network_path))
    net.eval()
    net.cuda()
    net.double()
    return  net


def test_squeeze():
    
    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

    for inputs, labels in data:
        inputs = inputs.cuda().double()
        assert inputs.shape[0] == 1 # one sample in a batch
        saturations = eval_one_sample(net, inputs)
        
        pnet = prune_network(net, saturations)

        outputs = net(inputs)
        outputs2 = pnet(inputs)
        print(nn.functional.mse_loss(outputs, outputs2))

        snet = squeeze_network(pnet)
        print(snet)

        outputs3 = snet(inputs)
        print(outputs)
        print(outputs3)
        print(nn.functional.mse_loss(outputs, outputs3))
        
        exit()


def lower_precision(net):
    return net.half().double()

def stack_linear_layers(layer1, layer2):

    wide_W1 = torch.hstack([layer1.weight.data,
                            torch.zeros(*layer1.weight.data.shape).double().cuda()])
    wide_W2 = torch.hstack([torch.zeros(*layer2.weight.data.shape).double().cuda(),
                            layer2.weight.data])
                            
    new_weight = torch.vstack([wide_W1, wide_W2])

    new_layer = nn.Linear(new_weight.shape[1], new_weight.shape[0]).double()
    new_layer.weight.data = new_weight
    new_layer.bias.data = torch.hstack([layer1.bias.data, layer2.bias.data])
    
    return new_layer

def create_comparing_network(net, net2):
    twin = lower_precision(net2) 

    layer_list = []

    sequence1 = next(iter(net.children()))
    assert isinstance(sequence1, nn.Sequential)

    sequence2 = next(iter(net2.children()))
    assert isinstance(sequence2, nn.Sequential)

    for layer1, layer2  in zip(sequence1, sequence2):
        if isinstance(layer1, nn.Flatten):
            assert isinstance(layer2, nn.Flatten)
            layer_list.append(layer1)
        elif isinstance(layer1, nn.Dropout):
            assert isinstance(layer2, nn.Dropout)
            layer_list.append(layer1)
        elif isinstance(layer1, nn.ReLU):
            assert isinstance(layer2, nn.ReLU)
            layer_list.append(layer1)
        elif isinstance(layer1, nn.Linear):
            assert isinstance(layer2, nn.Linear)
            layer_list.append(stack_linear_layers(layer1, layer2))
        else:
            raise NotImplementedError

    layer_list.append(nn.ReLU())
    
    output_layer = nn.Linear(20, 1) # TODO fix  the number
    
    layer_list.append(output_layer)
    
    return nn.Sequential(*layer_list).cuda()
    
def main(): 

    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    compnet = create_comparing_network(net, net2)

    print(compnet)

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

    for inputs, labels in data:
        inputs = inputs.cuda().double()
        assert inputs.shape[0] == 1 # one sample in a batch

        wide_inputs = torch.hstack([inputs, inputs])

        outputs = net(inputs)
        outputs2 = net2(inputs)
        wide_outputs = compnet(wide_inputs)

        print(outputs)
        print(outputs2)
        print(wide_outputs)
        
        exit()
        
if __name__ == "__main__":

    main()
