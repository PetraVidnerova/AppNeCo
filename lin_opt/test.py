import gc
import numpy as np
import tqdm
import torch
import torch.nn as nn

from network import SmallDenseNet, SmallConvNet
from dataset import create_dataset
from torch_conv_layer_to_fully_connected import torch_conv_layer_to_affine
from sparse_utils import SparseLinear, multidot

def eval_one_sample(net, sample):
    """evaluates one sample and returns boolean vector 
    of relu's saturations"""
    saturations = []
    
    outputs = sample
    assert isinstance(net, nn.Sequential)
    for layer in net:
        outputs = layer(outputs)
        if isinstance(layer, nn.ReLU):
            saturations.append(outputs != 0)     
    return saturations
            
def prune_network(net, saturations, cuda=True):
    layers = []
    assert isinstance(net, nn.Sequential)
    for layer in net:
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
        assert (isinstance(layers[i], nn.Linear) or
                isinstance(layers[i], SparseLinear))

        saturation = saturation.flatten()
        saturation_indices = (saturation.nonzero()).flatten()
        move = torch.cumsum(torch.logical_not(saturation), dim=0)

        if isinstance(layers[i], nn.Linear):
            W, b = layers[i].weight, layers[i].bias
            # filter out previous linear layer
            W2 = W[saturation]
            b2 = None if b is None else b[saturation]

            new_pre_layer = nn.Linear(W2.shape[1], W2.shape[0]).double()
            new_pre_layer.weight.data = W2
            assert b2 is not None
            new_pre_layer.bias.data = b2
        else:   
            W, b = layers[i].w, layers[i].b
            row_indices = saturation_indices
            indices = W._indices()
            values = W._values()
            bitmap = torch.isin(indices[0], row_indices)
            indices = indices[:, bitmap]
            indices[0] = indices[0] - move[indices[0]]
            values = values[bitmap]
            W2 = torch.sparse_coo_tensor(indices=indices,
                                         values=values,
                                         size=(len(row_indices), W.shape[1]))
            if b is None:
                b2 = None
            else:
                b2 = b[saturation]
            new_pre_layer = SparseLinear(W2, b2)        
        
        layers[i] = new_pre_layer

        # find next Linear layer 
        i += 1
        while not (isinstance(layers[i], nn.Linear) or
                   isinstance(layers[i], SparseLinear)): 
            i += 1
        assert (isinstance(layers[i], nn.Linear) or
                isinstance(layers[i], SparseLinear))

        if isinstance(layers[i], nn.Linear):    
            W, b = layers[i].weight, layers[i].bias

            W2 = W[:, saturation]
            b2 = None if b is None else torch.clone(b)
            new_post_layer = nn.Linear(W2.shape[1], W2.shape[0]).double()
            new_post_layer.weight.data = W2
            new_post_layer.bias.data = b2
            # bias stays the same
        else:
            W, b = layers[i].w, layers[i].b
            col_indices = saturation_indices
            indices = W._indices()
            values = W._values()
            bitmap = torch.isin(indices[1], col_indices)
            indices = indices[:, bitmap]
            indices[1] = indices[1] - move[indices[1]]
            values = values[bitmap]
            W2 = torch.sparse_coo_tensor(indices=indices,
                                         values=values,
                                         size=(W.shape[0], len(col_indices))
                                         )
            new_post_layer = SparseLinear(W2, None if b is None else torch.clone(b))
            
        layers[i] = new_post_layer
    
    # create a fixed network
    net = nn.Sequential(*layers)
    if cuda:
        net = net.cuda()
    else:
        net = net.cpu()
    net = net.eval()
    
    print(net)
    return net

def squeeze_network(net, cuda=True):
    layers = []
    assert isinstance(net, nn.Sequential)
    for layer in net:
        layers.append(layer)

    # get rid of ReLU (network already pruned)
    layers = [l for l in layers if not isinstance(l, nn.ReLU)]
    # we  do not need dropout (only eval mode)
    layers = [l for l in layers if not isinstance(l, nn.Dropout)]

    # check that all layers are linear (first can be flatten)
    #assert isinstance(layers[0], nn.Flatten) or isinstance(layers[0], nn.Linear)
    for l in layers:
        assert isinstance(l, nn.Linear) or isinstance(l, SparseLinear)
    

    # take only linear layers 
    lin_layers = [l for l in layers if (isinstance(l, nn.Linear) or isinstance(l, SparseLinear))] 

    def get_w(l):
        if isinstance(l, nn.Linear):
            return l.weight.data
        else:
            return l.w.cuda() if cuda else l.w.cpu()

    def get_b(l):
        if isinstance(l, nn.Linear):
            if l.bias is None:
                b = torch.zeros(l.weight.data.shape[0], dtype=torch.float64)
                if cuda:
                    b = b.cuda()
                else:
                    b = b.cpu()
                return b
            return l.bias.data
        else:
            if l.b is None:
                b = torch.zeros(l.w.shape[0], dtype=torch.float64)
                if cuda:
                    b = b.cuda()
                return b
            return l.b.cuda() if cuda else l.b.cpu()
    
    W = [get_w(l) for l in lin_layers[::-1]]
    b = [get_b(l) for l in lin_layers[::-1]]

    W_new = multidot(W)
    bias_new = b[0]
    for i, bias in enumerate(b[1:]):
        ii = i + 1
        if ii > 1:
            W_inter = multidot(W[:ii])
        else:
            W_inter = W[0]
        bias_new += torch.mm(W_inter, bias.reshape((-1, 1))).flatten()

    if W_new.is_sparse:
        W_new = W_new.to_dense()
    
    new_layer = nn.Linear(W[-1].shape[1], W[0].shape[0]).double()
    new_layer.weight.data = W_new
    new_layer.bias.data = bias_new

    new_layers = [new_layer]
    #    if isinstance(layers[0], nn.Flatten):
    #        new_layers = [layers[0]] + new_layers

    res = nn.Sequential(*new_layers)
    if cuda:
        res = res.cuda()
    res = res.eval()
    return res
    
    
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
    net = next(iter(net.children())) # extract the nn.Sequential

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

        assert nn.functional.mse_loss(outputs, outputs3).isclose(torch.tensor(0.0, dtype=torch.float64))
        print("oh yes")

def test_compnet():

    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    compnet = create_comparing_network(net, net2)

    net2 = lower_precision(net2)
    
    print(compnet)

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

        
    for inputs, labels in data:
        inputs = inputs.cuda().double()

        output = net(inputs)
        output2 = net2(inputs)

        err1 = (output-output2).abs().sum()
        err2 = compnet(inputs)

        assert err1.isclose(err2)
        print("oh yes")

def test_squeezed_compnet():

    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    compnet = create_comparing_network(net, net2)

    net2 = lower_precision(net2)
    
    print(compnet)

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

        
    for inputs, labels in data:
        inputs = inputs.cuda().double()

        output = net(inputs)
        output2 = net2(inputs)

        err1 = (output-output2).abs().sum()
        err2 = compnet(inputs)

        assert err1.isclose(err2)

        saturations = eval_one_sample(compnet, inputs)        
        target_net = squeeze_network(prune_network(compnet, saturations))

        err3 = target_net(inputs)
        assert err1.isclose(err3)
        
        print("oh yes")

        
def lower_precision(net):
    return net.half().double()

def widen_net(net, left=True):
    #make the first linear layer twice wider
    layer1 = net[0]
    print(layer1)
    
    new_layer = SparseLinear(
        layer1.weight.data.to_sparse(),
        None if layer1.bias is None else layer1.bias.data
    )
    if left:
        new_layer.w = torch.sparse_coo_tensor(
            indices = new_layer.w._indices(),
            values = new_layer.w._values(),
            size=(new_layer.w.shape[0], 2*new_layer.w.shape[1])
        )
    else:
        moved_indices = torch.vstack([
            new_layer.w._indices()[0],
            new_layer.w._indices()[1]+new_layer.w.shape[1]
        ])
        new_layer.w = torch.sparse_coo_tensor(
            indices = moved_indices,
            values = new_layer.w._values(),
            size = (new_layer.w.shape[0], 2*new_layer.w.shape[1])
        )
    net[0] = new_layer
    return net

"""
def stack_network_lists(layer1, layer2):
    new_network_list = [] 
    for net in layer1.network_list+layer2.network_list:
        wnet = widen_net(net)
        new_network_list.append(wnet)
    # make it flatten, all layers work with flatten 
    output_shape = (np.prod(layer1.output_shape) + np.prod(layer2.output_shape),)
    return ListOfNetworks(new_network_list, output_shape).cuda()
"""

def stack_linear_layers(layer1, layer2, common_input=False):

    
    
    if isinstance(layer1, nn.Linear):
        assert isinstance(layer2, nn.Linear)
        w1 = layer1.weight.data.to_sparse_coo()
        b1 = layer1.bias.data
        w2 = layer2.weight.data.to_sparse_coo()
        b2 = layer2.bias.data
    else:
        assert isinstance(layer1, SparseLinear)
        assert isinstance(layer2, SparseLinear)
        w1 = layer1.w.to_sparse_coo()
        b1 = layer1.b
        w2 = layer2.w.to_sparse_coo()
        b2 = layer2.b
        
    if common_input:
        wide_W1 =  w1
        wide_W2 =  w2
    else:
        wide_W1 = torch.sparse_coo_tensor(
            indices = w1._indices(),
            values = w1._values(),
            size=(w1.shape[0], w1.shape[1]+w2.shape[1]))

        moved_indices = torch.vstack([
            w2._indices()[0],
            w2._indices()[1]+ w1.shape[1]
        ])
        
        wide_W2 = torch.sparse_coo_tensor(
            indices = moved_indices,
            values = w2._values(),
            size = (w1.shape[0], w1.shape[1]+w2.shape[1])
        )
        
       # wide_W1 = torch.hstack([layer2.weight.data,
       #               torch.zeros(*layer1.weight.data.shape).double().cuda()])
       # wide_W2 = torch.hstack([torch.zeros(*layer2.weight.data.shape).double().cuda(),
       #                         layer2.weight.data])
       #
        
    # new_weight = torch.vstack([wide_W1, wide_W2])

    new_weight = torch.vstack([wide_W1, wide_W2])
    if b1 is not None and b2 is not None:
        new_layer = SparseLinear(new_weight, torch.hstack([b1, b2]))
    else:
        assert b1 is None
        assert b2 is None
        new_layer = SparseLinear(new_weight, None)
    
    #    new_layer.weight.data = new_weight
    #    new_layer.bias.data = torch.hstack([layer1.bias.data, layer2.bias.data])
    
    return new_layer

def magic_layer(layer1, layer2):
    """ equation (13) and (14) in Jirka's document """
    
    W1 = layer1.weight.data
    b1 = layer1.bias.data

    W2 = layer2.weight.data
    b2 = layer2.bias.data

    # magic_b1 = b1 - b2
    # mabic_b2 = b2 - b1
    magic_b = torch.hstack([b1-b2, b2-b1])

    # magic W  =  W1 -W2
    #            -W1  W2 
    magic_W = torch.vstack(
        [
            torch.hstack([W1, -W2]),
            torch.hstack([-W1, W2])
        ]
    )

    new_layer = nn.Linear(magic_W.shape[1], magic_W.shape[0]).double()
    new_layer.weight.data = magic_W
    new_layer.bias.data = magic_b

    return new_layer
    
    
    
def create_comparing_network(net, net2):
    twin = lower_precision(net2) 

    layer_list = []
    
    if not isinstance(net, nn.Sequential):
        sequence1 = next(iter(net.children()))
    else:
        sequence1 = net
    assert isinstance(sequence1, nn.Sequential)

    if not isinstance(twin, nn.Sequential):
        sequence2 = next(iter(twin.children()))
    else:
        sequence2 = twin
    assert isinstance(sequence2, nn.Sequential)

    first_linear = True
    
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
            layer_list.append(stack_linear_layers(layer1, layer2, common_input=first_linear))
            first_linear = False
        elif isinstance(layer1, SparseLinear):
            assert isinstance(layer2, SparseLinear)
            layer_list.append(stack_linear_layers(layer1, layer2, common_input=first_linear))
            first_linear = False
        #elif isinstance(layer1, ListOfNetworks):
        ##    assert isinstance(layer2, ListOfNetworks)
        #    layer_list.append(stack_network_lists(layer1, layer2))
        else:
            print(type(layer1))
            raise NotImplementedError

    assert isinstance(sequence1[-1], nn.Linear) 
    assert isinstance(sequence2[-1], nn.Linear)
    assert isinstance(layer_list[-1], nn.Linear) or isinstance(layer_list[-1], SparseLinear)

    layer_list = layer_list[:-1]

    layer_list.append(magic_layer(sequence1[-1], sequence2[-1]))
    
        
    layer_list.append(nn.ReLU())
    
    output_layer = nn.Linear(20, 1).double() # TODO fix  the number
    output_layer.weight.data = torch.ones(1, 20).double()
    output_layer.bias.data = torch.zeros(1).double()

    layer_list.append(output_layer)

    """
    if isinstance(layer_list[0], SparseLinear):
        sparse = layer_list[0]
        lin = nn.Linear(sparse.w.shape[1], sparse.w.shape[0]).double()
        lin.weight.data = sparse.w.to_dense()
        lin.bias.data = sparse.b
        layer_list[0] = lin
    """
    
    return nn.Sequential(*layer_list).cuda()

def create_c(compnet, inputs):
    assert inputs.shape[0] == 1 # one sample in a batch

    inputs = inputs.reshape(1, -1)

    # reduce and squeeze compnet 
    saturations = eval_one_sample(compnet, inputs)    

    output1 = compnet(inputs)

    target_net = squeeze_network(prune_network(compnet, saturations))

    output2 = target_net(inputs)
    
    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    assert W.shape[0] == 1
    
    c = torch.hstack([b, W.flatten()])

    return c 

def get_subnetwork(net, i):
    # network up to i-th linear layer
    layers = []
    for layer in net:
        layers.append(layer)
        if isinstance(layer, nn.Linear) or isinstance(layer, SparseLinear):
            i -= 1
        if i < 0:
            break
    return nn.Sequential(*layers)
    

def create_upper_bounds(net, inputs):

    # extract the sequential 
    #    net = next(iter(net.children())) NO NEED FOR COMPNET
    assert isinstance(net, nn.Sequential)
    
    saturations = eval_one_sample(net, inputs)

    A_list = [] 
    for i, saturation in enumerate(saturations):
        subnet = get_subnetwork(net, i)
        if i == 0:
            target = subnet
        else:
            target = squeeze_network(prune_network(subnet, saturations[:i]))

        if isinstance(target[-1], nn.Linear):
            W = target[-1].weight.data
            b = target[-1].bias.data
        else:
            W = target[-1].w.to_dense()
            b = target[-1].b
            if b is None:
                b = torch.zeros(W.shape[0])

        # saturation: True ~ U, False ~ S   
        W_lower = W[torch.logical_not(saturation).flatten()]
        b_lower = b[torch.logical_not(saturation).flatten()].reshape(-1, 1)
        W_higher = W[saturation.flatten()]
        b_higher = b[saturation.flatten()].reshape(-1, 1)
        
        W = torch.vstack([W_lower, -1*W_higher])
        b = torch.vstack([b_lower, -1*b_higher])

        A = torch.hstack([b, W])
        
        A_list.append(A)


    return torch.vstack(A_list)

def optimize(c, A_ub, b_ub, A_eq, b_eq, l, u):
    c = c.cpu().numpy()
    A_ub, b_ub = A_ub.cpu().detach().numpy(), b_ub.cpu().numpy()
    A_eq, b_eq = A_eq.cpu().detach().numpy(), b_eq.cpu().numpy()

    
    from scipy.optimize import linprog

    print("optimising", flush=True)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(l, u))
    print(res)

    assert res.success
    
    return res.fun, res.x 


def get_network_dense():

    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    
    return net, net2


def get_network_conv():
#    NETWORK = "mnist_conv_net.pt"
#    MODEL = SmallConvNet
#    LAYERS = 5
#    INPUT_SIZE = (1, 28, 28) 

#    net = load_network(MODEL, NETWORK)
#    net2 = load_network(MODEL, NETWORK)

#    net = to_dense(net, input_size=INPUT_SIZE)
#    net2 = to_dense(net2, input_size=INPUT_SIZE)

    PATH = "tmp/pa_conv_net.pt"
    print("Loading first ...  ", end="", flush=True)
    net = torch.load(PATH)
    print("OK", flush=True)
    print("Loading second ... ", end="", flush=True)
    net2 = torch.load(PATH)
    print("OK", flush=True)
    
    return net, net2

def step(i, inputs, compnet, N, NET_TYPE):
    inputs = inputs.cuda().double().reshape(1, -1)

    #        out1 = net(inputs.to("cuda:1"))
    #        out2 = net2(inputs.to("cuda:1"))
    #        real_error = (out2 - out1).abs().sum().item()
    computed_error = compnet(inputs).item()
        
        
    # min c @ x
    c = -1*create_c(compnet, inputs)
    
    # A_ub @ x <= b_ub
    A_ub = create_upper_bounds(compnet, inputs)
    b_ub = torch.zeros((A_ub.shape[0],), dtype=torch.float64)
        
    # A_eq @ x == b_eq
    A_eq = torch.zeros((1, N+1)).double()
    A_eq[0, 0] = 1.0
    b_eq = torch.zeros((1,)).double()
    b_eq[0] = 1.0                    

    # l <= x <= u 
    l = -0.5
    u = 3.0

    err, x = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u) 
    print("result:", -err)
                
    assert np.isclose(x[0], 1.0)
    
    y = torch.tensor(x[1:], dtype=torch.float64).reshape(1, -1).cuda()
    err_by_net = compnet(y).item()
    
    err_by_sol = (c @ torch.tensor(x, dtype=torch.float64).cuda()).item()
        
    assert np.isclose(-err, err_by_net)
    assert np.isclose(err, err_by_sol)

    with open(f"results/results_{NET_TYPE}4.csv", "a") as f:
        #print(f"{real_error:.6f},{computed_error:.6f},{-err:.6f}", file=f)
        print(f"NaN,{computed_error:.6f},{-err:.6f}", file=f)
    np.save(f"results/{i}_{NET_TYPE}.npy", np.array(x[1:], dtype=np.float64))
    

def main(): 

    BATCH_SIZE=1    
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 
    NET_TYPE = "conv"

    if NET_TYPE == "dense":
        net, net2 = get_network_dense()
    else:
        net, net2 = get_network_conv()
        

    compnet = create_comparing_network(net, net2)
    
    print(compnet)

    del net
    del net2
    
    #net = net.to("cuda:1")
    #net2 = lower_precision(net2).to("cuda:1")
    
    data = create_dataset(train=False, batch_size=BATCH_SIZE)

    i = 0    
    for inputs, labels in tqdm.tqdm(data):
        if i >= 4000 and i < 5000:
            step(i, inputs, compnet,
                 N, NET_TYPE)
        i += 1
        del inputs
        del labels
        gc.collect()
        torch.cuda.empty_cache()
        
        
if __name__ == "__main__":

    # test_squeeze() # 1.
    #test_compnet() # 2.
    #test_squeezed_compnet() # 3.

    main()
