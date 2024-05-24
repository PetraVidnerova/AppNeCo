import tqdm
import torch
import torch.nn as nn 

from network import SmallDenseNet, SmallConvNet
from dataset import create_dataset
from torch_conv_layer_to_fully_connected import torch_conv_layer_to_affine

RANDOM_DATO = None

def mac(w, x):
    # torch.sum(w*x, dim=1)
    #print(w.shape)
    #print(x.shape)
    result = torch.zeros(w.shape[0]).double()
    for i in range(w.shape[1]):
        result += w[:,i]*x[i]
    return result

def compute_input_ai_bi():
    test_data = create_dataset(train=False, batch_size=BATCH_SIZE)

    a, b = [], []

    i = 0 
    for x, _  in tqdm.tqdm(test_data):
        x = x.double() 
        if i == 7:
            global RANDOM_DATO
            RANDOM_DATO = x[0].unsqueeze(0)
        x = x.reshape(-1, N)
        print(x.shape)
        amin, _ = torch.min(x, dim=0)
        bmax, _  = torch.max(x, dim=0)
        a.append(amin)
        b.append(bmax)
        i += 1
        
    a, _ = torch.min(torch.vstack(a), dim=0)
    b, _ = torch.max(torch.vstack(b), dim=0)

    return a, b

def compute_output_ai_bi(a, b, w, bias, relu=True):

    R = nn.ReLU()
    
    w_neg = -1*R(-1*w)
    w_pos = R(w)
  
    print(bias.shape)
    print(w.shape)
    print(w_neg.shape)
    print(w_pos.shape)
    
    a_new = bias + mac(w_neg, b) + mac(w_pos, a)
    b_new = bias + mac(w_neg, a) + mac(w_pos, b)

    if relu:
        return R(a_new), R(b_new)
    else:
        return a_new, b_new

def compute_bias_shift(a, b, w, bias):
    bias = bias + mac(w, a)
    a, b = a-a, b-a
    return bias, a, b 

def load_network():
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK))
    net.eval()
    return  net.double()

def get_layer(net, layer):

#    weight  ... i*2
#    bias ... i*2 + 1
    x = torch.zeros(INPUT_SIZE).unsqueeze(0).double()
    input_size = INPUT_SIZE
    layer_pointer = None
    
    for child in net.children():
        assert isinstance(child, nn.Sequential)
        i = 0 # count only conv and fc 
        for l in child.children():
            print(l.__class__.__name__)
            if isinstance(l, nn.Dropout):
                continue
            print("evaluating")
            x = l(x)
            if isinstance(l, nn.Flatten):
                continue 
            if isinstance(l, nn.ReLU):
                continue
            if i == layer:
                layer_pointer = l
                break
            input_size = x.shape 
            i += 1
        
#    exit()

#    for i, p in enumerate(net.parameters()):
#        print(i, ":", p.shape)
#        if i == layer*2:
#            weight = p
#        elif i == layer*2+1:
#            bias = p 

    if type(layer_pointer) == nn.Conv2d:
        #print(weight.shape)
        fc = torch_conv_layer_to_affine(layer_pointer, input_size[-2:])

#        print(RANDOM_DATO.shape)
        
#        y1 = layer_pointer(RANDOM_DATO)

#       y2 = fc(RANDOM_DATO.reshape(1, -1)).reshape(1, -1, 28, 28)

        
#        print(y1.shape)
#       print(y2.shape)

#       print(torch.all(torch.isclose(y1, y2)))
#        
        return "conv", fc.weight, fc.bias, input_size, layer_pointer

    if type(layer_pointer) == nn.Linear:
        return "lin", layer_pointer.weight, layer_pointer.bias, input_size, layer_pointer

    if type(layer_pointer) == nn.MaxPool2d:
        return "pool", None, None, input_size, layer_pointer

    raise ValueError(f"{layer} is wierd")

#    print(layer)
#    return weight, bias 

def get_delta(w, bias):
    print(w)
    print(w.dtype)

    w_hat = w.half()
    print(w_hat.dtype)
    w_hat = w_hat.double()

    delta = w_hat - w
    print(delta.dtype)
    print(delta)

    bias_hat = bias.half()
    bias_hat = bias_hat.double()
    delta_bias = bias_hat - bias
    
    return delta, delta_bias 


def compute_output_alpha_beta(alpha, beta, b, delta_weight, delta_bias,
                              weight, bias, relu=True):
    weight_hat = weight.half().double()
    bias_hat = bias.half().double()

    
    R = nn.ReLU()
    
    delta_neg = -1*R(-1*delta_weight)
    delta_pos = R(delta_weight)

    alpha_new = mac(delta_neg, b)#torch.sum(delta_neg*b, dim=1)
    beta_new = mac(delta_pos, b) #torch.sum(delta_pos*b, dim=1)

    weight_hat_neg = -1*R(-1*weight_hat)
    weight_hat_pos = R(weight_hat)

    alpha_new += mac(weight_hat_pos, alpha) # torch.sum(weight_hat_pos*alpha, dim=1)
    alpha_new += mac(weight_hat_neg, beta) #torch.sum(weight_hat_neg*beta, dim=1)

    beta_new += mac(weight_hat_neg, alpha) #torch.sum(weight_hat_neg*alpha, dim=1)
    beta_new += mac(weight_hat_pos, beta) #torch.sum(weight_hat_pos*beta, dim=1)

    if relu:
        return (
            torch.minimum(torch.tensor(0), alpha_new + delta_bias),
            R(beta_new + delta_bias)
        )
    else:
        return alpha_new, beta_new
    
def compute_pool_alpha_beta(alpha, beta, shape, max_):
    alpha_new = alpha.reshape(*shape)
    beta_new = beta.reshape(*shape)

    return -1*max_(-1*alpha_new).flatten(), max_(beta_new).flatten()
    
    
def compute_pool_ai_bi(a, b, shape, max_):
    a_new = a.reshape(*shape)
    b_new = b.reshape(*shape)

    return max_(a_new).flatten(), max_(b_new).flatten()
    
def calculate_output_intervals(): 
    with torch.no_grad():
        a, b = compute_input_ai_bi()
        alpha = torch.zeros(N).double()
        beta = torch.zeros(N).double()  
        print(a)
        print(b)
        
        net = load_network()

        print(net)


        
        for layer in range(LAYERS):
        
            type_, weight, bias, input_shape, layer_pointer = get_layer(net, layer)

            if type_ == "pool":
                alpha, beta = compute_pool_alpha_beta(alpha, beta, input_shape, layer_pointer)
                a, b = compute_pool_ai_bi(a, b, input_shape, layer_pointer)

            else:    
                bias, a, b = compute_bias_shift(a, b, weight, bias) 
                
                print(bias.shape, a.shape, b.shape)
                print(bias, a, b)

                delta, delta_bias = get_delta(weight, bias)
                
                alpha, beta = compute_output_alpha_beta(alpha, beta, b, delta, delta_bias,
                                                        weight, bias,
                                                        relu=layer!=(LAYERS-1)) 
            
                a, b = compute_output_ai_bi(a, b, weight, bias, relu=layer!=(LAYERS-1))
        
                #            if layer != LAYERS-1:
                #                print(a)
                #                assert torch.all(a==0)
                del weight
                del bias
            assert torch.all(a<=b)
            assert torch.all(alpha<=0)
            assert torch.all(beta>=0)
                
            
        print("-----------------------")
        print(a)
        print(b)

        print(alpha)
        print(beta)
        print("ok")


    
    
    
if __name__ == "__main__":

            
    BATCH_SIZE=1024
    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 3 
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    calculate_output_intervals() 

    NETWORK="mnist_conv_net.pt"
    MODEL = SmallConvNet
    LAYERS = 5
    calculate_output_intervals() 
