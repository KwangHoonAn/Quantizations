from MobileNetV2 import mobilenet_v2, InvertedResidual
from quantops import *
import random
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

def replace_quant_ops(model):
    prev_module = None
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d):
            new_op = QuantConv(child)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, torch.nn.Linear):
            new_op = QuantLinear(child)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, (torch.nn.ReLU, torch.nn.ReLU6)):
            # prev_module.activation_function = child
            prev_module.activation_function = torch.nn.ReLU()
            setattr(model, child_name, PassThroughOp())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, PassThroughOp())
        else:
            replace_quant_ops(child)

def replace_quant_to_brecq_quant(model):
    for child_name, child in model.named_children():
        if isinstance(child, QuantConv):
            continue
        elif isinstance(child, QuantLinear):
            continue
        elif isinstance(child, QuantActivations):
            activation = child.activation_func
            new_op = LSQActivations(activation, child.act_quantizer.scale.data.cpu().numpy())
            setattr(model, child_name, new_op)
        else:
            replace_quant_to_brecq_quant(child)

def get_input_sequences(model):
    layer_bn_pairs = []
    def hook(name):
        def func(m, i, o):
            if m in (torch.nn.Conv2d, torch.nn.Linear):
                if not layer_bn_pairs:
                    layer_bn_pairs.append((m, name))
                else:
                    if layer_bn_pairs[-1][0] in (torch.nn.Conv2d, torch.nn.Linear):
                        layer_bn_pairs.pop()
            else:
                layer_bn_pairs.append((m, name))
        return func

    handlers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            handlers.append(module.register_forward_hook(hook(name)))
    dummy = torch.randn([1,3,224,224]).cuda()
    model(dummy)
    for handle in handlers:
        handle.remove()
    return layer_bn_pairs

def register_bn_params_to_prev_layers(model, layer_bn_pairs):
    

    idx = 0
    while idx + 1 < len(layer_bn_pairs):
        conv, bn = layer_bn_pairs[idx], layer_bn_pairs[idx + 1]
        conv, conv_name = conv
        bn, bn_name = bn
        bn_state_dict = bn.state_dict()
        conv.register_buffer('eps', torch.tensor(bn.eps))
        conv.register_buffer('gamma', bn_state_dict['weight'].detach())
        conv.register_buffer('beta', bn_state_dict['bias'].detach())
        conv.register_buffer('mu', bn_state_dict['running_mean'].detach())
        conv.register_buffer('var', bn_state_dict['running_var'].detach())
        idx += 2



def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

def model_eval(data_loader, batch_size=64):
    def eval_func(model, arguments):
        top1_acc = 0.0
        total_num = 0
        idx = 0
        iterations , use_cuda = arguments[0], arguments[1]
        if use_cuda:
            model.cuda()
        for sample, label in tqdm(data_loader):
            total_num += sample.size()[0]
            if use_cuda:
                sample = sample.cuda()
                label = label.cuda()
            logits = model(sample)
            pred = torch.argmax(logits, dim = 1)
            correct = sum(torch.eq(pred, label)).cpu().numpy()
            top1_acc += correct
            idx += 1
            if idx > iterations:
                break
        avg_acc = top1_acc * 100. / total_num
        print("Top 1 ACC : {:0.2f}".format(avg_acc))
        return avg_acc
    return eval_func


def seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def load_model(pretrained = True):
    model = mobilenet_v2(pretrained)
    model.eval()
    return model


def arguments():
    parser = argparse.ArgumentParser(description='Cross Layer Equalization in MV2')

    parser.add_argument('--images-dir',                 help='Imagenet eval image', default='./ILSVRC2012_PyTorch/', type=str)
    parser.add_argument('--seed',                       help='Seed number for reproducibility', type = int, default=0)
    parser.add_argument('--ptq',                        help='Post Training Quantization techniques to run - Select from CLS / HBA / Bias correction', nargs='+')
    
    parser.add_argument('--batch-size',                 help='Data batch size for a model', type = int, default=64)
    parser.add_argument('--num-workers',                help='Number of workers to run data loader in parallel', type = int, default=16)

    args = parser.parse_args()
    return args

def get_loaders(args):
    image_size = 224
    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : args.num_workers}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
            transforms.Resize(image_size + 24),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])
    val_data = datasets.ImageFolder(args.images_dir + '/val/', val_transforms)
    val_dataloader = DataLoader(val_data, args.batch_size, shuffle = False, pin_memory = True, **data_loader_kwargs)
    return val_dataloader

def blockwise_equalization(args, model):
    # Following setup gives best result.
    conv_layers = cross_layer_equalization(torch.nn.Sequential(model.features[0], model.features[1].conv, model.features[2].conv))
    for module in model.features[3:]:
        # Equalizing Residual connetcion wise - See 5.1.1. Cross-layer equalization in the paper
        if isinstance(module, InvertedResidual):
            conv_layers = cross_layer_equalization(module)

def get_conv_layers(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, QuantConv):
            conv_layers.append(module)
    return conv_layers

def cross_layer_equalization(model):
    conv_layers = get_conv_layers(model)
    '''
    Perform Cross Layer Scaling :
    Iterate modules until scale value is converged up to 1e-4 magnitude
    '''
    S_history = dict()
    eps = 1e-6
    converged = [False] * (len(conv_layers)-1)
    with torch.no_grad(): 
        while not np.all(converged):
            for idx in range(1, len(conv_layers)):

                prev, curr = conv_layers[idx-1].conv, conv_layers[idx].conv
                out_channel_prev, in_channel_curr = prev.weight.size()[0], curr.weight.size()[1]

                '''
                prev : [Out_channel, In_channel, H, W]
                curr : [Out_channel, In_channel, H, W]
                For prev layer, we need to obtain a range of 'output channel'
                For curr layer, we need to obtain a range of 'input channel'
                '''
                range_1 = 2.*torch.abs(prev.weight).max(axis = 1)[0].max(axis = 1)[0].max(axis = 1)[0]
                range_2 = 2.*torch.abs(curr.weight).max(axis = 0)[0].max(axis = -1)[0].max(axis = -1)[0]

                S = torch.sqrt(range_1 * range_2) / range_2

                if idx in S_history:
                    prev_s = S_history[idx]
                    if np.all(np.isclose(S.cpu().numpy(), prev_s.cpu().numpy(), atol = eps)):
                        converged[idx-1] = True
                        continue
                    else:
                        converged[idx-1] = False
                s_dim = S.size()[0]
                prev.weight.data.copy_( prev.weight.data /  S.view(s_dim, 1, 1, 1))
                prev.bias.data.copy_(prev.bias.data / S)

                # Generic Conv layer
                if in_channel_curr == out_channel_prev: 
                    curr.weight.data.copy_( curr.weight.data *  S.view(1, s_dim, 1, 1) )
                else:
                    # Depthwise Convolution
                    curr.weight.data.copy_( curr.weight.data * S.view(s_dim, 1, 1, 1) )
                S_history[idx] = S
    return conv_layers

'''
def high_bias_absorbing(conv_layers):
    for idx in range(1, len(conv_layers)):
        conv1, conv2 = conv_layers[idx-1].conv, conv_layers[idx].conv
        if not conv_layers[idx-1].activation_function:
            continue
        gamma, beta = conv1.gamma.detach(), conv1.beta.detach()
        c = (beta - 3 * gamma).clamp_(min = 0)
        conv1.bias.data.copy_(conv1.bias.data - c)
        if conv2.weight.size()[1] == 1:
            w_mul = conv2.weight.sum(dim = [1,2,3]) * c
            conv2.bias.data.copy_(w_mul + conv2.bias.data)
        else:
            w_mul = conv2.weight.sum(dim = [2,3]).mv(c)
            conv2.bias.data.copy_(w_mul + conv2.bias.data)
'''

def set_quant_mode(quantized):
    def set_precision_mode(module):
        if isinstance(module, (Quantizers, LSQActivations)):
            module.set_quantize(quantized)
            module.estimate_range(flag = False)
    return set_precision_mode
'''
class DataSaverHook:
'''
Code borrowed from 
https://github.com/yhhhli/BRECQ/blob/main/quant/data_utils.py
'''
    def __init__(self, store_input = False, store_output = False, stop_forward = False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_inoput:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException

class GetInpOut:
    def __init__(self, model):
        self.model = model
    def __call__(self, x, layer):
        self.model.eval()

        handler = layer.register_forward_hook(self.data_saver)
        handler.remove()


def empirical_bias_correction(args, model, eval_func):
    import copy
    model_q = copy.deepcopy(model)
    fp_inout = GetInpOut(model)
    q_inout = GetInpOut(model_q)
    for m, m_q in zip(model.modules(), model_q.modules()):
        if isinstance(m, QuantConv):
            m_q.turn_preactivation_on()
            m_q.weight_quantizer.set_quantize(True)
            m_q.act_quantizer.set_quantize(False)
            e_x_fp32 = model(x)
            e_x_int8 = model_q(x)
            m_q.weight_quantizer.set_quantize(False)
    exit(1)
'''
def main():
    args = arguments()
    seed(args)
    model = load_model(pretrained = True)

    val_dataloader = get_loaders(args)

    eval_func = model_eval(val_dataloader, batch_size=args.batch_size)

    model.cuda()

    layer_bn_pairs = get_input_sequences(model)
    register_bn_params_to_prev_layers(model, layer_bn_pairs)

    def bn_fold(module):
        if isinstance(module, (QuantConv)):
            module.batchnorm_folding()

    def run_calibration(calibration):
        def estimate_range(module):
            if isinstance(module, Quantizers):
                module.estimate_range(flag = calibration)
        return estimate_range

    replace_quant_ops(model)
    model.apply(bn_fold)

    if 'cls' in args.ptq:
        blockwise_equalization(args, model)
    if 'bias_correction' in args.ptq:
        empirical_bias_correction(args, model, eval_func)

    model.apply(run_calibration(calibration = True))
    eval_func(model, (1024./args.batch_size, True))

    # replace_quant_to_brecq_quant(model)
    model.apply(set_quant_mode(quantized = True))


    eval_func(model, (9999999, True))

if __name__ == '__main__':
    main()
