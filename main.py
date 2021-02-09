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
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d):
            new_op = QuantConv(child)
            setattr(model, child_name, new_op)
        elif isinstance(child, torch.nn.Linear):
            new_op = QuantLinear(child)
            setattr(model, child_name, new_op)
        elif isinstance(child, (torch.nn.ReLU, torch.nn.ReLU6)):
            activation = torch.nn.ReLU()
            # new_op = QuantActivations(child)
            new_op = QuantActivations(activation)
            setattr(model, child_name, new_op)
        elif isinstance(child, torch.nn.BatchNorm2d):
            new_op = QuantBN(child)
            setattr(model, child_name, new_op)
        else:
            replace_quant_ops(child)

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

def register_bn_params_to_prev_layers(model):
    
    layer_bn_pairs = get_input_sequences(model)

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
    parser.add_argument('--ptq',                        help='Post Training Quantization techniques to run', choices=['cle'])
    
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

def blockwise_equalization(model):
    # Following setup gives best result.
    cross_layer_equalization(torch.nn.Sequential(model.features[0], model.features[1].conv, model.features[2].conv))
    for module in model.features[3:]:
        # Equalizing Residual connetcion wise - See 5.1.1. Cross-layer equalization in the paper
        if isinstance(module, InvertedResidual):
            cross_layer_equalization(module)

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
    eps = 1e-4
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


def main():
    args = arguments()
    seed(args)
    model = load_model(pretrained = True)

    val_dataloader = get_loaders(args)

    eval_func = model_eval(val_dataloader, batch_size=args.batch_size)

    model.cuda()
    
    register_bn_params_to_prev_layers(model)

    def bn_fold(module):
        if isinstance(module, (QuantConv, QuantBN)):
            module.batchnorm_folding()

    def run_calibration(calibration):
        def estimate_range(module):
            if isinstance(module, Quantizers):
                module.estimate_range(flag = calibration)
        return estimate_range

    def set_quant_mode(quantized):
        def set_precision_mode(module):
            if isinstance(module, Quantizers):
                module.set_quantize(quantized)
                module.estimate_range(flag = False)
        return set_precision_mode

    replace_quant_ops(model)
    model.apply(bn_fold)
    if 'cle' in args.ptq:
        blockwise_equalization(model)

    model.apply(run_calibration(calibration = True))
    eval_func(model, (1024./args.batch_size, True))
    model.apply(set_quant_mode(quantized = True))


    '''
    Fuse Conv / activation in one operation etc : Conv -> ReLU 
    '''
    layers = []
    def hook(name):
        def func(m, i, o):
            layers.append(m)
        return func
    handlers = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv, QuantActivations)):
            handlers.append(module.register_forward_hook(hook(name)))
    dummy = torch.randn([1,3,224,224]).cuda()
    model.cuda()
    model(dummy)
    for handle in handlers:
        handle.remove()
    for idx in range(len(layers)-1):
        prev, cur = layers[idx], layers[idx+1]
        if isinstance(prev, QuantConv) and isinstance(cur, QuantActivations):
            prev.act_quantizer.is_quantize = False
    eval_func(model, (9999999, True))

if __name__ == '__main__':
    main()
