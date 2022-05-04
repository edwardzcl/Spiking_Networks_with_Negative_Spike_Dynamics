import torch
import torch.nn as nn
import numpy as np
from quantization import quantize_finetune
import datasets
from validation import validate_snn, validate_ann
import argparse
from ann_parser import parse_ann_model
from build_network import get_net_by_name
from build_criterion import get_criterion_by_name
import json
from snn_transformer import SNNTransformer, V1ChipQuantSNNTransformer
from models.relu_threshold_shift import ReLUThresholdShift
import os
from collections import OrderedDict


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('net_name', help='network name to train')
    parser.add_argument('--hardware', default='v1', choices=['cpu', 'v1'], help="The target hardware")
    parser.add_argument('--ann_weight', type=str,
                        help='the location of the trained weights')
    parser.add_argument('--dataset', default='CIFAR10',
                        type=str, help='the location of the dataset')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for dataset')
    parser.add_argument('--save_file', default="./out_snn.pth",
                        type=str, help='the output location of the transferred weights')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--timesteps', '-T', default=16, type=int)
    parser.add_argument('--reset_mode', default='subtraction',
                        type=str, choices=['zero', 'subtraction'])
    parser.add_argument('--weight_bitwidth', default=4,
                        type=int, help='weight quantization bitwidth')
    parser.add_argument('--criterion', default='cross_entropy',
                        type=str, help='name of criterion used in finetune')
    parser.add_argument('--finetune_lr', default=0.005,
                        type=float, help='finetune learning rate')
    parser.add_argument('--quantization_channel_wise', '-qcw',
                        action='store_true', help='quantize in each channel')
    parser.add_argument('--finetune_epochs', '-e', default=5,
                        type=int, help='finetune epochs')
    parser.add_argument('--finetune_wd', default=5e-4,
                        type=float, help='finetune weight decay')
    parser.add_argument('--finetune_momentum', default=0.9,
                        type=float, help='finetune momentum')
    parser.add_argument('--statistics_iters', default=30, type=int,
                        help='iterations for gather activation statistics')
    parser.add_argument('--relu_threshold', default=100, type=float)
    parser.add_argument('--bias_map', default='weight', choices=['weight','leakage'])
    # parser.add_argument('--relu_shift', default='V/2T',type=str, choices=['0', 'V/2T'])
    parser.add_argument('--device', default='0,1,2,3', type=str, help='choose cuda devices')

    args = parser.parse_args()
    args.activation_bitwidth = np.log2(args.timesteps)
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Preparing the dataset
    train_loader, test_loader = datasets.get_dataset(args)

    # Build Model
    net = get_net_by_name(args.net_name)
    print(net)
    # Load weight
    if args.ann_weight:
        if args.ann_weight != "not_load":
            params=torch.load(args.ann_weight,map_location=device)
            if "net" in params.keys():
                params=params["net"]
            new_params=OrderedDict()
            for k,v in params.items():
                if "module." == k[:7]:
                    new_params[k[7:]]=v
                else:
                    new_params[k]=v            
            net.load_state_dict(new_params)
    net.to(device)

    # Validate
    criterion = get_criterion_by_name(args.criterion)
    net_top1, net_loss = validate_ann(net, test_loader, device, criterion)

    # replace the Activation functions with ReLUThresholdShift
    for name,module in net.named_modules():
        if type(module) in [nn.ReLU,nn.PReLU,nn.ReLU6]:
            new_activation=ReLUThresholdShift(args.relu_threshold,None,args.timesteps)
            module.new_activation=new_activation
            module.forward=new_activation.forward

    # Quantization and Finetune
    qnet = quantize_finetune(net, train_loader, criterion, device, args)
    qnet_top1, qnet_loss = validate_ann(qnet, test_loader, device, criterion)

    # Transform
    if args.hardware == 'v1':
        transformer = V1ChipQuantSNNTransformer(args, qnet, device)
    elif args.hardware == 'cpu':
        transformer = SNNTransformer(args, qnet, device)
    else:
        raise NotImplementedError()
    snn=transformer.generate_snn(train_loader, args.statistics_iters)
    
    # Test the results
    snn_top1, snn_loss, firing_ratios = validate_snn(
        snn, test_loader, device, criterion, args.timesteps,)

    # Save the SNN
    torch.save(snn, args.save_file)
    torch.save(snn.state_dict(), args.save_file+'.weight')
    print("Save the SNN in {}".format(args.save_file))

    # Save the snn info
    snn_info = {
        'qnet_top1': float(qnet_top1),
        'snn_top1': float(snn_top1),
        'mean_firing_ratio': float(firing_ratios.mean()),
        'firing_ratios': [float(_) for _ in firing_ratios],
    }
    with open(args.save_file+'.json', 'w') as f:
        json.dump(snn_info, f)
