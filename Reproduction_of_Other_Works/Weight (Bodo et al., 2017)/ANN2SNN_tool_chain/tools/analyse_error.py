import sys
sys.path.insert(0,'.')
import torch
import torch.nn as nn
import numpy as np
from quantization import quantize_finetune
import datasets
from validation import validate_snn,validate_ann
import argparse
from ann_parser import parse_ann_model
from build_network import get_net_by_name
from build_criterion import get_criterion_by_name
import json
from snn_transformer import SNNTransformer,V1ChipQuantSNNTransformer,SNNTransformer
import copy
import ann_parser
import spike_layers
from spike_tensor import SpikeTensor
import torch.nn.functional as F

def compare_snn_ann(snn,ann,test_loader,device,criterion,timesteps):
    # switch to evaluate mode
    snn.eval()
    ann.eval()

    ann_inputs=[]
    ann_outputs=[]
    ann_weights=[]
    def ann_hook(m,i,o):
        ann_inputs.append(i[0].data)
        ann_outputs.append(o.data)
        ann_weights.append(m.weight)
    snn_inputs=[]
    snn_outputs=[]
    snn_spikes=[]
    snn_weights=[]
    def snn_hook(m,i,o):
        snn_inputs.append(i[0].to_float())
        snn_spikes.append(o.data.view(args.timesteps,-1,*o.data.size()[1:]))
        snn_outputs.append(o.to_float())
        snn_weights.append(m.weight)
    # wrap the layers in snn and ann
    for name,m in ann.named_modules():
        if isinstance(m, spike_layers.SpikeConv2d):
            m.register_forward_hook(ann_hook)
    for name,m in snn.named_modules():
        if isinstance(m, spike_layers.SpikeConv2d):
            m.register_forward_hook(snn_hook)
    
    total_firing_ratios=[]
    cnt=0
    with torch.no_grad():
        for data_test in test_loader:
            print("Test at: ",cnt)
            if cnt>0:
                break
            cnt+=1
            data, target = data_test
            data = data.to(device)
            ann_out=ann(data)
            replica_data = torch.cat([data for _ in range(timesteps)], 0)
            data = SpikeTensor(replica_data, timesteps, scale_factor=1)
            snn_out=snn(data)
            
        for ai,a,aw,sw,si,s,ss in zip(ann_inputs,ann_outputs,ann_weights,snn_weights,snn_inputs,snn_outputs,snn_spikes):
            print(a.shape,s.shape)
            print("===ANN Inputs===\n",ai[0,:3,:6,:6])
            print("===ANN Outputs===\n",F.relu(a[0,:3,:6,:6]))
            print("===ANN Weights===\n",aw[:3,:3])
            print("===SNN Weights===\n",sw[:3,:3])
            print("===SNN Inputs===\n",si[0,:3,:6,:6])
            print("===SNN Outputs(to float)===\n",s[0,:3,:6,:6])
            print("===SNN Spike===\n",torch.sum(ss,0)[0,:3,:6,:6])

    for layer in net.modules():
        if hasattr(layer,'mem_potential'):
            layer.mem_potential=None


if __name__=='__main__':
    # parse args
    parser=argparse.ArgumentParser()
    parser.add_argument('net_name',help='network name to train')
    parser.add_argument('--hardware',default='v1',choices=['cpu','v1'])
    parser.add_argument('--ann_weight',type=str,help='the location of the trained weights')
    parser.add_argument('--dataset',default='CIFAR10',type=str,help='the location of the dataset')
    parser.add_argument('--num_workers', default=4, type=int,help='number of workers for dataset')
    parser.add_argument('--save_file',default="./out_snn.pth",type=str,help='the output location of the transferred weights')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--timesteps','-T', default=16, type=int)
    parser.add_argument('--reset_mode', default='subtraction', type=str, choices=['zero', 'subtraction'])
    parser.add_argument('--weight_bitwidth', default=4, type=int,help='weight quantization bitwidth')
    parser.add_argument('--criterion', default='cross_entropy', type=str,help='name of criterion used in finetune')
    parser.add_argument('--finetune_lr', default=0.005, type=float,help='finetune learning rate')
    parser.add_argument('--quantization_channel_wise','-qcw', action='store_true',help='quantize in each channel')
    parser.add_argument('--finetune_epochs','-e', default=5, type=int,help='finetune epochs')
    parser.add_argument('--finetune_wd', default=5e-4, type=float,help='finetune weight decay')
    parser.add_argument('--finetune_momentum', default=0.9, type=float,help='finetune momentum')
    parser.add_argument('--statistics_iters', default=30, type=int,help='iterations for gather activation statistics')
    parser.add_argument('--compare', action='store_true')

    args=parser.parse_args()
    args.activation_bitwidth=np.log2(args.timesteps)
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # Preparing the dataset
    train_loader,test_loader=datasets.get_dataset(args)

    # Build Model
    net=get_net_by_name(args.net_name)
    print(net)
    # Load weight
    if args.ann_weight:
        net.load_state_dict(torch.load(args.ann_weight),strict=False)
    net.to(device)

    # Validate
    ann=ann_parser.parse_ann_model(net,[torch.ones([1,3,32,32]).to(device)])
    ann.to(device)
    criterion=get_criterion_by_name(args.criterion)
    net_top1,net_loss=validate_ann(ann,test_loader,device,criterion)

    # # Weight quantization
    qnet=quantize_finetune(ann,train_loader,criterion,device,args)
    qnet_top1,qnet_loss=validate_ann(qnet,test_loader,device,criterion)

    print("========= TEST V1 =========")
    transformer=V1ChipQuantSNNTransformer(args,qnet,device)
    transformer.inference_get_status(train_loader,args.statistics_iters)
    snnv1=transformer.generate_snn()
    snnv1_top1,snn_loss,firing_ratios=validate_snn(snnv1,test_loader,device,criterion,args.timesteps)
    if args.compare:
        compare_snn_ann(snnv1,ann,test_loader,device,criterion,args.timesteps)
    print("========= TEST =========")
    transformer=SNNTransformer(args,qnet,device)
    transformer.inference_get_status(train_loader,args.statistics_iters)
    snn=transformer.generate_snn()
    # Test the results
    snn_top1,snn_loss,firing_ratios=validate_snn(snn,test_loader,device,criterion,args.timesteps)
    if args.compare:
        compare_snn_ann(snn,qnet,test_loader,device,criterion,args.timesteps)

