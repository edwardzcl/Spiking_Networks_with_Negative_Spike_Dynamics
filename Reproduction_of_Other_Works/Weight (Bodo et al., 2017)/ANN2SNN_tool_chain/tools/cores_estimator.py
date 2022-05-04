import sys
sys.path.insert(0,'./')
import torch
import torch.nn as nn
import numpy as np
from quantization import quantize_finetune
import datasets
from validation import validate_snn,validate_ann
import argparse
from build_network import *
import spike_layers
from ann_parser import parse_ann_model

input_ports=1024
output_ports=1024


def conv2d_forward(self,x,out):
    # out=super().forward(x)
    x=x[0]
    in_window_size=input_ports/self.in_channels
    in_width_number=max(1,(x.size(2)-self.kernel_size[0]+1)/max(1,in_window_size**0.5-self.kernel_size[0]+1))
    in_use_cores=int(in_width_number**2)
    rescaled_out_window_size=max(1,(np.sqrt(in_window_size)-(self.kernel_size[0]-1))/self.stride[0])
    out_window_size=(out[0].size(2)/in_width_number)**2*self.out_channels
    out_use_cores=int(max(1,np.ceil(out_window_size/output_ports)))
    self.estimated_cores=out_use_cores*in_use_cores
    self.info=(x.size(),out[0].size(),in_window_size,in_width_number,in_use_cores,out_window_size,out_use_cores)
    self.input_used=self.in_channels*self.kernel_size[0]*self.kernel_size[1]
    

def pool2d_forward(self,x,out):
    # out=super().forward(x)
    x=x[0]
    in_use_cores=x.size(1)*x.size(2)*x.size(3)/input_ports
    self.estimated_cores=in_use_cores
    self.info=(x.size(),out[0].size(),in_use_cores)
    self.input_used=self.kernel_size[0]*self.kernel_size[1]


if __name__=='__main__':
    # parse args
    parser=argparse.ArgumentParser()
    parser.add_argument('net_name',help='network name to train')
    parser.add_argument('input_shape',  type=str, help="input shape, such as 1,3,32,32")
    parser.add_argument('--ann_weight',type=str,help='the location of the trained weights')
    parser.add_argument('--num_workers', default=4, type=int,help='number of workers for dataset')
    
    parser.add_argument('--reset_mode', default='subtraction', type=str, choices=['zero', 'subtraction'])

    args=parser.parse_args()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    net=get_net_by_name(args.net_name)
    # Load weight
    if args.ann_weight:
        net.load_state_dict(torch.load(args.ann_weight),strict=False)
    # Transform to DAG
    inp_data=torch.ones([int(_) for _ in args.input_shape.split(',')])
    dag_net=parse_ann_model(net,[inp_data])
    for name,layer in dag_net.named_modules():
        if isinstance(layer,spike_layers.SpikeConv2d):
            layer.register_forward_hook(conv2d_forward)
        if isinstance(layer,spike_layers.SpikeAvgPool2d):
            layer.register_forward_hook(pool2d_forward)
    dag_net(inp_data)
    # net.to(device)
    input_used_avg=0
    tot_estimated_cored=0
    for name,layer in dag_net.named_modules():
        if hasattr(layer,"estimated_cores"):
            tot_estimated_cored+=layer.estimated_cores
            input_used =layer.input_used if layer.input_used<1024 else 1024
            print(f"{name},{input_used/1024},{int(layer.estimated_cores)}")
            input_used_avg+=input_used/1024*int(layer.estimated_cores)
    print("tot",tot_estimated_cored)
    print("input used avg",input_used_avg/tot_estimated_cored)
