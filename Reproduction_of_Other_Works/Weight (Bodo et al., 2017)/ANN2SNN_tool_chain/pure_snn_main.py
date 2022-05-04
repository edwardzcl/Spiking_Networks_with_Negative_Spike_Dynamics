import torch
import torch.nn as nn
import torch.nn.functional as F

from spike_dag import SpikeDAGModule, DAGViewOp
from spike_layers import SpikeConv2d, SpikeLinear
from spike_tensor import SpikeTensor
from tdlayers import tdLayer

import os
import argparse
from pure_snn_config import SpikeNN, Vths, View, in_data

quantized_layers = []
def quantize_tensor(layer,tensor,bitwidth,channel_level=False):
    if channel_level:
        _max = tensor.abs().view(tensor.size(0),-1).max(1)[0]
    else:
        _max = tensor.abs().max()
    scale = (2 ** (bitwidth - 1) - 1) / _max
    if tensor.dim() == 4:
        scale = scale.view(-1, 1, 1, 1)
    else:
        scale = scale.view(-1, 1)
    new_tensor = torch.round(scale * tensor)
    return new_tensor, scale

def init_quantize_net(net,weight_bitwidth):
    first = 1
    for name,m in net.named_modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            if first == 1:       
                print("jump the first layer:", name, ",",m)
                first = 0     
                continue
        # if isinstance(m,SpikeConv2d):
            if hasattr(m.weight,'weight_back'):
                continue
            #print(name)
            quantized_layers.append(m)
            m.weight.weight_back=m.weight.data.clone()
            if m.bias is not None:
                raise NotImplementedError

def quantize_layers_final(bitwidth, rescale, Vths):
    #print(quantized_layers)
    counter = 0
    for layer in quantized_layers:
        with torch.no_grad():
            quantized_w, scale = quantize_tensor(layer,layer.weight.weight_back,bitwidth,False)

            # 自动跳过第一层的Vth设置
            counter += 1
            
            scale_Vths = scale.cpu()
            #+-误差
            Vths[counter] = torch.round(scale_Vths * Vths[counter])
            layer.weight= nn.Parameter(quantized_w/scale) if rescale else nn.Parameter(quantized_w)
    return Vths

def weightsdistribute(model):
    print("================show every layer's weights distribute================")
    for key, value in model.named_parameters():
        print(key)
        unique, count = torch.unique(value.detach(), sorted=True, return_counts= True)
        print(unique,":", unique.shape)

def snn_set_Vth(net, Vths):
    print("setting snn Vth now")
    #net=copy.deepcopy(raw_net)
    net.eval()

    for layer_i,(layer_name,layer) in enumerate(net.named_modules()):
        if layer_i == 0:
            continue
        print(layer_i - 1)
        print(layer)
        layer.Vthr=Vths[layer_i - 1]
        print("layer.Vthr")
        print(layer.Vthr)
        print()

    print(f"SNN Vth setting Finished")
    return net


def gen_snn(net_trans, View):
    device = torch.device("cpu")

    spike_layers = []
    spike_layers_name = []


    View_Add = 0
    for name, layer in net_trans.named_modules():
        if isinstance(layer, nn.Conv2d):
            spike_layers.append(SpikeConv2d(layer.in_channels,
                                            layer.out_channels,
                                            layer.kernel_size,
                                            layer.stride,
                                            layer.padding,
                                            bias = False)#layer.bias)
                                            )
            spike_layers_name.append(name)
            # print(layer)
            # print(layer.bias)
        if isinstance(layer, nn.Linear):
            #判断何时添加View层
            if View == 1:
                View_Add += 1
                if View_Add == 1:
                    spike_layers.append(DAGViewOp([-1, layer.in_features]))
                    spike_layers_name.append("View")

            spike_layers.append(SpikeLinear(layer.in_features,
                                            layer.out_features,
                                            bias = False)#layer.bias)
                                            )  
            spike_layers_name.append(name)

    dict_net = net_trans.state_dict().copy()
    list_net_keys = list(net_trans.state_dict().keys())


    DAGView_after = 0
    for i in range(len(spike_layers)):
        if isinstance(spike_layers[i], DAGViewOp):
            print("jump DAGView's weight load")
            DAGView_after = 1
            continue

        if DAGView_after == 0:
            spike_layers[i].weight = nn.Parameter(dict_net[list_net_keys[i]])
            spike_layers[i].Vthr = torch.tensor([1.0],device=device)
            spike_layers[i].quant_base = torch.tensor([1.0],device=device)
        elif DAGView_after == 1:
            spike_layers[i].weight = nn.Parameter(dict_net[list_net_keys[i - 1]])
            spike_layers[i].Vthr = torch.tensor([1.0],device=device)
            spike_layers[i].quant_base = torch.tensor([1.0],device=device)

    dag=SpikeDAGModule()
    dag.inputs_nodes.append("dag_input")
    for i in range(len(spike_layers)):
        if i == 0:
            dag.add_op(spike_layers_name[i], spike_layers[i], ["dag_input"], [spike_layers_name[i] + "_out"])
        elif i == len(spike_layers) - 1:
            dag.add_op(spike_layers_name[i], spike_layers[i], [spike_layers_name[i - 1] + "_out"], [spike_layers_name[i] + "_out"])
            dag.outputs_nodes = (spike_layers_name[i] + "_out",)
        else:
            dag.add_op(spike_layers_name[i], spike_layers[i], [spike_layers_name[i - 1] + "_out"], [spike_layers_name[i] + "_out"])
    
    print(dag)
    return dag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net_name', type=str, help='network name')
    parser.add_argument('-sw','--snn_weight', type=str,
                        help='the location of the trained weights')
    parser.add_argument('-t','--timesteps', default=16, type=int)
    parser.add_argument('-b','--weight_bitwidth', default=4,
                        type=int, help='weight quantization bitwidth')
    parser.add_argument('--dataset', default='',
                        type=str, help='dataset name')
    parser.add_argument('-vr','--vth_round', action='store_true', default=False,
                        help='如果训练时vth是小数，请用这个')

    parser.add_argument('--pattern',default='1',type=str)
    parser.add_argument("--in_chip_x", default=1, help='add chip_x', type=int)
    parser.add_argument("--in_chip_y", default=1, help='add chip_y', type=int)
    parser.add_argument("--in_core_x", default=0, help='add core_x', type=int)
    parser.add_argument("--in_core_y", default=0, help='add core_y', type=int)
    parser.add_argument("--out_chip_x", default=0, help='out_chip_x', type=int)
    parser.add_argument("--out_chip_y", default=2, help='out_chip_y', type=int)

    args = parser.parse_args()
    weight_bitwidth = args.weight_bitwidth
# =======================================================================# 
    net_trans = SpikeNN()
    in_data_snn = torch.cat([in_data for _ in range(args.timesteps)], 0)
    in_data_test = SpikeTensor(in_data_snn, args.timesteps, 1)
    
    if args.snn_weight:
        print("model load: "+ args.snn_weight)
        dict_pth = torch.load(args.snn_weight, map_location='cuda:1')
        #stbp vth=0.3
        if args.vth_round:
            print("========== round your vth to int ==========")
            dict_net = net_trans.state_dict().copy()
            list_net_keys = list(net_trans.state_dict().keys())
            list_pth_keys = list(dict_pth.keys())

            # 测试新创建的网络与加载的pth的各层是否完全对应
            # print(list_net_keys)
            # print(list_pth_keys)

            for i in range(len(list_net_keys)):
                if i == 0:
                    print("jump the first conv layer")
                    continue
                dict_net[list_net_keys[i]] = dict_pth[list_pth_keys[i]] * 10
            
            net_trans.load_state_dict(dict_net)
        else:
            print("========== no round ==========")
            net_trans.load_state_dict(dict_pth)
    else:
        print("weight not load correct !")

    # 测试网络权重是否正确加载
    # for name, param in net_trans.named_parameters():
    #     if name == "fc1.weight":
    #         print(param)

# =======================================================================#
    init_quantize_net(net_trans, weight_bitwidth)
    #out_test = net_trans(in_data)
    Vths_rescaled = quantize_layers_final(weight_bitwidth, False, Vths)

    weightsdistribute(net_trans)

    snn = gen_snn(net_trans, View = View)
    snn_set_Vth(snn, Vths_rescaled)

    torch.save(in_data,f'{args.dataset}_inp_data.pth')
    #print(in_data)
    out_data = snn(in_data_test)
    #由于in_data不是spike_tensor，所以这里测试的输出会按照正常的Conv计算
    #由于权重最后都会int化，所以会看不出权重到底有没有*10，这是正常的
    print("=====dag SNN output: ============")
    print(out_data.to_float())
    print()
    torch.save(snn, "snn.pth")
    
    output_path = os.path.realpath(f"../output/{args.net_name}_T{args.timesteps}B{args.weight_bitwidth}P{args.pattern}_IN{args.in_chip_x},{args.in_chip_y}_OUT{args.out_chip_x},{args.out_chip_y}")
    os.system(f"rm -rf {output_path}")
    os.makedirs(output_path)

    print(f"mv snn.pth {output_path}/{args.net_name}.pth")
    print(f"mv {args.dataset}_inp_data.pth {output_path}/_inp_data.pth")
    os.system(f"mv snn.pth {output_path}/{args.net_name}.pth")
    os.system(f"mv {args.dataset}_inp_data.pth {output_path}/_inp_data.pth")