import ann_parser
import torch
from spike_layers import *


def is_layer_weighted_spike(layer):
    return isinstance(layer, SpikeConv2d) or isinstance(layer, SpikeLinear) or isinstance(layer,
                                                                                          SpikeConvTranspose2d) or isinstance(
        layer, SpikeAdd)


class SNNTransformer():
    def __init__(self, args, net, device):
        """
        The basic transformer to generate the snn for inference on cpu
        """
        self.original_net = net
        self.timesteps = args.timesteps
        self.relu_threshold = args.relu_threshold
        self.device = device
        self.snn_dag = None
        self.ann_snn_layer_mapping = {}
        self.reset_mode = args.reset_mode
        self.layer2name = {}
        self.input_status = {}
        self.output_status = {}
        self.input_generator = {}
        self.channel_wise = args.quantization_channel_wise

    def init_dag(self, inputs):
        """
        parse the network to get the snn DAG graph.
        """
        self.snn_dag = ann_parser.parse_ann_model(self.original_net, inputs)
        self.snn_dag.to(self.device)
        # register forward hook
        return self.snn_dag

    def _set_out_scale(self, train_loader, num_iters):
        """
        inference with original ANN to set the output scale
        """
        outs = {}
        dag = self.snn_dag
        for i, (data, target) in enumerate(train_loader):
            if i > num_iters:
                break
            dag.nodes[dag.inputs_nodes[0]] = data.to(self.device)  # assign tensor[b,c,h,w] as dag_input0
            for op_name, op in dag.ops.items():
                dag.do_operation(op_name)
                for node_name in op['out_nodes']:
                    out = dag.nodes[node_name].detach().clamp(0, self.relu_threshold)  # relu_thres
                    out_max = torch.quantile(out.view(out.size(0), -1), 0.99, dim=1).mean()
                    if outs.get(op_name, None) is None:
                        outs[op_name] = []
                    outs[op_name].append(out_max)  # for more than one out_nodes of new op
        for op_name, op in dag.ops.items():
            if is_layer_weighted_spike(op['op']):
                op['op'].out_scales.data[...] = sum(outs[op_name]) / len(outs[op_name])  # set out_scales within each spike layer

    def generate_snn(self, train_loader, num_iters):
        """
        generate the snn model.
        """

        datas = []  # copy input data into SpikeTensor
        num_iters = min(num_iters, len(train_loader))
        for i, (data, target) in enumerate(train_loader):
            if i > num_iters:
                break
            replica_data = torch.cat([data for _ in range(self.timesteps)], 0)
            data = SpikeTensor(replica_data, self.timesteps, scale_factor=1)
            datas.append(data)

        dag = self.init_dag([datas[0].to_float()])  # build snn_dag with the first batch data(SpikeTensor)
        self._set_out_scale(train_loader, num_iters)

        nodes_data_pool = {}  # store all nodes in the model
        nodes_data_pool[dag.inputs_nodes[0]] = datas

        for op_name, op in dag.ops.items():
            for node_name in op['out_nodes']:
                nodes_data_pool[node_name] = []
            delta_mem_potential_max = []
            # floating point calculation to set the SNN parameters
            if is_layer_weighted_spike(op['op']):
                for i in range(num_iters):
                    for in_node in op['in_nodes']:
                        dag.nodes[in_node] = nodes_data_pool[in_node][i].data * nodes_data_pool[in_node][i].scale_factor
                        dag.nodes[in_node] = dag.nodes[in_node].to(self.device)
                    dag.do_operation(op_name)
                    # Determine threshold-balancing factor according to maximum SNN activation
                    for node_name in op['out_nodes']:
                        b = dag.nodes[node_name].size(0)
                        delta_mem_potential_max.append(
                            torch.quantile(dag.nodes[node_name].detach().view(b, -1), 0.99, dim=1).mean())
                delta_mem_potential_max = sum(delta_mem_potential_max) / len(delta_mem_potential_max)
                # set the SNN parameters
                # delta_mem_potential_max=min(delta_mem_potential_max,self.relu_threshold)
                layer = op['op']
                layer.weight.data[...] = layer.weight.data
                layer.Vthr.data[...] = delta_mem_potential_max
                layer.out_scales.data[...] = delta_mem_potential_max
                layer.in_scales = nodes_data_pool[in_node][0].scale_factor.view(-1)[0].item()  # ?
                if hasattr(layer, 'bias'):
                    if layer.bias is None:
                        layer.bias = nn.Parameter(torch.Tensor([0] * layer.weight.size(0)).to(self.device))
                    new_bias = layer.bias.data + layer.Vthr / 2 / self.timesteps
                    layer.bias.data[...] = new_bias
                print(f"set {op_name}: Vthr {layer.Vthr} bias {layer.bias[:3].cpu()} in_scales {layer.in_scales}")
            # spiking dynamic to get the output
            for i in range(num_iters):
                for in_node in op['in_nodes']:
                    dag.nodes[in_node] = nodes_data_pool[in_node][i]
                    dag.nodes[in_node].data = dag.nodes[in_node].data.to(self.device)

                dag.do_operation(op_name)
                for node_name in op['out_nodes']:
                    nodes_data_pool[node_name].append(dag.nodes[node_name])
            # print(f"Debug dag.nodes[{in_node}].data {dag.nodes[in_node].data.max()}")
        return self.snn_dag


class V1ChipQuantSNNTransformer(SNNTransformer):
    def __init__(self, args, net, device):
        """
        The basic transformer to generate the snn for inference on V1.0 SNN chip
        """
        self.weight_bitwidth = args.weight_bitwidth
        assert self.weight_bitwidth <= 4
        self.codebook_bitwidth = 11
        self.leakage_bitwidth = 19
        self.Vthr_bitwidth = 18
        self.layer_scale_maps = {}
        self.input_scale_maps = {}
        self.bias_map = args.bias_map
        super().__init__(args, net, device)

    def generate_snn(self, train_loader, num_iters):
        """
        generate the snn model.
        """
        dag = super().generate_snn(train_loader, num_iters)
        print(f"Debug dag out {dag.outputs_nodes}")
        for op_name, op in dag.ops.items():
            if is_layer_weighted_spike(op['op']):
                layer = op['op']
                layer.weight.data[...] = layer.weight.data / layer.weight.alpha
                layer.Vthr[...] = layer.Vthr.data / layer.weight.alpha / layer.in_scales
                layer.out_scales.data[...] = layer.out_scales.data

                # layer.bias.data[...]=layer.bias.data/layer.weight.alpha/layer.in_scales

                weight_max_scale = torch.floor(
                    2 ** (self.codebook_bitwidth - self.weight_bitwidth) / layer.weight.data.max())
                Vthr_max_scale = torch.floor(2 ** (self.Vthr_bitwidth - 1) / layer.Vthr.data.max())
                map_scale = min(weight_max_scale, Vthr_max_scale)

                # layer.weight.data*=map_scale
                layer.weight.data = torch.round(layer.weight.data * map_scale)
                layer.Vthr.data = torch.round(layer.Vthr * map_scale)
                if hasattr(layer, 'bias'):
                    if self.bias_map == 'weight':
                        layer.bias.data[...] = torch.round(
                            layer.bias.data / layer.weight.alpha / layer.in_scales).clamp(
                            -2 ** (self.weight_bitwidth - 1), 2 ** (self.weight_bitwidth - 1) - 1)
                    elif self.bias_map == 'leakage':
                        layer.bias.data[...] = torch.round(
                            layer.bias.data / layer.weight.alpha / layer.in_scales * map_scale).clamp(
                            -2 ** (self.leakage_bitwidth - 1), 2 ** (self.leakage_bitwidth - 1) - 1)
                    else:
                        raise NotImplementedError()
                layer.bias.data *= map_scale
                layer.quant_base = map_scale

                layer.in_scales = None
                print(f"set {op_name}: Vthr {layer.Vthr} weight {layer.weight.detach().view(-1)[:3].cpu().numpy()}" +
                      f"bias {layer.bias[:3].detach().cpu().numpy()} out_scales {layer.out_scales[0].item()} quant_base {layer.quant_base}")
        return dag


if __name__ == '__main__':
    # just for debuging the function of the transformer
    import torch.nn as nn
    from spike_tensor import SpikeTensor
    import argparse

    args = argparse.Namespace(timesteps=16, relu_threshold=12, quantization_channel_wise=False,
                              reset_mode='subtraction', weight_bitwidth=4, )
    net = nn.Sequential(nn.Conv2d(2, 2, 3), nn.ReLU(), nn.Conv2d(2, 5, 1), nn.ReLU(), nn.Conv2d(5, 2, 3))
    net[0].weight.data.fill_(1.01)
    net[2].weight.data.fill_(1.01)
    net[0].weight.alpha = 0.5
    net[2].weight.alpha = 1
    net[4].weight.alpha = 10
    # net = nn.Sequential(nn.Conv2d(2, 2, 3), nn.ReLU())
    # net[0].weight.alpha=0.1
    t = SNNTransformer(args, net, torch.device('cpu'))

    input = torch.rand([1, 2, 8, 8])
    loader = [(input, torch.ones([4]))]
    t.inference_get_status(loader, 1)
    snn = t.generate_snn()
    replica_data = torch.cat([input for _ in range(args.timesteps)], 0)
    data = SpikeTensor(replica_data, args.timesteps, scale_factor=1)
    a_out = snn(data)
    print(a_out.to_float())

    # t = SNNTransformer(args, net, torch.device('cpu'))
    t = V1ChipQuantSNNTransformer(args, net, torch.device('cpu'))
    t.inference_get_status(loader, 1)
    v1snn = t.generate_snn()
    v1_out = v1snn(data)
    print(v1_out.to_float())
    print("Diff fraction", v1_out.to_float() / a_out.to_float())

    # print("==weight==",snn.conv1.weight,v1snn.conv1.weight)
    # print("==weight==",snn.conv2.weight,v1snn.conv2.weight)
    # print("==bias==",snn.conv2.bias,v1snn.conv2.bias)
    # print("==vthr==",snn.conv2.Vthr,v1snn.conv2.Vthr)
