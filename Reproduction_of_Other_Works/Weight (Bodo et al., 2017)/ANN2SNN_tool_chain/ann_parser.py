import numpy as np
import torch
import torch.nn.functional as F
from models import relu_threshold_shift
from spike_dag import *
from spike_layers import *

"""
1. Wrap the operations in ann with wrappers of different types, such as F.conv2d, F.linear and so on.
2. Run the network and trigger these wrapper.
3. Wrappers generate the nodes and edges in directed acyclic graph (DAG) when they are triggered.

`dag` is a global variable that can be accessed by all wrappers.
`find_node_by_tensor` is a function that used by wrappers, which finds the input nodes.
`WrappedTensor` is a wrapper for torch.Tensor to support the methods of Tensor instances, such as c=a+b, x.view and so on.

Each wrapper do the following steps:
1. find the input nodes
2. build a operation instance or some operations, and initialize the parameters in the operation
3. generate the output tensor (DO NOT use original function) and wrap the output tensor with WrappedTensor
4. add the node and operation(s) to the dag.

"""

dag = None  # init empty graph


def find_node_by_tensor(tensor):
    rst = [k for k, v in dag.nodes.items() if v is tensor]  # v and tensor are the same, not value ==
    if len(rst) == 0:
        raise ValueError("cannot find the tensor, Size: ", tensor.size())
    elif len(rst) > 1:
        raise ValueError("More than one node save the tensor, Size", tensor.size())
    return rst[0]


def find_op_by_out_node(node_name):
    """
    for bn fuse to find the conv ahead
    """
    for op_name, op in dag.ops.items():
        if node_name in op['out_nodes']:
            return op
    raise ValueError(f"cannot find op with out_nodes {node_name}")


view_count = 0


class WrappedTensor(torch.Tensor):
    def view(self, *size):
        global view_count
        view_count += 1
        op_name = f'view{view_count}'
        in_nodes = [find_node_by_tensor(self)]
        op = DAGViewOp(size)
        new_size = []
        for i in range(len(size)):
            if size[i] != -1:
                new_size.append(size[i])
            else:
                ori_num = np.prod([_ for _ in self.size()])
                other_num = np.prod([_ for _ in size if _ != -1])
                if ori_num % other_num != 0:
                    raise ValueError(f"view size mismatch {self.size()}->{size}")
                new_size.append(int(ori_num / other_num))
        out = torch.zeros(new_size).to(self.device)
        out = WrappedTensor(out)
        out_nodes = [f'{op_name}_out{1}']
        dag.add_op(op_name, op, in_nodes, out_nodes)
        dag.add_node(out_nodes[0], out)
        return out

    def __add__(self, other):
        out = add_wrapper(self, other)
        return out

    def __iadd__(self, other):
        raise NotImplementedError("not support += operation, please change to c=a+b ")


wrapped_functions = {}

conv2d_count = 0


def conv2d_wrapper(inp, weight, bias, stride, padding, dilation, groups):
    global conv2d_count
    conv2d_count += 1
    op_name = f'conv{conv2d_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeConv2d(inp.size(1), weight.size(0), weight.size()[2:], stride, padding, dilation, groups)
    op.weight = weight
    op.bias = bias
    # out=wrapped_functions["conv2d"](inp, weight, bias, stride, padding, dilation, groups)
    h = (inp.size(2) + padding[0] * 2 - dilation[0] * (weight.size(2) - 1) - 1) // stride[0] + 1
    w = (inp.size(3) + padding[1] * 2 - dilation[1] * (weight.size(3) - 1) - 1) // stride[1] + 1
    out = torch.zeros([inp.size(0), weight.size(0), h, w]).to(inp.device)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


conv_transpose2d_count = 0


def conv_transpose2d_wrapper(inp, weight, bias, stride, padding, out_padding, groups, dilation):
    global conv_transpose2d_count
    conv_transpose2d_count += 1
    op_name = f'conv_transpose2d{conv_transpose2d_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeConvTranspose2d(inp.size(1), weight.size(0), weight.size()[2:], stride, padding, out_padding, groups,
                              bias is not None, dilation)
    op.weight = weight
    op.bias = bias
    out = wrapped_functions["conv_transpose2d"](inp, weight, bias, stride, padding, out_padding, groups, dilation)
    # TODO change to fake data
    # out=torch.zeros()
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


linear_count = 0


def linear_wrapper(inp, weight, bias=None):
    global linear_count
    linear_count += 1
    op_name = f'fc{linear_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeLinear(inp.size(1), weight.size(0))
    op.weight = weight
    op.bias = bias
    # out = wrapped_functions["linear_wrapper"](inp.size(0), weight.size(0))
    out = torch.zeros([inp.size(0), weight.size(0)]).to(inp.device)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


relu_count = 0


def relu_wrapper(inp, inplace=False):
    """
    default inplace=False to avoid branch reuse error
    """
    global relu_count
    relu_count += 1
    op_name = f'relu{relu_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeReLU(None)
    # out = wrapped_functions["relu_wrapper"](inp)
    out = torch.zeros(inp.size()).to(inp.device)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def relu_thresh_shift_wrapper(inp, thresh, shift):
    global relu_count
    relu_count += 1
    op_name = f'relu{relu_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeReLU(thresh)
    # out = wrapped_functions["relu_thresh_shift_wrapper"](inp, thresh, shift)
    out = torch.zeros(inp.size()).to(inp.device)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


avg_pool2d_count = 0


def avg_pool2d_wrapper(inp, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                       divisor_override=None):
    """
    replace avg_pool2d by fixed weight & no bias depthwise conv2d
    """
    global avg_pool2d_count
    avg_pool2d_count += 1
    op_name = f'avg_pool2d{avg_pool2d_count}'
    in_nodes = [find_node_by_tensor(inp)]
    # op=SpikeAvgPool2d(kernel_size,stride,padding)
    op = SpikeConv2d(inp.size(1), inp.size(1), kernel_size, stride, padding, groups=inp.size(1), bias=False)
    op.weight.data[...] = 1 / (op.kernel_size[0]) ** 2  # assign pure tensor without influencing para's grad
    op.weight.alpha = 1 / (op.kernel_size[0]) ** 2  # !
    # out = wrapped_functions["avg_pool2d_wrapper"](inp, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    assert ceil_mode == False
    if isinstance(padding, int):
        padding = [padding, padding]
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = [stride, stride]
    h = (inp.size(2) + padding[0] * 2 - (kernel_size[0] - 1) - 1) // stride[0] + 1
    w = (inp.size(3) + padding[1] * 2 - (kernel_size[1] - 1) - 1) // stride[1] + 1
    out = torch.zeros([inp.size(0), inp.size(1), h, w]).to(inp.device)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


batch_norm_count = 0


def batch_norm_wrapper(inp, running_mean, running_var, weight=None, bias=None,
                       training=False, momentum=0.1, eps=1e-5):
    """
    fuse into the conv, not add bn op
    e.g. in SpikeConv2d, if self.bn, calculate bn after conv2d
    """
    global batch_norm_count
    batch_norm_count += 1
    in_nodes = [find_node_by_tensor(inp)]
    in_op = find_op_by_out_node(in_nodes[0])
    if not (isinstance(in_op['op'], SpikeConv2d) or isinstance(in_op['op'], SpikeConvTranspose2d)):
        raise ValueError(
            f"Conv2d/ConvTranspose2d is expected before BatchNorm, but {type(in_op['op'])} found. \n {in_op}")
    bn = nn.BatchNorm2d(weight.size(0), eps, momentum, True).to(inp.device)
    bn.eval()
    bn.weight.data = weight.data
    bn.bias.data = bias.data
    bn.running_mean.data[...] = running_mean
    bn.running_var.data[...] = running_var
    # print(f"Debug ann_parser.batch_norm_wrapper\n  running mean {running_mean} \n running_var {running_var}")
    in_op['op'].bn = bn
    # out=wrapped_functions['batch_norm'](inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
    out = torch.zeros(inp.size()).to(inp.device)
    out = WrappedTensor(out)
    dag.nodes[in_nodes[0]] = out
    return out


add_count = 0


def add_wrapper(a, b):
    # fuse into the conv
    global add_count
    add_count += 1
    in_nodes = [find_node_by_tensor(a), find_node_by_tensor(b)]
    op_name = f"add{add_count}"
    op = SpikeAdd(a.size(1))
    # out = wrapped_functions["add"](a, b)
    x = torch.ones_like(a) + torch.ones_like(b)
    out = torch.zeros(x.size()).to(a.device)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


concat_count = 0


def concat_wrapper(tensors, dim=None):
    global concat_count
    concat_count += 1
    op_name = f"concat{concat_count}"
    in_nodes = [find_node_by_tensor(tensor) for tensor in tensors]
    op = ConcatOp(dim=dim)
    out = wrapped_functions['concat'](tensors, dim)
    # TODO change to fake data
    # out=torch.zeros()
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def wrap():
    """
    refactor nn.functional or other function
    """
    raw = F.conv2d  # 1
    wrapped_functions["conv2d"] = raw
    F.conv2d = conv2d_wrapper

    raw = F.conv_transpose2d  # 2
    wrapped_functions["conv_transpose2d"] = raw
    F.conv_transpose2d = conv_transpose2d_wrapper

    raw = F.linear  # 3
    wrapped_functions["linear_wrapper"] = raw
    F.linear = linear_wrapper

    raw = F.relu  # 4
    wrapped_functions["relu_wrapper"] = raw
    F.relu = relu_wrapper

    raw = relu_threshold_shift.relu_thresh_shift  # 5
    wrapped_functions["relu_thresh_shift_wrapper"] = raw
    relu_threshold_shift.relu_thresh_shift = relu_thresh_shift_wrapper

    raw = F.avg_pool2d
    wrapped_functions["avg_pool2d_wrapper"] = raw
    F.avg_pool2d = avg_pool2d_wrapper

    raw = torch.cat
    wrapped_functions["concat"] = raw
    torch.cat = concat_wrapper

    raw = F.batch_norm
    wrapped_functions["batch_norm"] = raw
    F.batch_norm = batch_norm_wrapper

    raw = torch.add
    wrapped_functions["add"] = raw
    torch.add = add_wrapper
    # raw=F.max_pool2d
    # wrapped_functions[avg_pool2d_wrapper]=raw
    # F.max_pool2d=avg_pool2d_wrapper


def unwrap():
    """
    refactor nn.functional or other function back
    """
    F.conv2d = wrapped_functions["conv2d"]
    F.conv_transpose2d = wrapped_functions["conv_transpose2d"]
    F.linear = wrapped_functions["linear_wrapper"]
    F.relu = wrapped_functions["relu_wrapper"]
    relu_threshold_shift.relu_thresh_shift = wrapped_functions["relu_thresh_shift_wrapper"]
    # F.leaky_relu=wrapped_functions[relu_wrapper]
    F.avg_pool2d = wrapped_functions["avg_pool2d_wrapper"]
    # F.max_pool2d=wrapped_functions[avg_pool2d_wrapper]
    torch.cat = wrapped_functions["concat"]
    F.batch_norm = wrapped_functions['batch_norm']
    torch.add = wrapped_functions["add"]


def parse_ann_model(model, inputs):
    global dag
    dag = SpikeDAGModule()  # init empty dag
    model.eval()
    model.cpu()
    wrapped_input = []
    for i, x in enumerate(inputs):
        inp = WrappedTensor(x.cpu())
        wrapped_input.append(inp)
        name = f'dag_input{i}'
        dag.nodes[name] = inp
        dag.inputs_nodes.append(name)
    # print([k for k,v in dag.nodes.items() if v in inputs])
    wrap()
    model(*wrapped_input)  # run once to build dag. tensor no need to be true but right size, afterwards to be cleared.
    unwrap()
    dag.clear_nodes()
    dag.outputs_nodes = dag.find_end_nodes()
    return dag


if __name__ == '__main__':
    o = torch.ones([3, 3, 3])
    o = WrappedTensor(o)
    print(o)
