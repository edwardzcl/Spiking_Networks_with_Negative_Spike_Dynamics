import models.example_net as example_net
import models.nopool_net as nopool_net
import models.debug_net as debug_net
import models.branch_net as branch_net
from models.yolov3 import Darknet
import models.example_group_net as example_group_net
import re
import torch
import models.test_net as test_net

def get_net_by_name(net_name):
    try:
        version=int(re.findall('(\d+)$',net_name)[0])
    except:
        version=0
    if 'example_small' in net_name:
        net=example_net.ExampleSmall()
    elif 'example_net' in net_name:
        net=getattr(example_net,f'ExampleNet{version}')()
    elif 'example_group_net' in net_name:
        net=getattr(example_group_net,f'ExampleGroupNet{version}')()
    elif 'debug_net' in net_name:
        net=getattr(debug_net,f'DebugNet{version}')()
    elif 'branch_net' in net_name:
        net=getattr(branch_net,f'BranchNet{version}')()
    elif 'yolov3_tiny' in net_name:
        net=Darknet('models/yolov3-tiny.cfg')
    elif 'nopool_net' in net_name:
        net=getattr(nopool_net,f'NoPoolNet{version}')()
    elif 'test_net' in net_name:
        net = getattr(test_net, f'TestNet{version}')()
    else:
        raise NotImplementedError
    return net


if __name__=='__main__':
    model=get_net_by_name('test_net')
    print(model)