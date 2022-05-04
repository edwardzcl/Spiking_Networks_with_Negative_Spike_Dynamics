import torch
import argparse
import numpy as np
import os
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder,DatasetFolder
import torch.utils.data
import re
from tools.coco_dataset import COCOListDataset

class SetSplittor():
    def __init__(self,fraction=0.2):
        self.fraction=fraction
    
    def split(self,dataset):
        pass

class LoaderGenerator():
    """
    """
    def __init__(self,root,name,train_batch_size=1,test_batch_size=1,num_workers=0,kwargs={}):
        self.root=root
        self.name=str.lower(name)
        self.train_batch_size=train_batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        self.kwargs=kwargs
        self.items=[]
        self.train_set=None
        self.val_set=None
        self.test_set=None
        self.trainval_set=None
        self.train_loader_kwargs = {
            'num_workers': self.num_workers ,
            'pin_memory': kwargs.get('pin_memory',True),
            'drop_last':kwargs.get('drop_last',False)
            }
        self.test_loader_kwargs=self.train_loader_kwargs.copy()
        self.load()
    
    def split_train_val(self,splittor):
        self.trainval_set=self.train_set
        self.train_set,self.val_set=splittor.split(self.train_set)

    def load(self):
        pass

    def train_loader(self):
        assert self.train_set is not None
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True,  **self.train_loader_kwargs)
    
    def test_loader(self):
        assert self.test_set is not None
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False,  **self.test_loader_kwargs)
    
    def val_loader(self):
        assert self.val_set is not None
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.test_batch_size, shuffle=False,  **self.test_loader_kwargs)
    
    def trainval_loader(self):
        assert self.trainval_set is not None
        return torch.utils.data.DataLoader(self.trainval_set, batch_size=self.train_batch_size, shuffle=True,  **self.train_loader_kwargs)
        
class CIFARLoaderGenerator(LoaderGenerator):
    def load(self):
        if self.name=='cifar100':
            dataset_fn=datasets.CIFAR100
            normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                             std=[0.2673, 0.2564, 0.2762])
        elif self.name=='cifar10':
            dataset_fn=datasets.CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2470, 0.2435, 0.2616])
        else:
            raise NotImplementedError
        transform_train = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.train_set=dataset_fn(self.root, train=True, download=True, transform=transform_train)
        self.test_set=dataset_fn(self.root, train=False, transform=transform_test)

class COCOLoaderGenerator(LoaderGenerator):
    def load(self):
        self.train_set = COCOListDataset(self.root, augment=self.kwargs.get('coco_augment',True), multiscale=self.kwargs.get('coco_train_multiscale',True))
        self.test_set = COCOListDataset(self.root, img_size=self.kwargs.get('coco_test_img_size',416), augment=False, multiscale=False)
        self.train_loader_kwargs['collate_fn']=self.train_set.collate_fn
        self.test_loader_kwargs['collate_fn']=self.test_set.collate_fn

class DebugLoaderGenerator(LoaderGenerator):

    def load(self):
        version=re.findall("\d+",self.name)[0]
        class DebugSet(torch.utils.data.Dataset):
            def __getitem__(self,idx):
                if version=='0':
                    return torch.ones([1,4,4]),0
                if version=='1':
                    return torch.ones([1,8,8]),0
                if version=='2':
                    return torch.ones([1,1,1]),0
                if version=='3':
                    return torch.ones([1,3,3]),0
                if version == '4':
                    return torch.ones([3, 128, 128]), 0
                else:
                    raise NotImplementedError(f"version {version} of Debug dataset is not supported")
            def __len__(self): return 1
        self.train_set=DebugSet()
        self.test_set=DebugSet()

def get_dataset(args:argparse.Namespace):
    """ Preparing Datasets, args: 
        dataset (required): MNIST, cifar10/100, ImageNet, coco
        dataset_root: str, default='./datasets'
        num_workers: int
        batch_size: int
        test_batch_size: int
        val_fraction: float, default=0
        
    """
    dataset_name=str.lower(args.dataset)
    dataset_root=getattr(args,'dataset_root','./datasets') 
    num_workers=args.num_workers if hasattr(args,'num_workers') else 4
    batch_size=args.batch_size if hasattr(args,'batch_size') else 64
    test_batch_size=args.test_batch_size if hasattr(args,'test_batch_size') else batch_size
    val_fraction=args.val_fraction if hasattr(args,"val_fraction") else 0
    if args.dataset == "MNIST":
        train_transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])
        test_transfrom=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset_fn=datasets.MNIST
    elif "cifar" in dataset_name:
        # Data loading code
        g=CIFARLoaderGenerator(dataset_root,args.dataset,batch_size,test_batch_size,num_workers)
    elif "coco" in dataset_name:
        g=COCOLoaderGenerator(dataset_root,args.dataset,batch_size,test_batch_size,num_workers)
    elif "debug" in dataset_name:
        g=DebugLoaderGenerator(dataset_root,args.dataset,batch_size,test_batch_size,num_workers)
    return g.train_loader(),g.test_loader()
    # elif args.dataset=='ImageNet':
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #     transform_train = transforms.Compose([transforms.Scale(256),
    #                                              transforms.RandomCrop(224),
    #                                              transforms.RandomHorizontalFlip(),
    #                                              transforms.ToTensor(),
    #                                              transforms.Normalize(mean, std),
    #                                          ] )

    #     transform_test = transforms.Compose([transforms.Scale(256),
    #                                             transforms.CenterCrop(224),
    #                                             transforms.ToTensor(),
    #                                             transforms.Normalize(mean, std),
    #                                         ] )
    #     def dataset_fn(dataset_root,train,download=False,transform=None):
    #         if train:
    #             return ImageFolder(os.path.join(dataset_root,'train'), transform)
    #         else:
    #             return ImageFolder(os.path.join(dataset_root,'val'), transform)
    # elif 'debug' in args.dataset:
    #     version=re.findall('(\d+)$',args.dataset)[0]
    #     class DebugSet(torch.utils.data.Dataset):
    #         def __getitem__(self,idx):
    #             if version=='0':
    #                 return torch.ones([1,4,4]),0
    #             if version=='1':
    #                 return torch.ones([1,8,8]),0
    #             if version=='2':
    #                 return torch.ones([1,1,1]),0
    #             if version=='3':
    #                 return torch.ones([1,3,3]),0
    #             else:
    #                 raise NotImplementedError(f"version {version} of Debug dataset is not supported")
    #         def __len__(self): return 1
    #     loader=torch.utils.data.DataLoader(DebugSet(),1,False)
    #     return loader,loader,loader,loader
    # else:
    #     raise NotImplementedError

    # if not os.path.exists(dataset_root):
    #     os.makedirs(dataset_root)
    
    
    # rand_inds = np.arange(len(train_val_set))
    # np.random.seed(3)
    # np.random.shuffle(rand_inds)
    # val_set=copy.deepcopy(train_val_set)
    # train_set=copy.deepcopy(train_val_set)
    # n_val=int(len(train_set)*val_fraction)
    # if isinstance(train_val_set,DatasetFolder):
    #     train_set.samples=train_set.samples[n_val:]
    #     val_set.samples=val_set.samples[:n_val]
    # else:
    #     train_set.data=train_set.data[n_val:]
    #     val_set.data=val_set.data[:n_val]
    # train_set.targets=train_set.targets[n_val:]
    # val_set.targets=val_set.targets[:n_val]

    
    # return test_loader,val_loader,train_loader,train_val_loader

if __name__=='__main__':
    # Preparing the inp_data for analysis
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='CIFAR10',type=str,help='the location of the dataset')
    parser.add_argument('--inp_data_size', default=1, type=int,help='batchsize of input data for testing')
    parser.add_argument('--output_dir',default='./',type=str,help='output path of the inp_data.pth')
    args=parser.parse_args()

    train_loader,test_loader=get_dataset(args)
    inp_data=[test_loader.dataset[np.random.randint(len(test_loader.dataset))][0].unsqueeze(0) for _ in range(args.inp_data_size)]
    save_file=f'{args.output_dir}/{args.dataset}_inp_data.pth'
    print(f"saving to {save_file}")
    torch.save(torch.cat(inp_data,0),save_file)