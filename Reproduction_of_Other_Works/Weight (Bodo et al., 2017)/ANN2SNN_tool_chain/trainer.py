from tqdm import tqdm
import torch
from evaluator import BasicEvaluator
import numpy as np

class BasicIntervalProcessor(object):
    def __call__(self,trainer):
        pass

class BasicItemLogger():
    def __init__(self,name):
        self.name=name
        self.reset()
    
    def add(self,value):
        if isinstance(value,torch.Tensor):
            value=value.cpu().detach().numpy()
        self._add(value)

    def _add(self,value):
        pass

    def reset(self):
        pass

    def __repr__(self):
        return "{}".format(self.name)

class AvgItemLogger(BasicItemLogger):
    def _add(self,value):
        self.sum+=value
        self.count+=1
        self.avg=self.sum/self.count
    
    def reset(self):
        self.sum=0
        self.count=0
        self.avg=0
    
    def __repr__(self):
        return "{}:{}".format(self.name,self.avg)

class HistogramLogger(BasicItemLogger):
    def __init__(self,name):
        self.values=[]
        super().__init__(name)
        
    
    def histogram(self,bins=12):
        if len(self.values)==0:
            return [[],[]]
        hist=np.histogram(self.values,bins)
        return hist
    def reset(self):
        self.values.clear()
    def _add(self,value):
        self.values.append(value)

    def __repr__(self):
        hist=self.histogram()
        return "{}:hist:{}".format(self.name,hist)

class AvgStdItemLogger(BasicItemLogger):
    def __init__(self,name):
        pass


basic_log_items={'now_loss':AvgItemLogger,'now_evaluation':AvgItemLogger,'now_target':HistogramLogger,'now_output':HistogramLogger}
class BasicTrainerLogger(object):
    def __init__(self,print_interval=100,log_items=basic_log_items):
        self.print_interval=print_interval
        self.item_loggers={name:item_logger_class(name) for name,item_logger_class in log_items.items()}

    def log_iter(self,trainer):
        for name,item_logger in self.item_loggers.items():
            value=getattr(trainer,name)
            item_logger.add(value)
        if trainer.now_iter%self.print_interval==0:
            print("\t".join(str(item_logger) for item_logger in self.item_loggers.values()))
    
    def epoch_reset(self):
        for name,item_logger in self.item_loggers.items():
            item_logger.reset()

class BasicTrainer(object):
    """
    designed with Strategy design pattern
    """
    def __init__(self,model,optimizer,criterion,trainloader,evaluator=None,logger=None,
        before_inference_processor=None,before_loss_processor=None,before_update_processor=None):
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.trainloader=trainloader
        self.evaluator=BasicEvaluator() if evaluator is None else evaluator
        self.logger=BasicTrainerLogger() if logger is None else logger
        self.before_inference_processor=BasicIntervalProcessor() if before_inference_processor is None else before_inference_processor
        self.before_loss_processor=BasicIntervalProcessor() if before_loss_processor is None else before_loss_processor
        self.before_update_processor=BasicIntervalProcessor() if before_update_processor is None else before_update_processor
        self.now_iter=0
        self.now_data=None
        self.now_target=None
        self.now_output=None
        self.now_loss=None
        self.now_evaluation=None

    def run_iter(self):
        self.now_iter+=1
        self.optimizer.zero_grad()
        self.before_inference_processor(self)
        output=self.model(self.now_data)
        self.now_output=output
        self.before_loss_processor(self)
        loss=self.criterion(output,self.now_target)
        loss.backward()
        self.now_loss=loss
        evaluation=self.evaluator(self)
        self.now_evaluation=evaluation
        self.before_update_processor(self)
        self.optimizer.step()
        self.logger.log_iter(self)


    def run_epoch(self):
        self.model.train()
        for data,target in self.trainloader:
            self.now_data=data
            self.now_target=target
            self.run_iter()
            
        
    def reset_trainer(self):
        self.now_iter=0
        self.now_data=None
        self.now_target=None
        self.now_output=None
        self.now_loss=None
        self.now_evaluation=None
