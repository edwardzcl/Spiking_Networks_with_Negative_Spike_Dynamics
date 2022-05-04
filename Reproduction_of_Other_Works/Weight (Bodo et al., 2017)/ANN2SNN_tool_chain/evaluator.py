import torch
import numpy as np

class BasicEvaluator(object):
    def __call__(self,trainer):
        return -1

class L1Evaluator(BasicEvaluator):
    def __call__(self,trainer):
        evaluation=trainer.now_target-trainer.now_output
        return torch.mean(torch.abs(evaluation)).item()


class RelativeErrorEvaluator(BasicEvaluator):
    def __call__(self,trainer):
        evaluation=(trainer.now_target-trainer.now_output)/trainer.now_target
        return torch.mean(torch.abs(evaluation)).item()