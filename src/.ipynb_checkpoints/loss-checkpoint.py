import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class FocalLoss(nn.Module):
    """ Update BCE loss to deal with extreme class imbalance
    
    To address the class imbalance this loss reshapes the standard cross entropy loss 
    such that it down-weights the loss assigned to well-classified examples
    
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    
    Adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """
    def __init__(self, alpha=1, gamma=10, logits=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return F_loss.mean()


def dice_metric(pred, target, eps=1.0e-3):
    """Useful for dealing with class imbalance in segmentation problems, see Milletari et al. 2016.
    
    params:
        pred (torch.tensor): model predictions (Logits)
        targets (torch.tensor): target 
    """
    # Convert the raw output into probabilities
    pred = torch.sigmoid(pred)
    return ((2.0 * pred * target).sum() + eps) / ((pred + target).sum() + eps)


def dice_loss(inputs, target):
    return 1 - dice_metric(inputs, target)


class MixedLoss(nn.Module):
    """Combination of focal and dice loss"""
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, inputs, target):
        loss = self.alpha*self.focal(inputs, target) - torch.log(dice_metric(inputs, target))
        return loss.mean()
    

class MetricCollector:
    """Class to store and calculate loss metrics
    
    params:
        eval_losses (dict): dict contained loss metric name and loss function    
    """
    
    def __init__(self, eval_losses, set_label='Validation'):
        
        self.eval_losses = eval_losses
        self.set_label = set_label
        self.loss_names = self.eval_losses.keys()
        self.batch_losses, self.epoch_losses = {}, {}
        self.create_and_reset_lists()

    def batch_loss(self, outputs, targets):
        # Compute the loss metrics on the input batch.
        # All of the metrics stored in eval_losses are computed
        for loss_name, loss_fn in self.eval_losses.items():
            self.batch_losses[loss_name].append(loss_fn(outputs, targets).item())
    
    def epoch_loss(self, epoch_num):
        # At the end of an epoch we can aggregrate all of the batch level losses
        for loss_name in self.loss_names:
            epoch_loss = np.array(self.batch_losses[loss_name]).mean()
            print(f'\t {self.set_label} (epoch = {epoch_num}): {loss_name}={round(epoch_loss, 3)}')
            # Keep track of the epoch level stats
            self.epoch_losses[loss_name].append((epoch_num, epoch_loss))
        
        # Once the epoch level stats are computed we can reset the list of batch losses
        self.create_and_reset_lists(batch_only=True)
        
    def create_and_reset_lists(self, batch_only=False):
        # At the end of an epoch we need to remove all the stored
        # batch loss values
        for loss_name in self.loss_names:
            self.batch_losses[loss_name] = []
            if not batch_only:
                self.epoch_losses[loss_name] = []