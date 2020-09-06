import numpy as np
import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, loss_fn, data_loader, device, epoch, metric_collector=None):
    """ Train the input model for a single epoch"""
    #TODO: Add gradient clipping and gradient accumulation
    model.train()
    batch_losses = []
    for batch_ix, (images, targets) in enumerate(tqdm(data_loader)):
        # Put the batch on the gpu
        images, targets = images.to(device), targets.to(device)
        # Foward pass
        pred = model(images)
        pred = pred.squeeze()
        # This is the average loss per batch.
        loss = loss_fn(pred, targets)
        # Compute other loss metrics
        metric_collector.batch_loss(pred, targets)
        # This ignores the possibility of having imbalanced batches. But this doesn't 
        # occur much and so wont impact the results much.
        batch_losses.append(loss.item())
        # Could add gradient accumulation here.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = np.array(batch_losses).mean()
    print(f'Epoch={epoch}')
    print(f'\t Training loss = {round(epoch_loss, 3)}')
    metric_collector.epoch_loss(epoch_num=epoch)


@torch.no_grad()
def evaluate(model, data_loader, device, metric_collector, epoch):
    """Evaluate the input models loss on an entire epoch of data 
    
    The losses to evaluate are passed using the dict eval_losses.
    The dict 'eval_losses' containes a loss function are the dicts value
    and the loss functions name (e.g., BCE) for the dicts key. This is 
    crude and should be updated.
    """
    model.eval()
    for images, targets in tqdm(data_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        outputs = outputs.squeeze()
        metric_collector.batch_loss(outputs, targets)
    # Compute the epoch level stats
    metric_collector.epoch_loss(epoch_num=epoch)
