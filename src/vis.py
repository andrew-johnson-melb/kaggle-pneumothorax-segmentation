import matplotlib.pylab as plt 
import torch
import numpy as np
from torchvision.transforms import Normalize


def visualize(image, mask, label='raw', fontsize = 18):
    """Plot an image with the mask overlaid. Should we used
    with raw data (not transformed) and channels should 
    be the last dim.
    """
    f, ax = plt.subplots(1, 3, figsize=(21, 8))
    ax[0].imshow(image)
    ax[0].set_title(f'Image {label}', fontsize=fontsize)
    ax[1].imshow(mask)
    ax[1].set_title(f'Mask {label}', fontsize=fontsize)
    ax[2].imshow(image)
    ax[2].imshow(mask, alpha=0.5)
    ax[2].set_title(f'Image + Mask {label}', fontsize=fontsize)
    return ax


def compare_masks(mask_pred, mask_target, label, fontsize=12):
    """ Plot the predicted and the target mask to asses how well
    the model is doing and to see where it is making mistakes.
    
    params:
        mask_pred (torch.tensor): 2D tensor of model predictions
        mask_target (torch.tensor): 2D tensor of target values
        label (str): plot index label
    """
    mask_pred = mask_pred.detach().numpy()
    mask_target = mask_target.detach().numpy()
    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(mask_pred)
    ax[0].set_title(f'Mask Pred {label}', fontsize=fontsize)
    ax[1].imshow(mask_target)
    ax[1].set_title(f'Mask Target {label}', fontsize=fontsize)
    plt.show()


def show_batch(train_dataloader, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Create a set of plots for a random batch showingthe image mask.
    
    Each call will generate a new set of examples
    This function will sample a batch and then undo the normalization 
    transform on the images. The other augmentations will be visible.
    
    This function can be useful for assessing if the data augmentations
    are reasonable. And for debugging.
    
    params:
        train_dataloader (pytorch dataloader): training dataloader
            should return image and mask
        mean: (list[float]): mean values used to normalize the raw images
        str: (list[float]): Standard deviation values used to normalizae the
            raw images
    
    returns:
        axis of image
    
    """

    # Get a sample batch
    img, mask = next(iter(train_dataloader))
    
    # Construct a inverse normalization transform to revert
    # the pixels to their pre-normalization values.
    invert_mean = list(-(np.array(mean) / np.array(std)))
    invert_std = list(1/np.array(std))
    inv_normalize = Normalize(mean=invert_mean, std=invert_std)

    mask = np.array(mask, dtype=int)
    
    # Transform the images:
    # Make sure the channel axis is last as this is expected by imshow.
    # Revert the normalization that is applied
    # on the transforms. imshow expects floats
    # in the range 0-1 for images. A few values will exceed this,
    # clip values outside 0-1 (avoids matplotlib complaining!)
    images = []
    # Loop over the batch
    for i in range(img.shape[0]):
        im = inv_normalize(img[i]).permute(1,2,0)
        im = torch.clamp(im, min=0.0, max=1.0)
        images.append(im)
    
    # Plot image and image with mask overlaid.
    # We will plot 4 images per row which will display
    # 2 images and 2 images with masks overlaid
    #TODO, fix variable naming
    col_num, row_num = 4, 2
    plot_row_num = int(img.shape[0] / row_num)
    f, ax = plt.subplots(plot_row_num, col_num, figsize=(21, 6*plot_row_num))
    batch_num = 0
    for i in range(plot_row_num):
        for j in [0,row_num]:
            image_tmp = images[i+j]
            mask_tmp = mask[i+j]
            ax[i,0+j].imshow(image_tmp)
            ax[i, 0+j].set_title(f'Image (batch num={batch_num})', fontsize=13)

            ax[i,1+j].imshow(image_tmp)
            ax[i,1+j].imshow(mask_tmp, alpha=0.5)
            ax[i,1+j].set_title(f'Image + Mask (batch={batch_num})', fontsize=13)
            batch_num += 1    
    return plt.gca()