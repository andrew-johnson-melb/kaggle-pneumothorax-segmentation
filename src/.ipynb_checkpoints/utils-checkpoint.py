import torch
import os
import numpy as np


def init_seed(seed=42):
    # Seed all computation to ensure results are deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def train_dev_split(label_df, train_frac=0.8):
    """Very basic train/validation split which keeps 'train_frac'
    for the training set. 
    
    Likely the training and val sets should be constructed using 
    stratified sampling.
    
    params:
        labels_df (pd.DataFrame): Dataframe containing metadata of images and mask
        train_frac (float [0,1]): fraction of data to use for the training set
    
    returns:
        training data (pd.DataFrame), testing data (pd.DataFrame)
        
    """
    train_df = label_df.sample(frac=train_frac,replace=False)
    val_df = label_df.drop(index=train_df.index)
    return train_df, val_df


def gen_upsampling_weights(train_df, ratio_pos_neg=1, pos_labels=['multiple-regions', 'single-region']):
    """ Add a new column 'weights' to train_df. 
    
    These weights can be used with the pytorch WeightedRandomSampler class to increase 
    the probablity of drawing a positive sample. This help the class imbalance problem in the dataset.
    
    params:
        train_df (pd.DataFrame): training dataset with class labels accessable via 'label'
        ratio_pos_neg (int): Desired ratio of positive to negative classes when drawing batch samples
        pos_labels (list[str]): Names of the positive labels
    
    returns:
        pd.DataFrame, with added 'weights' columns
        
    """
    # Create a new weights col which we will updated.
    train_df['weights'] = 1.0
    pos_bool =  train_df.label.isin(pos_labels).values
    num_pos = pos_bool.sum()
    num_neg = train_df.shape[0] - num_pos
    print(f'Number of negative examples = {num_neg}')
    print(f'Number of positive examples = {num_pos}')
    print(f'Fraction Neg Examples (raw data) = {round(num_neg / (num_pos + num_neg), 4)}')
    # Compute the amount we need to increase the prob of drawing 
    # positive samples by
    upsample_frac = (num_neg / num_pos) * ratio_pos_neg
    # Update the weight of the positive examples
    train_df.loc[pos_bool, 'weights'] = train_df.loc[pos_bool, 'weights'] * upsample_frac
    print(f'Upsampling amount = {round(upsample_frac,1)}')
    return train_df


def print_model_sizes(model, img):
    """ Display the size of the output data at each layer of the network
    
    params:
        model (pytorch model) 
        img (torch tensor): model input
    
    """
    print(f'Input size = {img.shape}')
    # Attach hooks to all of the model layers, this will
    # give as access to the output once a forward pass 
    # is made
    hookF = [Hook(model._modules[i]) for i in model._modules.keys()]
    _ = model(img)
    # Get a list of the layer names
    names = list(model._modules.keys())
    for ix, hook in enumerate(hookF):
        print(f'Layer = {ix} ({names[ix]}) Shape Out = {hook.output.shape}')  # Shape In = {hook.input.shape}' )
    # Remove hooks
    _ = [hook.close() for hook in hookF]

    
class Hook():
    # A simple hook class that returns the input and output of a layer during forward/backward pass
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def get_sample(dataloader):
    return next(iter(dataloader))

def check_tensors_equal(t1, t2):
    return torch.all(torch.eq(t1, t2))