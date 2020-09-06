import torch
import torchvision.models as models
from torch import nn
from utils import Hook
from torch.nn import functional as F


RESNET_LAYERS_TO_HOOK = {'layer1': '4', 'layer2': '5', 'layer3':'6', 'layer4': '7'}


def get_vision_model(model_name='resnet34', layers_to_detach=2, remove_head=True, freeze_params=True, pretrained=True):
    """ Helper function to load a pretrained model and remove it's head
    
    params:
        model_name (str): name of the vision model
        remove_head (bool): Whether or not to remove the head of the network
        layers_to_detach (int): The number of final layers to remove
            in the loaded network
        freeze_params (bool): Whether or not to freeze the models parameters
        pretrained (bool): Whether or not to use pretrained weights
    
    """
    encoder = getattr(models, model_name)(pretrained=pretrained)
    # We need to detach the head of this model
    # annoyingly we will lose the labels for the layers
    if remove_head:
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-layers_to_detach]))
    # Freeze the resnet parameters
    for layer in encoder.parameters():
        layer.requires_grad = not freeze_params
    return encoder
            

class ConvBlock(nn.Module):
    """Basic 3 by 3 convolution layer followed by a relu. Used within
    the Unets DecoderBlock"""
    #TODO add batch norm to this block
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        block = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)]
        block += [nn.ReLU()]
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        return self.block(x)
        
    
class DecoderBlock(nn.Module):
    """ Building block for the decoder slide of the Unet.
    
    Expected input is the residual connection (from encoder)
    and the decoder path data concatenated.
    
    The decoder block increases the size of the image using
    a transpose convolution. Prior to applying the transposed 
    convolution a standard 2D convolution is applied: this layer
    will create interactions between the residual connection data
    and the decoder data.
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        block = [ConvBlock(in_channels, middle_channels)]
        block += [nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,padding=1)]
        block += [nn.ReLU()]
        self.block = nn.Sequential(*block)
        
    def forward(self,x):
        return self.block(x)
    

class UNet(nn.Module):
    """ Unet with a reset for the encoder. Uses transposed convolution for
    the decoder."""

    def __init__(self, layers_to_hook=None, encoder_model='resnet34', pretrained=True, freeze_encoder=True):
        super().__init__()

        if layers_to_hook is None:
            layers_to_hook = RESNET_LAYERS_TO_HOOK

        self.encoder = get_vision_model(
            model_name=encoder_model, pretrained=pretrained, freeze_params=freeze_encoder)
        self.create_hooks(layers_to_hook)
        self.pool = nn.MaxPool2d(2, 2)
        # Hard coded for simplicity now
        # TODO: remove hard coded variables
        self.center_layer = DecoderBlock(512, 512, 256)
        self.up_block_1 = DecoderBlock(768, 512, 256)
        self.up_block_2 = DecoderBlock(512, 512, 256)
        self.up_block_3 = DecoderBlock(384, 256, 64)
        self.up_block_4 = DecoderBlock(128, 128, 128)
        self.up_block_5 = DecoderBlock(128, 128, 32)
        self.block_5_conv1 = ConvBlock(32, 32)
        self.block_5_conv2 = nn.Conv2d(32, 1, kernel_size=1)


    def create_hooks(self, layers_to_hook):
        self.hooks = [Hook(self.encoder._modules[layers_to_hook[l]]) for l in layers_to_hook.keys()]

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = self.center_layer(x)
        x = self.up_block_1(torch.cat((x, self.hooks[3].output), 1))
        x = self.up_block_2(torch.cat((x, self.hooks[2].output), 1))
        x = self.up_block_3(torch.cat((x, self.hooks[1].output), 1))
        x = self.up_block_4(torch.cat((x, self.hooks[0].output), 1))
        x = self.up_block_5(x)
        x = self.block_5_conv1(x)
        x = self.block_5_conv2(x)
        return x

    def close(self):
        for hk in self.hooks:
            hk.close()